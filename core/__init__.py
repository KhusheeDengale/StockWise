"""StockWise Core Backend Public API.

Exposes the 13 required high-level service functions for computational workflows and UI integration.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd

from config import (
    CLEANED_TRANSACTIONS_PARQUET,
    CLEANING_AUDIT_JSON,
    DAILY_DEMAND_PARQUET,
    DEMO_PRODUCTS_CSV,
    PRODUCT_CATALOG_CSV,
    RAW_RETAIL_EXCEL,
    SAMPLE_INVENTORY_CSV,
)
from core.features import build_supervised_features, select_features_for_regression
from core.forecast_evaluation import compare_and_select_models
from core.forecast_service import ForecastService
from core.fuzzy_engine import evaluate_fuzzy_urgency
from core.inventory import (
    load_inventory_config,
    reset_demo_inventory,
    save_inventory_config,
    validate_inventory_config,
)
from core.preprocessing import prepare_dataset
from core.recommendation_service import generate_recommendations
from core.reporting import export_results_csv, format_recommendation_summary_table
from core.schemas import (
    ForecastResult,
    FuzzyResult,
    RecommendationReport,
    SchemaValidationError,
)
from core.statistics import (
    calculate_acf_pacf,
    calculate_correlations,
    calculate_rolling_statistics,
    calculate_sales_summary,
    decompose_time_series,
    describe_numeric,
    perform_adf_test,
)

# Global in-memory cache for fast UI retrieval
_DATA_CACHE: Dict[str, Any] = {}
_FORECAST_SERVICE = ForecastService()


def initialize_project(force_recompute: bool = False) -> Dict[str, Any]:
    """1. Ensure processed datasets and inventory configuration exist, loading them into memory."""
    if (
        not force_recompute
        and CLEANED_TRANSACTIONS_PARQUET.exists()
        and DAILY_DEMAND_PARQUET.exists()
        and PRODUCT_CATALOG_CSV.exists()
        and CLEANING_AUDIT_JSON.exists()
    ):
        daily_df = pd.read_parquet(DAILY_DEMAND_PARQUET)
        catalog_df = pd.read_csv(PRODUCT_CATALOG_CSV, dtype={"StockCode": str})
        demo_df = (
            pd.read_csv(DEMO_PRODUCTS_CSV, dtype={"StockCode": str})
            if DEMO_PRODUCTS_CSV.exists()
            else catalog_df.head(30)
        )
        with open(CLEANING_AUDIT_JSON, "r", encoding="utf-8") as f:
            audit = json.load(f)
    else:
        _, _, daily_df, catalog_df, audit = prepare_dataset()
        demo_df = pd.read_csv(DEMO_PRODUCTS_CSV, dtype={"StockCode": str})

    # Check sample inventory configuration
    if not SAMPLE_INVENTORY_CSV.exists() or force_recompute:
        inv_df = reset_demo_inventory(demo_df, daily_df)
    else:
        inv_df = load_inventory_config()

    _DATA_CACHE["daily_demand"] = daily_df
    _DATA_CACHE["product_catalog"] = catalog_df
    _DATA_CACHE["demo_products"] = demo_df
    _DATA_CACHE["inventory_config"] = inv_df
    _DATA_CACHE["cleaning_audit"] = audit

    return {
        "status": "ready",
        "unique_products": len(catalog_df),
        "demo_products_count": len(demo_df),
        "total_daily_records": len(daily_df),
        "date_range": (str(daily_df["Date"].min()), str(daily_df["Date"].max())),
    }


def get_dataset_summary(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """2. Return cleaning audit JSON and transaction-level sales summary."""
    if "cleaning_audit" not in _DATA_CACHE:
        initialize_project()

    audit = _DATA_CACHE.get("cleaning_audit", {})
    if CLEANED_TRANSACTIONS_PARQUET.exists():
        cleaned_tx = pd.read_parquet(CLEANED_TRANSACTIONS_PARQUET)
        sales_sum = calculate_sales_summary(cleaned_tx, filters=filters)
    else:
        sales_sum = {}

    return {
        "audit": audit,
        "sales_summary": sales_sum,
    }


def get_product_catalog(demo_only: bool = False) -> pd.DataFrame:
    """3. Retrieve full product catalog or demo products subset."""
    if "product_catalog" not in _DATA_CACHE:
        initialize_project()
    return _DATA_CACHE["demo_products"] if demo_only else _DATA_CACHE["product_catalog"]


def get_product_daily_history(stock_code: str) -> pd.DataFrame:
    """4. Return complete calendar daily demand series for a specific product."""
    if "daily_demand" not in _DATA_CACHE:
        initialize_project()
    daily_df = _DATA_CACHE["daily_demand"]
    prod_history = daily_df[daily_df["StockCode"] == str(stock_code)].sort_values("Date").reset_index(drop=True)
    return prod_history


def get_statistical_analysis(stock_code: str) -> Dict[str, Any]:
    """5. Calculate comprehensive descriptive and time-series statistics for a product."""
    prod_df = get_product_daily_history(stock_code)
    if prod_df.empty:
        raise ValueError(f"No daily history found for StockCode '{stock_code}'.")

    series = prod_df["DailyUnits"]
    descriptive = describe_numeric(series)
    rolling = calculate_rolling_statistics(series)
    adf = perform_adf_test(series)
    acf_pacf = calculate_acf_pacf(series)
    decomp = decompose_time_series(series)

    # Price vs demand correlation if transaction columns present
    corr = calculate_correlations(prod_df, ["DailyUnits", "DailyRevenueGBP", "AverageSellingPriceGBP"])

    return {
        "stock_code": stock_code,
        "descriptive_statistics": descriptive,
        "rolling_statistics": rolling.to_dict(orient="list"),
        "adf_test": adf,
        "acf_pacf": acf_pacf,
        "decomposition": decomp,
        "correlations": corr,
    }


def get_feature_analysis(stock_code: str, horizon: int = 14) -> Dict[str, Any]:
    """6. Generate supervised feature definitions and feature selection results for a product."""
    prod_df = get_product_daily_history(stock_code)
    features, target, definitions = build_supervised_features(prod_df["DailyUnits"], prod_df["Date"])
    
    # Run backtest to obtain feature selection on train/val
    eval_res = compare_and_select_models(prod_df, horizon=horizon)

    return {
        "stock_code": stock_code,
        "feature_definitions": definitions,
        "feature_selection": eval_res["feature_selection"],
        "supervised_sample": features.head(10).to_dict(orient="records"),
    }


def compare_forecasting_models(stock_code: str, horizon: int = 14) -> Dict[str, Any]:
    """7. Compare the four forecasting models using chronological non-overlapping validation blocks."""
    prod_df = get_product_daily_history(stock_code)
    return compare_and_select_models(prod_df, horizon=horizon)


def forecast_product(
    stock_code: str,
    horizon: int = 14,
    min_lead_time_days: int = 0,
) -> ForecastResult:
    """8. Evaluate, select winner, refit on all history, and forecast future demand for one product."""
    prod_df = get_product_daily_history(stock_code)
    catalog = get_product_catalog()
    desc_match = catalog[catalog["StockCode"] == str(stock_code)]
    desc = str(desc_match["Description"].iloc[0]) if not desc_match.empty else "UNKNOWN"

    return _FORECAST_SERVICE.forecast_product(
        daily_df=prod_df,
        stock_code=stock_code,
        horizon=horizon,
        product_description=desc,
        min_lead_time_days=min_lead_time_days,
    )


def forecast_products(
    stock_codes: List[str],
    horizon: int = 14,
) -> Dict[str, ForecastResult]:
    """9. Batch forecast multiple products."""
    results = {}
    for code in stock_codes:
        results[code] = forecast_product(code, horizon=horizon)
    return results


def get_fuzzy_risk(
    stock_code: str,
    forecast: ForecastResult,
    inventory_row: Union[pd.Series, Dict[str, Any]],
    horizon: int = 14,
) -> FuzzyResult:
    """10. Compute Mamdani fuzzy reorder urgency for a product."""
    lt_days = int(inventory_row["LeadTimeDays"])
    curr_stock = int(inventory_row["CurrentStock"])
    extended_forecast = forecast.model_parameters.get(
        "extended_lead_time_predictions", forecast.daily_predictions
    )
    return evaluate_fuzzy_urgency(
        current_stock=curr_stock,
        lead_time_days=lt_days,
        planning_horizon_days=horizon,
        lead_time_forecast=extended_forecast,
        uncertainty_score=forecast.uncertainty_score,
    )


def optimize_reorders(
    stock_codes: List[str],
    budget_gbp: float,
    horizon: int = 14,
    inventory_config: Optional[pd.DataFrame] = None,
    run_stability_check: bool = False,
) -> RecommendationReport:
    """11. Complete purchase optimization across selected products given a budget and horizon."""
    if inventory_config is None:
        if "inventory_config" not in _DATA_CACHE:
            initialize_project()
        inventory_config = _DATA_CACHE["inventory_config"]

    # Subset config
    cfg_sub = inventory_config[inventory_config["StockCode"].astype(str).isin(stock_codes)].copy()
    if cfg_sub.empty:
        raise ValueError(f"None of the provided stock_codes {stock_codes} were found in inventory config.")

    # Forecast each product
    fc_dict: Dict[str, ForecastResult] = {}
    for _, row in cfg_sub.iterrows():
        code = str(row["StockCode"])
        lt_days = int(row["LeadTimeDays"])
        fc_dict[code] = forecast_product(code, horizon=horizon, min_lead_time_days=lt_days)

    return generate_recommendations(
        forecast_results=fc_dict,
        inventory_config=cfg_sub,
        budget_gbp=budget_gbp,
        planning_horizon=horizon,
        run_stability_check=run_stability_check,
    )


def generate_recommendation_report(report: RecommendationReport) -> Dict[str, Any]:
    """12. Generate formatted tabular summaries and baseline comparison tables."""
    baseline_df = format_recommendation_summary_table(report)
    return {
        "summary_metrics": report.summary_metrics,
        "baseline_comparison_table": baseline_df.to_dict(orient="records"),
        "recommendations_count": len(report.recommendations),
    }


# Export core symbols
__all__ = [
    "initialize_project",
    "get_dataset_summary",
    "get_product_catalog",
    "get_product_daily_history",
    "get_statistical_analysis",
    "get_feature_analysis",
    "compare_forecasting_models",
    "forecast_product",
    "forecast_products",
    "get_fuzzy_risk",
    "optimize_reorders",
    "generate_recommendation_report",
    "export_results_csv",
    "load_inventory_config",
    "save_inventory_config",
    "reset_demo_inventory",
    "validate_inventory_config",
]
