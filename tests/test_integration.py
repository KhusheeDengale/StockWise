"""End-to-end integration tests verifying dataflow from history through forecasting, fuzzy risk, optimization, and reports.
"""
import numpy as np
import pandas as pd
import pytest

from core.forecast_service import ForecastService
from core.fuzzy_engine import evaluate_fuzzy_urgency
from core.inventory import validate_inventory_config
from core.recommendation_service import generate_recommendations
from core.schemas import ForecastResult


@pytest.fixture
def mock_dataset_and_inventory():
    """Create a realistic small deterministic environment with 2 products."""
    rng = np.random.RandomState(42)
    n = 100
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    
    # Product 1: High demand
    p1_units = 25.0 + 5.0 * np.sin(np.arange(n) * (2 * np.pi / 7)) + rng.normal(0, 1, n)
    p1_df = pd.DataFrame({"Date": dates, "DailyUnits": np.clip(p1_units, 0, None), "StockCode": "ITEM_A"})

    # Product 2: Lower demand
    p2_units = 8.0 + 2.0 * np.cos(np.arange(n) * (2 * np.pi / 7)) + rng.normal(0, 0.5, n)
    p2_df = pd.DataFrame({"Date": dates, "DailyUnits": np.clip(p2_units, 0, None), "StockCode": "ITEM_B"})

    inv_df = pd.DataFrame([
        {
            "StockCode": "ITEM_A",
            "Description": "Item A Fast Mover",
            "CurrentStock": 15,
            "SupplierCostGBP": 4.50,
            "LeadTimeDays": 4,
            "StorageCapacity": 300,
            "MinimumOrderQuantity": 0,
            "OrderMultiple": 1,
        },
        {
            "StockCode": "ITEM_B",
            "Description": "Item B Steady Mover",
            "CurrentStock": 50,
            "SupplierCostGBP": 8.00,
            "LeadTimeDays": 5,
            "StorageCapacity": 200,
            "MinimumOrderQuantity": 0,
            "OrderMultiple": 1,
        },
    ])
    validate_inventory_config(inv_df)
    return p1_df, p2_df, inv_df


def test_end_to_end_pipeline(mock_dataset_and_inventory):
    """Verify that historical data feeds forecasting -> fuzzy risk -> optimizer -> feasible recommendations."""
    p1_df, p2_df, inv_df = mock_dataset_and_inventory
    service = ForecastService()

    # 1. Forecast both products
    fc1 = service.forecast_product(p1_df, "ITEM_A", horizon=14, product_description="Item A Fast Mover")
    fc2 = service.forecast_product(p2_df, "ITEM_B", horizon=14, product_description="Item B Steady Mover")
    assert fc1.selected_model in ["Seasonal Naive", "Moving Average (7-day)", "Linear Regression", "ARIMA"]
    assert len(fc1.daily_predictions) == 14
    assert len(fc2.daily_predictions) == 14

    # 2. Run recommendations with £500 budget
    fc_dict = {"ITEM_A": fc1, "ITEM_B": fc2}
    report = generate_recommendations(fc_dict, inv_df, budget_gbp=500.0, planning_horizon=14)

    assert len(report.recommendations) == 2
    assert report.ga_metrics.Feasible is True
    assert report.ga_metrics.TotalPurchaseCost <= 500.0
    # Invariant: GA <= greedy
    assert report.ga_metrics.ObjectiveValue <= report.greedy_metrics.ObjectiveValue + 1e-6

    # Verify recommendations data structure
    rec_a = next(r for r in report.recommendations if r.StockCode == "ITEM_A")
    assert rec_a.RecommendedQuantity >= 0
    assert rec_a.ExpectedShortageAfter <= rec_a.ExpectedShortageBefore + 1e-4


def test_changing_stock_alters_fuzzy_risk_and_order(mock_dataset_and_inventory):
    """Verify that altering CurrentStock changes Coverage Ratio, Fuzzy Risk, and Recommended Quantity."""
    p1_df, _, inv_df = mock_dataset_and_inventory
    service = ForecastService()
    fc1 = service.forecast_product(p1_df, "ITEM_A", horizon=14)

    # Low stock case (stock = 5)
    inv_low = inv_df.copy()
    inv_low.loc[inv_low["StockCode"] == "ITEM_A", "CurrentStock"] = 5
    rep_low = generate_recommendations({"ITEM_A": fc1}, inv_low, budget_gbp=300.0, planning_horizon=14)
    rec_low = rep_low.recommendations[0]

    # Ample stock case (stock = 250)
    inv_high = inv_df.copy()
    inv_high.loc[inv_high["StockCode"] == "ITEM_A", "CurrentStock"] = 250
    rep_high = generate_recommendations({"ITEM_A": fc1}, inv_high, budget_gbp=300.0, planning_horizon=14)
    rec_high = rep_high.recommendations[0]

    # Contract Section 20: Changing stock changes dependent calculations
    assert rec_low.RiskScore > rec_high.RiskScore
    assert rec_low.ExpectedShortageBefore > rec_high.ExpectedShortageBefore
    assert rec_high.ExpectedEndingStock >= rec_low.ExpectedEndingStock
    assert rep_low.ga_metrics.Feasible is True
    assert rep_high.ga_metrics.Feasible is True


def test_changing_budget_scales_feasible_purchases(mock_dataset_and_inventory):
    """Verify that expanding budget allows greater purchases and reduces shortages."""
    p1_df, p2_df, inv_df = mock_dataset_and_inventory
    service = ForecastService()
    fc1 = service.forecast_product(p1_df, "ITEM_A", horizon=14)
    fc2 = service.forecast_product(p2_df, "ITEM_B", horizon=14)
    fc_dict = {"ITEM_A": fc1, "ITEM_B": fc2}

    # Tight budget: £50
    rep_tight = generate_recommendations(fc_dict, inv_df, budget_gbp=50.0, planning_horizon=14)
    assert rep_tight.ga_metrics.TotalPurchaseCost <= 50.0

    # Generous budget: £1000
    rep_generous = generate_recommendations(fc_dict, inv_df, budget_gbp=1000.0, planning_horizon=14)
    assert rep_generous.ga_metrics.TotalPurchaseCost <= 1000.0

    # Expected shortage with generous budget must be <= shortage with tight budget
    assert rep_generous.ga_metrics.TotalExpectedShortage <= rep_tight.ga_metrics.TotalExpectedShortage + 1e-4
