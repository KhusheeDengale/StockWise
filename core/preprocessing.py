"""Preprocessing pipeline, data cleaning, and daily aggregation for retail transactions.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import (
    CLEANED_TRANSACTIONS_PARQUET,
    CLEANING_AUDIT_JSON,
    DAILY_DEMAND_PARQUET,
    DEMO_CRITERIA,
    DEMO_PRODUCTS_CSV,
    EXCLUDED_TRANSACTIONS_PARQUET,
    NON_PRODUCT_DESCRIPTIONS,
    NON_PRODUCT_STOCKCODES,
    PRODUCT_CATALOG_CSV,
    PROCESSED_DATA_DIR,
    RAW_RETAIL_EXCEL,
)
from core.data_loader import load_raw_dataset, normalize_column_names


def identify_non_merchandise(
    stock_code: Any, description: Any
) -> bool:
    """Determine whether a transaction line corresponds to non-merchandise fees/services.
    
    Args:
        stock_code: Stock code string or object.
        description: Description string or object.
        
    Returns:
        True if item is non-merchandise (e.g., postage, manual adjustment, bank charge).
    """
    code_str = str(stock_code).strip().upper() if pd.notna(stock_code) else ""
    desc_str = str(description).strip().upper() if pd.notna(description) else ""

    if code_str in NON_PRODUCT_STOCKCODES:
        return True

    for pattern in NON_PRODUCT_DESCRIPTIONS:
        if pattern == desc_str or pattern in desc_str:
            return True

    return False


def clean_transactions(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute the exact sequential data cleaning rules on raw transactions.
    
    Args:
        raw_df: Combined raw transactions dataframe.
        
    Returns:
        Tuple of (cleaned_transactions, excluded_transactions, audit_dict).
    """
    df = raw_df.copy()
    raw_rows = len(df)
    
    # 1. Type conversions
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    
    # Format strings
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    # Normalize common 'nan' string representation from missing
    df.loc[df["Description"].str.lower() == "nan", "Description"] = np.nan
    
    # 2. Duplicate detection (excluding SourceSheet)
    cols_for_dup = [c for c in df.columns if c != "SourceSheet"]
    dup_mask = df.duplicated(subset=cols_for_dup, keep="first")
    duplicate_rows = int(dup_mask.sum())
    
    # Track reasons for exclusion
    exclusion_reasons: List[List[str]] = [[] for _ in range(len(df))]
    
    for idx, is_dup in enumerate(dup_mask):
        if is_dup:
            exclusion_reasons[idx].append("DUPLICATE_ROW")
            
    # Invoice cancellation check
    inv_str = df["InvoiceNo"].astype(str).str.strip()
    cancel_mask = inv_str.str.startswith(("C", "c"))
    cancelled_rows = int(cancel_mask.sum())
    for idx, is_cancel in enumerate(cancel_mask):
        if is_cancel:
            exclusion_reasons[idx].append("CANCELLED_INVOICE")
            
    # Quantity <= 0
    nonpos_qty_mask = df["Quantity"].isna() | (df["Quantity"] <= 0)
    nonpositive_quantity_rows = int(nonpos_qty_mask.sum())
    for idx, is_nonpos in enumerate(nonpos_qty_mask):
        if is_nonpos:
            exclusion_reasons[idx].append("NON_POSITIVE_QUANTITY")
            
    # UnitPrice <= 0
    nonpos_price_mask = df["UnitPrice"].isna() | (df["UnitPrice"] <= 0)
    nonpositive_price_rows = int(nonpos_price_mask.sum())
    for idx, is_nonpos in enumerate(nonpos_price_mask):
        if is_nonpos:
            exclusion_reasons[idx].append("NON_POSITIVE_PRICE")
            
    # Missing StockCode
    missing_stock_mask = df["StockCode"].isna() | (df["StockCode"] == "") | (df["StockCode"].str.lower() == "nan")
    missing_stockcode_rows = int(missing_stock_mask.sum())
    for idx, is_miss in enumerate(missing_stock_mask):
        if is_miss:
            exclusion_reasons[idx].append("MISSING_STOCK_CODE")
            
    # Missing Date
    missing_date_mask = df["InvoiceDate"].isna()
    missing_date_rows = int(missing_date_mask.sum())
    for idx, is_miss in enumerate(missing_date_mask):
        if is_miss:
            exclusion_reasons[idx].append("MISSING_INVOICE_DATE")
            
    # Missing Description
    missing_desc_mask = df["Description"].isna() | (df["Description"] == "")
    missing_description_rows = int(missing_desc_mask.sum())
    for idx, is_miss in enumerate(missing_desc_mask):
        if is_miss:
            exclusion_reasons[idx].append("MISSING_DESCRIPTION")
            
    # Non-merchandise check
    non_merch_mask = [
        identify_non_merchandise(sc, desc)
        for sc, desc in zip(df["StockCode"], df["Description"])
    ]
    non_merchandise_rows = int(sum(non_merch_mask))
    for idx, is_nonm in enumerate(non_merch_mask):
        if is_nonm:
            exclusion_reasons[idx].append("NON_MERCHANDISE_LINE")
            
    # Compile exclusion table
    has_exclusion = [len(r) > 0 for r in exclusion_reasons]
    df["ExclusionReasons"] = ["; ".join(r) if r else "" for r in exclusion_reasons]
    
    excluded_df = df[has_exclusion].copy()
    cleaned_df = df[~np.array(has_exclusion)].copy()
    
    # Calculate revenue and date for cleaned transactions
    cleaned_df["RevenueGBP"] = cleaned_df["Quantity"] * cleaned_df["UnitPrice"]
    cleaned_df["Date"] = pd.to_datetime(cleaned_df["InvoiceDate"].dt.date)
    
    # Sort strictly
    cleaned_df = cleaned_df.sort_values(by=["StockCode", "InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
    
    cleaned_rows = len(cleaned_df)
    unique_products = int(cleaned_df["StockCode"].nunique())
    min_date = str(cleaned_df["Date"].min().strftime("%Y-%m-%d")) if cleaned_rows > 0 else ""
    max_date = str(cleaned_df["Date"].max().strftime("%Y-%m-%d")) if cleaned_rows > 0 else ""
    
    audit_dict = {
        "raw_rows": raw_rows,
        "duplicate_rows": duplicate_rows,
        "cancelled_rows": cancelled_rows,
        "nonpositive_quantity_rows": nonpositive_quantity_rows,
        "nonpositive_price_rows": nonpositive_price_rows,
        "missing_stockcode_rows": missing_stockcode_rows,
        "missing_date_rows": missing_date_rows,
        "missing_description_rows": missing_description_rows,
        "non_merchandise_rows": non_merchandise_rows,
        "cleaned_rows": cleaned_rows,
        "excluded_rows": len(excluded_df),
        "unique_products": unique_products,
        "min_date": min_date,
        "max_date": max_date,
    }
    
    return cleaned_df, excluded_df, audit_dict


def build_daily_demand(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions to daily product-level series with zero-day insertion.
    
    For each product:
    - Activity window is defined from first valid sale date to last valid sale date.
    - Missing calendar days within that window are inserted with zero demand and zero revenue.
    - Explicit flag 'IsObserved' distinguishes observed transaction days from inserted zero days.
    
    Args:
        cleaned_df: Cleaned transaction dataframe.
        
    Returns:
        DataFrame containing complete calendar daily demand for each product.
    """
    if cleaned_df.empty:
        return pd.DataFrame(
            columns=[
                "StockCode",
                "Date",
                "DailyUnits",
                "DailyRevenueGBP",
                "TransactionCount",
                "AverageSellingPriceGBP",
                "IsObserved",
            ]
        )

    # 1. Group by StockCode and Date
    grouped = (
        cleaned_df.groupby(["StockCode", "Date"], as_index=False)
        .agg(
            DailyUnits=("Quantity", "sum"),
            DailyRevenueGBP=("RevenueGBP", "sum"),
            TransactionCount=("InvoiceNo", "nunique"),
        )
    )
    grouped["AverageSellingPriceGBP"] = grouped["DailyRevenueGBP"] / grouped["DailyUnits"]
    grouped["IsObserved"] = 1

    # 2. Expand each product to all calendar days within its activity window
    daily_records = []
    
    for stock_code, product_df in grouped.groupby("StockCode"):
        start_date = product_df["Date"].min()
        end_date = product_df["Date"].max()
        
        # Complete calendar index
        full_date_idx = pd.date_range(start=start_date, end=end_date, freq="D", name="Date")
        product_full = (
            product_df.set_index("Date")
            .reindex(full_date_idx)
            .reset_index()
        )
        
        product_full["StockCode"] = stock_code
        product_full["IsObserved"] = product_full["IsObserved"].fillna(0).astype(int)
        product_full["DailyUnits"] = product_full["DailyUnits"].fillna(0.0)
        product_full["DailyRevenueGBP"] = product_full["DailyRevenueGBP"].fillna(0.0)
        product_full["TransactionCount"] = product_full["TransactionCount"].fillna(0).astype(int)
        
        # Average selling price for inserted zero days: fill with median observed selling price or 0.0
        median_price = product_df["AverageSellingPriceGBP"].median()
        product_full["AverageSellingPriceGBP"] = (
            product_full["AverageSellingPriceGBP"].fillna(median_price).fillna(0.0)
        )
        
        daily_records.append(product_full)

    all_daily_df = pd.concat(daily_records, ignore_index=True)
    all_daily_df = all_daily_df.sort_values(by=["StockCode", "Date"]).reset_index(drop=True)
    return all_daily_df


def build_product_catalog(
    cleaned_df: pd.DataFrame, daily_df: pd.DataFrame
) -> pd.DataFrame:
    """Build a catalog of products summarizing sales metrics and activity windows.
    
    Args:
        cleaned_df: Cleaned transactions dataframe.
        daily_df: Complete daily demand dataframe.
        
    Returns:
        Catalog DataFrame with metadata and summary metrics for each product.
    """
    # Most common description for each StockCode
    desc_series = (
        cleaned_df.groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else "UNKNOWN")
    )

    catalog = (
        daily_df.groupby("StockCode", as_index=False)
        .agg(
            StartDate=("Date", "min"),
            EndDate=("Date", "max"),
            CalendarDays=("Date", "count"),
            PositiveDemandDays=("DailyUnits", lambda s: int((s > 0).sum())),
            TotalUnitsSold=("DailyUnits", "sum"),
            TotalRevenueGBP=("DailyRevenueGBP", "sum"),
            AverageDailyDemand=("DailyUnits", "mean"),
        )
    )
    catalog["Description"] = catalog["StockCode"].map(desc_series)
    
    # Calculate median selling price from transactions
    median_prices = (
        cleaned_df.groupby("StockCode")["UnitPrice"].median()
    )
    catalog["HistoricalMedianPriceGBP"] = catalog["StockCode"].map(median_prices).fillna(0.0)
    
    # Format dates
    catalog["StartDate"] = catalog["StartDate"].dt.strftime("%Y-%m-%d")
    catalog["EndDate"] = catalog["EndDate"].dt.strftime("%Y-%m-%d")
    
    catalog = catalog.sort_values(by="TotalUnitsSold", ascending=False).reset_index(drop=True)
    return catalog


def select_demo_products(
    catalog_df: pd.DataFrame, daily_df: pd.DataFrame
) -> pd.DataFrame:
    """Select the 30 qualifying demo products based on exact section 3.3 criteria.
    
    Criteria:
    - At least 240 calendar days of activity
    - At least 60 days of positive demand
    - At least 30 positive-demand days in the last 120 observed days
    - Last observed sale no more than 30 days before the dataset's global final date
    - Valid StockCode and Description
    - Sort eligible products by total units sold, descending. Take first 30.
    
    Args:
        catalog_df: Product catalog.
        daily_df: Complete daily demand dataframe.
        
    Returns:
        DataFrame containing selected demo products.
    """
    global_max_date = daily_df["Date"].max()
    qualifying_codes = []

    for _, row in catalog_df.iterrows():
        code = row["StockCode"]
        desc = str(row["Description"])
        if not code or not desc or desc == "UNKNOWN" or desc.lower() == "nan":
            continue

        # Calendar days >= 240
        if row["CalendarDays"] < DEMO_CRITERIA["min_calendar_days"]:
            continue

        # Positive demand days >= 60
        if row["PositiveDemandDays"] < DEMO_CRITERIA["min_positive_demand_days"]:
            continue

        # Last observed sale <= 30 days before dataset global final date
        prod_end = pd.to_datetime(row["EndDate"])
        days_from_end = (global_max_date - prod_end).days
        if days_from_end > DEMO_CRITERIA["max_days_before_dataset_end"]:
            continue

        # Check positive-demand days in the last 120 observed days of this product
        prod_daily = daily_df[daily_df["StockCode"] == code].sort_values("Date")
        recent_120 = prod_daily.tail(DEMO_CRITERIA["recent_window_days"])
        recent_pos_days = int((recent_120["DailyUnits"] > 0).sum())
        if recent_pos_days < DEMO_CRITERIA["min_recent_positive_demand_days"]:
            continue

        qualifying_codes.append(code)

    eligible_df = catalog_df[catalog_df["StockCode"].isin(qualifying_codes)].copy()
    eligible_df = eligible_df.sort_values(by="TotalUnitsSold", ascending=False).reset_index(drop=True)
    demo_df = eligible_df.head(DEMO_CRITERIA["target_demo_count"]).copy()
    return demo_df


def prepare_dataset(
    raw_path: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute complete dataset loading, preprocessing, aggregation, and caching.
    
    Returns:
        Tuple of (cleaned_transactions, excluded_transactions, daily_demand, product_catalog, cleaning_audit).
    """
    target_processed = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
    target_processed.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_df, _ = load_raw_dataset(filepath=raw_path)

    # 2. Clean transactions
    cleaned_df, excluded_df, audit_dict = clean_transactions(raw_df)

    # 3. Daily aggregation
    daily_df = build_daily_demand(cleaned_df)

    # 4. Product catalog
    catalog_df = build_product_catalog(cleaned_df, daily_df)

    # 5. Select demo products
    demo_df = select_demo_products(catalog_df, daily_df)

    # 6. Save required processed files
    cleaned_df.to_parquet(target_processed / "cleaned_transactions.parquet", index=False)
    excluded_df.to_parquet(target_processed / "excluded_transactions.parquet", index=False)
    daily_df.to_parquet(target_processed / "daily_demand.parquet", index=False)
    catalog_df.to_csv(target_processed / "product_catalog.csv", index=False)
    demo_df.to_csv(target_processed / "demo_products.csv", index=False)

    with open(target_processed / "cleaning_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=2)

    return cleaned_df, excluded_df, daily_df, catalog_df, audit_dict
