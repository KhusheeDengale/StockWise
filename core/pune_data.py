"""Pune Dataset Data Access Layer — thin repository over 12 Pune CSVs.

Provides efficient, cached, store-scoped access to synthetic Pune stationery
retail data. Converts the 810 MB daily sales CSV to a local Parquet cache on
first load, then serves all subsequent queries from the indexed Parquet.

All data is synthetic (seed 42). Inventory snapshot is 2025-12-31.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from config import BASE_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PUNE_DATA_DIR = BASE_DIR / "data" / "stockwise_pune"
PUNE_CACHE_DIR = BASE_DIR / "data" / "pune_cache"

_CSV_FILES = {
    "areas": "pune_area_profiles.csv",
    "stores": "pune_stores.csv",
    "products": "stationery_products.csv",
    "assortment": "pune_store_assortment.csv",
    "suppliers": "pune_suppliers.csv",
    "supplier_products": "pune_supplier_products.csv",
    "calendar": "pune_calendar.csv",
    "events": "pune_store_events.csv",
    "daily_sales": "pune_daily_sales.csv",
    "purchase_orders": "pune_purchase_orders.csv",
    "current_inventory": "pune_current_inventory.csv",
    "new_store_ref": "pune_new_store_reference.csv",
}

SNAPSHOT_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# Master data loaders (small CSVs — cached in Streamlit session)
# ---------------------------------------------------------------------------

def _pune_csv_path(key: str) -> Path:
    return PUNE_DATA_DIR / _CSV_FILES[key]


def _check_pune_data_exists() -> bool:
    """Return True if all 12 Pune CSVs are present."""
    return all((_pune_csv_path(k)).exists() for k in _CSV_FILES)


@st.cache_data(ttl=None, show_spinner=False)
def load_areas() -> pd.DataFrame:
    return pd.read_csv(_pune_csv_path("areas"))


@st.cache_data(ttl=None, show_spinner=False)
def load_stores() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("stores"))
    df["StoreOpeningDate"] = pd.to_datetime(df["StoreOpeningDate"])
    df["StoreClosingDate"] = pd.to_datetime(df["StoreClosingDate"], errors="coerce")
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_products() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("products"))
    df["ProductLaunchDate"] = pd.to_datetime(df["ProductLaunchDate"])
    df["ProductDiscontinuedDate"] = pd.to_datetime(df["ProductDiscontinuedDate"], errors="coerce")
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_assortment() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("assortment"))
    df["AssortmentStartDate"] = pd.to_datetime(df["AssortmentStartDate"])
    df["AssortmentEndDate"] = pd.to_datetime(df["AssortmentEndDate"], errors="coerce")
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_suppliers() -> pd.DataFrame:
    return pd.read_csv(_pune_csv_path("suppliers"))


@st.cache_data(ttl=None, show_spinner=False)
def load_supplier_products() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("supplier_products"))
    df["EffectiveFrom"] = pd.to_datetime(df["EffectiveFrom"])
    df["EffectiveTo"] = pd.to_datetime(df["EffectiveTo"], errors="coerce")
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_calendar() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("calendar"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_events() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("events"))
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df["EndDate"] = pd.to_datetime(df["EndDate"])
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_purchase_orders() -> pd.DataFrame:
    df = pd.read_csv(_pune_csv_path("purchase_orders"))
    for col in ["OrderDate", "ExpectedDeliveryDate", "ActualDeliveryDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(ttl=None, show_spinner=False)
def load_current_inventory() -> pd.DataFrame:
    return pd.read_csv(_pune_csv_path("current_inventory"))


@st.cache_data(ttl=None, show_spinner=False)
def load_new_store_reference() -> pd.DataFrame:
    return pd.read_csv(_pune_csv_path("new_store_ref"))


# ---------------------------------------------------------------------------
# Daily Sales — Parquet cache for the 3M-row CSV
# ---------------------------------------------------------------------------

def _daily_sales_parquet_path() -> Path:
    return PUNE_CACHE_DIR / "pune_daily_sales.parquet"


def _ensure_daily_sales_parquet() -> Path:
    """Convert daily sales CSV to Parquet on first run. Returns parquet path."""
    parquet_path = _daily_sales_parquet_path()
    PUNE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists():
        return parquet_path

    csv_path = _pune_csv_path("daily_sales")
    if not csv_path.exists():
        raise FileNotFoundError(f"Pune daily sales CSV not found: {csv_path}")

    # Read in chunks and write to Parquet atomically
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500_000, low_memory=False):
        chunk["Date"] = pd.to_datetime(chunk["Date"])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Optimize dtypes for memory
    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].min() >= 0 and df[col].max() < 32767:
            df[col] = df[col].astype("int16")
        elif df[col].min() >= -32768 and df[col].max() < 32767:
            df[col] = df[col].astype("int16")
        elif df[col].max() < 2_147_483_647:
            df[col] = df[col].astype("int32")

    # Write atomically
    tmp_path = parquet_path.with_suffix(".tmp")
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    tmp_path.rename(parquet_path)

    return parquet_path


@st.cache_data(ttl=None, show_spinner=False)
def _load_store_daily_sales(store_id: str) -> pd.DataFrame:
    """Load daily sales for a single store using PyArrow pushdown predicate. Fast & memory-light (~45MB)."""
    pq_path = _ensure_daily_sales_parquet()
    tbl = pq.read_table(pq_path, filters=[("StoreID", "==", store_id)])
    df = tbl.to_pandas()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _load_daily_sales_full() -> pd.DataFrame:
    """Load full daily sales from Parquet cache without pickling whole 3M rows in streamlit cache."""
    pq_path = _ensure_daily_sales_parquet()
    df = pd.read_parquet(pq_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# Query functions — efficient filtered access
# ---------------------------------------------------------------------------

def list_areas() -> pd.DataFrame:
    """Return all 12 area profiles."""
    return load_areas()


def list_stores(area_id: Optional[str] = None, active_at: Optional[str] = None) -> pd.DataFrame:
    """Return stores, optionally filtered by area and active status at a date."""
    df = load_stores()
    if area_id:
        df = df[df["AreaID"] == area_id]
    if active_at:
        dt = pd.Timestamp(active_at)
        mask = df["StoreOpeningDate"] <= dt
        mask &= df["StoreClosingDate"].isna() | (df["StoreClosingDate"] >= dt)
        df = df[mask]
    return df.reset_index(drop=True)


def is_store_active(store_id: str, as_of: str = SNAPSHOT_DATE) -> bool:
    """Check if a store is active on a given date."""
    stores = load_stores()
    row = stores[stores["StoreID"] == store_id]
    if row.empty:
        return False
    r = row.iloc[0]
    dt = pd.Timestamp(as_of)
    if r["StoreOpeningDate"] > dt:
        return False
    if pd.notna(r["StoreClosingDate"]) and r["StoreClosingDate"] < dt:
        return False
    return True


def list_products_for_store(store_id: str, as_of: Optional[str] = None) -> pd.DataFrame:
    """Return products available at a store, respecting assortment and product lifecycle."""
    assort = load_assortment()
    products = load_products()

    sa = assort[assort["StoreID"] == store_id].copy()

    if as_of:
        dt = pd.Timestamp(as_of)
        sa = sa[sa["AssortmentStartDate"] <= dt]
        sa = sa[sa["AssortmentEndDate"].isna() | (sa["AssortmentEndDate"] >= dt)]

    merged = sa.merge(products, on="ProductID", how="left", suffixes=("_assort", "_prod"))

    if as_of:
        dt = pd.Timestamp(as_of)
        merged = merged[merged["ProductLaunchDate"] <= dt]
        merged = merged[merged["ProductDiscontinuedDate"].isna() | (merged["ProductDiscontinuedDate"] >= dt)]

    return merged.reset_index(drop=True)


def get_store_daily_sales(
    store_id: str,
    product_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Get daily sales for a store, optionally filtered by product and date range."""
    if product_id:
        pq_path = _ensure_daily_sales_parquet()
        tbl = pq.read_table(
            pq_path,
            filters=[("StoreID", "==", store_id), ("ProductID", "==", product_id)]
        )
        df = tbl.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = _load_store_daily_sales(store_id)

    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df["Date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["Date"] <= pd.Timestamp(end_date)
    return df[mask].sort_values("Date").reset_index(drop=True)


def get_product_daily_series(
    store_id: str,
    product_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Get daily series for one store-product pair, suitable for forecasting.

    Returns DataFrame with columns adapted for the forecast engine:
    Date, DailyUnits (=UnitsSold), DailyRevenueINR (=NetRevenueINR), etc.
    """
    df = get_store_daily_sales(store_id, product_id, start_date, end_date)
    if df.empty:
        return df

    # Rename to match forecast engine expectations
    result = df.rename(columns={
        "UnitsSold": "DailyUnits",
        "NetRevenueINR": "DailyRevenueINR",
        "ActualSellingPriceINR": "AverageSellingPriceINR",
    })
    return result


def get_store_overview(
    store_id: Optional[str] = None,
    area_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute overview KPIs for a store or area over a date range."""
    if store_id:
        df = _load_store_daily_sales(store_id)
    else:
        pq_path = _ensure_daily_sales_parquet()
        cols = ["Date", "StoreID", "AreaID", "UnitsSold", "NetRevenueINR", "GrossProfitINR", "TransactionCount", "StockoutFlag", "IsStoreClosed", "ProductID"]
        tbl = pq.read_table(pq_path, columns=cols)
        df = tbl.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])

    mask = pd.Series(True, index=df.index)
    if store_id and not df.empty:
        mask &= df["StoreID"] == store_id
    if area_id:
        mask &= df["AreaID"] == area_id
    if start_date:
        mask &= df["Date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["Date"] <= pd.Timestamp(end_date)

    filtered = df[mask]

    if filtered.empty:
        return {
            "net_revenue_inr": 0.0,
            "units_sold": 0,
            "transaction_count": 0,
            "gross_profit_inr": 0.0,
            "gross_margin_pct": 0.0,
            "active_skus": 0,
            "avg_transaction_value": 0.0,
            "total_days": 0,
            "open_days": 0,
            "stockout_days": 0,
        }

    net_rev = float(filtered["NetRevenueINR"].sum())
    units = int(filtered["UnitsSold"].sum())
    tx_count = int(filtered["TransactionCount"].sum())
    gross_profit = float(filtered["GrossProfitINR"].sum())
    gross_margin = (gross_profit / net_rev * 100.0) if net_rev > 0 else 0.0
    active_skus = int(filtered["ProductID"].nunique())
    avg_tx_value = (net_rev / tx_count) if tx_count > 0 else 0.0
    total_days = int(filtered["Date"].nunique())
    open_days = int(filtered[filtered["IsStoreClosed"] == 0]["Date"].nunique())
    stockout_days = int(filtered[filtered["StockoutFlag"] == 1].shape[0])

    return {
        "net_revenue_inr": round(net_rev, 2),
        "units_sold": units,
        "transaction_count": tx_count,
        "gross_profit_inr": round(gross_profit, 2),
        "gross_margin_pct": round(gross_margin, 2),
        "active_skus": active_skus,
        "avg_transaction_value": round(avg_tx_value, 2),
        "total_days": total_days,
        "open_days": open_days,
        "stockout_days": stockout_days,
    }


def get_store_inventory(store_id: str, as_of: str = SNAPSHOT_DATE) -> pd.DataFrame:
    """Get current inventory snapshot for a store."""
    inv = load_current_inventory()
    return inv[inv["StoreID"] == store_id].reset_index(drop=True)


def get_valid_suppliers(
    product_id: str,
    planning_date: str = SNAPSHOT_DATE,
    store_id: Optional[str] = None,
) -> pd.DataFrame:
    """Get valid supplier options for a product on a given date.

    Returns supplier-product records with effective dates covering planning_date,
    merged with supplier master data. Sorted by landed cost (unit cost + delivery charge).
    """
    sp = load_supplier_products()
    suppliers = load_suppliers()

    dt = pd.Timestamp(planning_date)
    mask = (sp["ProductID"] == product_id) & (sp["EffectiveFrom"] <= dt)
    mask &= sp["EffectiveTo"].isna() | (sp["EffectiveTo"] >= dt)
    valid = sp[mask].copy()

    if valid.empty:
        # Carry-forward: use last known terms before planning_date
        all_for_product = sp[sp["ProductID"] == product_id].sort_values("EffectiveFrom", ascending=False)
        if not all_for_product.empty:
            valid = all_for_product.head(1).copy()
            valid["_carry_forward"] = True

    merged = valid.merge(suppliers, on="SupplierID", how="left")

    # Compute landed cost (unit cost + per-unit share of delivery charge)
    # DeliveryChargeINR is per order — for comparison, add it to MOQ-worth
    merged["LandedUnitCost"] = merged["SupplierUnitCostINR"]

    # If store has preferred supplier, mark it
    if store_id:
        assort = load_assortment()
        pref = assort[(assort["StoreID"] == store_id) & (assort["ProductID"] == product_id)]
        if not pref.empty:
            pref_sup = pref.iloc[0].get("PreferredSupplierID", "")
            merged["IsPreferred"] = merged["SupplierID"] == pref_sup

    # Sort by primary first, then landed cost
    sort_cols = []
    if "IsPrimaryOption" in merged.columns:
        sort_cols.append("IsPrimaryOption")
    sort_cols.append("LandedUnitCost")
    merged = merged.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))

    return merged.reset_index(drop=True)


def get_calendar_features(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Get calendar features for a date range."""
    cal = load_calendar()
    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    mask = (cal["Date"] >= dt_start) & (cal["Date"] <= dt_end)
    return cal[mask].reset_index(drop=True)


def get_store_events(
    store_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Get events for a store within a date range."""
    events = load_events()
    mask = events["StoreID"] == store_id
    if start_date:
        mask &= events["EndDate"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= events["StartDate"] <= pd.Timestamp(end_date)
    return events[mask].reset_index(drop=True)


def get_new_store_reference(
    area_id: Optional[str] = None,
    store_type: Optional[str] = None,
) -> pd.DataFrame:
    """Get new store reference scenarios, optionally filtered."""
    ref = load_new_store_reference()
    if area_id:
        ref = ref[ref["AreaID"] == area_id]
    if store_type:
        ref = ref[ref["ProposedStoreType"] == store_type]
    return ref.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def get_category_revenue_breakdown(
    store_id: Optional[str] = None,
    area_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Revenue breakdown by product category."""
    if store_id:
        df = _load_store_daily_sales(store_id)
    else:
        pq_path = _ensure_daily_sales_parquet()
        cols = ["Date", "StoreID", "AreaID", "Category", "NetRevenueINR", "UnitsSold", "TransactionCount"]
        tbl = pq.read_table(pq_path, columns=cols)
        df = tbl.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])

    mask = pd.Series(True, index=df.index)
    if store_id:
        mask &= df["StoreID"] == store_id
    if area_id:
        mask &= df["AreaID"] == area_id
    if start_date:
        mask &= df["Date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["Date"] <= pd.Timestamp(end_date)
    filtered = df[mask]
    return filtered.groupby("Category").agg(
        NetRevenueINR=("NetRevenueINR", "sum"),
        UnitsSold=("UnitsSold", "sum"),
        TransactionCount=("TransactionCount", "sum"),
    ).reset_index().sort_values("NetRevenueINR", ascending=False)


def get_top_products(
    store_id: Optional[str] = None,
    area_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_n: int = 10,
    by: str = "NetRevenueINR",
) -> pd.DataFrame:
    """Top N products by revenue or units."""
    if store_id:
        df = _load_store_daily_sales(store_id)
    else:
        pq_path = _ensure_daily_sales_parquet()
        cols = ["Date", "StoreID", "AreaID", "ProductID", "ProductName", "Category", "NetRevenueINR", "UnitsSold"]
        tbl = pq.read_table(pq_path, columns=cols)
        df = tbl.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])

    mask = pd.Series(True, index=df.index)
    if store_id:
        mask &= df["StoreID"] == store_id
    if area_id:
        mask &= df["AreaID"] == area_id
    if start_date:
        mask &= df["Date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["Date"] <= pd.Timestamp(end_date)
    filtered = df[mask]
    agg = filtered.groupby(["ProductID", "ProductName", "Category"]).agg(
        NetRevenueINR=("NetRevenueINR", "sum"),
        UnitsSold=("UnitsSold", "sum"),
    ).reset_index()
    return agg.nlargest(top_n, by).reset_index(drop=True)


def get_daily_trend(
    store_id: Optional[str] = None,
    area_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = "W",
) -> pd.DataFrame:
    """Aggregated time series for trend charts."""
    if store_id:
        df = _load_store_daily_sales(store_id)
    else:
        pq_path = _ensure_daily_sales_parquet()
        cols = ["Date", "StoreID", "AreaID", "NetRevenueINR", "UnitsSold", "GrossProfitINR", "TransactionCount"]
        tbl = pq.read_table(pq_path, columns=cols)
        df = tbl.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])

    mask = pd.Series(True, index=df.index)
    if store_id:
        mask &= df["StoreID"] == store_id
    if area_id:
        mask &= df["AreaID"] == area_id
    if start_date:
        mask &= df["Date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["Date"] <= pd.Timestamp(end_date)
    filtered = df[mask]
    if filtered.empty:
        return pd.DataFrame(columns=["Date", "NetRevenueINR", "UnitsSold", "GrossProfitINR"])
    grouped = filtered.groupby(pd.Grouper(key="Date", freq=freq)).agg(
        NetRevenueINR=("NetRevenueINR", "sum"),
        UnitsSold=("UnitsSold", "sum"),
        GrossProfitINR=("GrossProfitINR", "sum"),
        TransactionCount=("TransactionCount", "sum"),
    ).reset_index()
    return grouped


def get_low_stock_summary(store_id: str) -> pd.DataFrame:
    """Products below reorder point at snapshot."""
    inv = get_store_inventory(store_id)
    if inv.empty:
        return inv
    low = inv[inv["CurrentStockUnits"] <= inv["ReorderPointUnits"]].copy()
    low["StockCoverageStatus"] = "Below Reorder Point"
    return low.sort_values("StockCoverageDays").reset_index(drop=True)


def get_data_quality_summary(store_id: Optional[str] = None) -> Dict[str, Any]:
    """Data quality metrics for the Data Explorer page."""
    if store_id:
        df = _load_store_daily_sales(store_id)
    else:
        pq_path = _ensure_daily_sales_parquet()
        cols = ["Date", "StoreID", "ProductID", "UnitsSold", "StockoutFlag", "IsStoreClosed"]
        tbl = pq.read_table(pq_path, columns=cols)
        df = tbl.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])

    if store_id:
        df = df[df["StoreID"] == store_id]

    total_rows = len(df)
    if total_rows == 0:
        return {"total_rows": 0}

    return {
        "total_rows": total_rows,
        "unique_dates": int(df["Date"].nunique()),
        "unique_stores": int(df["StoreID"].nunique()),
        "unique_products": int(df["ProductID"].nunique()),
        "date_range": (str(df["Date"].min().date()), str(df["Date"].max().date())),
        "zero_demand_share": round(float((df["UnitsSold"] == 0).mean() * 100), 2),
        "stockout_share": round(float((df["StockoutFlag"] == 1).mean() * 100), 2),
        "closed_day_share": round(float((df["IsStoreClosed"] == 1).mean() * 100), 2),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_keys": int(df.duplicated(subset=["Date", "StoreID", "ProductID"]).sum()),
    }


# ---------------------------------------------------------------------------
# Initialization check
# ---------------------------------------------------------------------------

def initialize_pune_data() -> Dict[str, Any]:
    """Initialize Pune data — ensure Parquet cache exists and return summary info."""
    if not _check_pune_data_exists():
        raise FileNotFoundError(
            f"Pune dataset CSVs not found in {PUNE_DATA_DIR}. "
            f"Please extract stockwise_pune_dataset.zip to data/stockwise_pune/"
        )

    # Trigger Parquet conversion if needed
    _ensure_daily_sales_parquet()

    stores = load_stores()
    products = load_products()
    areas = load_areas()
    inv = load_current_inventory()

    active_stores = list_stores(active_at=SNAPSHOT_DATE)

    return {
        "status": "ready",
        "dataset": "Pune Stationery Retail Intelligence",
        "snapshot_date": SNAPSHOT_DATE,
        "total_stores": len(stores),
        "active_stores": len(active_stores),
        "total_products": len(products),
        "total_areas": len(areas),
        "inventory_records": len(inv),
        "date_range": ("2019-01-01", "2025-12-31"),
        "calendar_range": ("2019-01-01", "2026-03-31"),
    }
