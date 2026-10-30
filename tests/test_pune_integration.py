"""Tests for Pune Stationery Retail Dataset Integration & Services."""
import os
import pytest
import pandas as pd
from pathlib import Path

from config import PUNE_SNAPSHOT_DATE, PUNE_DATA_DIR
from core.pune_data import (
    load_areas,
    load_stores,
    load_products,
    load_suppliers,
    load_calendar,
    load_assortment,
    load_events,
    load_current_inventory,
    load_new_store_reference,
    load_purchase_orders,
    is_store_active,
    list_stores,
    list_products_for_store,
    get_store_inventory,
    get_product_daily_series,
)
from core.forecast_service import ForecastService
from core.fuzzy_engine import evaluate_fuzzy_urgency


def test_pune_master_files_exist_and_counts():
    """Verify that all 12 Pune synthetic CSVs exist and meet expected row counts."""
    assert os.path.isdir(PUNE_DATA_DIR), f"Directory not found: {PUNE_DATA_DIR}"

    areas = load_areas()
    assert len(areas) == 12, f"Expected 12 areas, got {len(areas)}"

    stores = load_stores()
    assert len(stores) == 18, f"Expected 18 stores, got {len(stores)}"

    prods = load_products()
    assert len(prods) == 100, f"Expected 100 products, got {len(prods)}"

    supps = load_suppliers()
    assert len(supps) == 20, f"Expected 20 suppliers, got {len(supps)}"

    cal = load_calendar()
    assert len(cal) == 2647, f"Expected 2647 calendar rows, got {len(cal)}"

    assort = load_assortment()
    assert len(assort) == 1465, f"Expected 1465 assortment rows, got {len(assort)}"

    events = load_events()
    assert len(events) == 891, f"Expected 891 event rows, got {len(events)}"

    inv = load_current_inventory()
    assert len(inv) == 1182, f"Expected 1182 snapshot rows, got {len(inv)}"

    ref = load_new_store_reference()
    assert len(ref) == 840, f"Expected 840 new store reference scenarios, got {len(ref)}"


def test_closed_stores_handling():
    """Verify S02 and S12 are identified as closed and excluded from active lists."""
    assert not is_store_active("S02", as_of=PUNE_SNAPSHOT_DATE)
    assert not is_store_active("S12", as_of=PUNE_SNAPSHOT_DATE)
    assert is_store_active("S01", as_of=PUNE_SNAPSHOT_DATE)

    active_stores = list_stores(active_at=PUNE_SNAPSHOT_DATE)
    assert len(active_stores) == 16
    assert "S02" not in active_stores["StoreID"].values
    assert "S12" not in active_stores["StoreID"].values


def test_store_inventory_snapshot():
    """Verify inventory is retrieved for an active store and contains expected columns."""
    inv = get_store_inventory("S01")
    assert not inv.empty
    expected_cols = {"ProductID", "ProductName", "Category", "CurrentStockUnits", "LeadTimeDays", "SupplierUnitCostINR"}
    assert expected_cols.issubset(set(inv.columns))


def test_pune_forecast_and_fuzzy_integration():
    """Verify that a Pune product daily sales series flows into ForecastService and Fuzzy risk evaluation."""
    series_df = get_product_daily_series("S01", "P001")
    assert not series_df.empty
    assert "Date" in series_df.columns
    assert "DailyUnits" in series_df.columns

    # Filter up to snapshot date and use recent history for fast unit test
    series_df = series_df[series_df["Date"] <= pd.Timestamp(PUNE_SNAPSHOT_DATE)].tail(120).copy()
    assert len(series_df) == 120

    # Run forecast
    fc_service = ForecastService()
    res = fc_service.forecast_product(
        daily_df=series_df,
        stock_code="P001",
        horizon=14,
        product_description="Test Pune Product",
        use_cache=False,
    )
    assert res.selected_model in ["Seasonal Naive", "Moving Average (7-day)", "Linear Regression", "ARIMA"]
    assert len(res.daily_predictions) == 14
    assert res.total_predicted_demand >= 0

    # Run fuzzy risk
    fuzzy_res = evaluate_fuzzy_urgency(
        current_stock=20,
        lead_time_days=7,
        planning_horizon_days=14,
        lead_time_forecast=res.daily_predictions[:7],
        uncertainty_score=res.uncertainty_score,
    )
    assert 0.0 <= fuzzy_res.centroid <= 100.0
    assert fuzzy_res.category in ["Low", "Moderate", "High", "Critical"]
