"""Automated deterministic tests for data loader, schema normalization, cleaning, and daily aggregation.
"""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from core.data_loader import normalize_column_names
from core.preprocessing import build_daily_demand, clean_transactions
from core.schemas import SchemaValidationError


def test_schema_alias_mapping():
    """Verify that raw columns are correctly mapped to canonical names."""
    raw_df = pd.DataFrame({
        "Invoice": ["489434"],
        "StockCode": ["85048"],
        "Description": ["TEST ITEM"],
        "Quantity": [10],
        "InvoiceDate": [datetime(2009, 12, 1, 7, 45)],
        "Price": [2.50],
        "Customer ID": [13085],
        "Country": ["United Kingdom"],
    })
    normalized = normalize_column_names(raw_df)
    assert "InvoiceNo" in normalized.columns
    assert "UnitPrice" in normalized.columns
    assert "CustomerID" in normalized.columns
    assert "Invoice" not in normalized.columns
    assert "Price" not in normalized.columns


def test_schema_validation_error():
    """Verify that missing required columns raise a descriptive SchemaValidationError."""
    incomplete_df = pd.DataFrame({"Invoice": ["123"], "StockCode": ["ABC"]})
    with pytest.raises(SchemaValidationError) as exc_info:
        normalize_column_names(incomplete_df)
    assert "Missing required canonical columns" in str(exc_info.value)


def test_cancellation_and_negative_quantities_excluded():
    """Verify that cancellations (C-prefix) and non-positive quantities are marked and excluded."""
    data = pd.DataFrame({
        "InvoiceNo": ["489434", "C489435", "489436", "489437"],
        "StockCode": ["85048", "85048", "85048", "85048"],
        "Description": ["ITEM A", "ITEM A", "ITEM A", "ITEM A"],
        "Quantity": [10, -5, 0, 15],
        "InvoiceDate": [
            datetime(2009, 12, 1, 10, 0),
            datetime(2009, 12, 1, 11, 0),
            datetime(2009, 12, 1, 12, 0),
            datetime(2009, 12, 1, 13, 0),
        ],
        "UnitPrice": [2.50, 2.50, 2.50, 2.50],
        "CustomerID": [1001, 1001, 1001, 1002],
        "Country": ["United Kingdom"] * 4,
        "SourceSheet": ["Year 2009-2010"] * 4,
    })
    cleaned, excluded, audit = clean_transactions(data)
    assert len(cleaned) == 2
    assert audit["cancelled_rows"] == 1
    assert audit["nonpositive_quantity_rows"] == 2
    assert set(cleaned["InvoiceNo"].tolist()) == {"489434", "489437"}


def test_duplicate_handling_and_revenue():
    """Verify that duplicate rows are counted and removed, and revenue is computed correctly."""
    data = pd.DataFrame({
        "InvoiceNo": ["489434", "489434", "489438"],
        "StockCode": ["85048", "85048", "85048"],
        "Description": ["ITEM A", "ITEM A", "ITEM A"],
        "Quantity": [10, 10, 5],
        "InvoiceDate": [
            datetime(2009, 12, 1, 10, 0),
            datetime(2009, 12, 1, 10, 0),
            datetime(2009, 12, 2, 10, 0),
        ],
        "UnitPrice": [3.00, 3.00, 4.00],
        "CustomerID": [1001, 1001, 1002],
        "Country": ["United Kingdom"] * 3,
        "SourceSheet": ["Year 2009-2010"] * 3,
    })
    cleaned, excluded, audit = clean_transactions(data)
    assert audit["duplicate_rows"] == 1
    assert len(cleaned) == 2
    # Revenue = Quantity * UnitPrice
    row1 = cleaned[cleaned["InvoiceNo"] == "489434"].iloc[0]
    assert row1["RevenueGBP"] == 30.00
    row2 = cleaned[cleaned["InvoiceNo"] == "489438"].iloc[0]
    assert row2["RevenueGBP"] == 20.00


def test_missing_customer_id_does_not_invalidate_sales():
    """Verify that missing CustomerID is preserved in valid sales."""
    data = pd.DataFrame({
        "InvoiceNo": ["489439"],
        "StockCode": ["85048"],
        "Description": ["VALID ITEM"],
        "Quantity": [10],
        "InvoiceDate": [datetime(2009, 12, 1, 10, 0)],
        "UnitPrice": [5.00],
        "CustomerID": [np.nan],
        "Country": ["United Kingdom"],
        "SourceSheet": ["Year 2009-2010"],
    })
    cleaned, _, _ = clean_transactions(data)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["RevenueGBP"] == 50.00


def test_daily_aggregation_and_zero_day_insertion():
    """Verify that daily aggregation creates complete calendar days with zero demand inserted."""
    cleaned = pd.DataFrame({
        "InvoiceNo": ["INV1", "INV2", "INV3"],
        "StockCode": ["SKU1", "SKU1", "SKU1"],
        "Quantity": [10, 5, 20],
        "RevenueGBP": [20.0, 10.0, 40.0],
        "Date": [
            pd.to_datetime("2010-01-01"),
            pd.to_datetime("2010-01-01"),  # Same day: should aggregate
            pd.to_datetime("2010-01-04"),  # Gap on 2010-01-02 and 2010-01-03
        ],
    })
    daily = build_daily_demand(cleaned)
    assert len(daily) == 4  # 01, 02, 03, 04

    # Day 1: 10 + 5 = 15 units, 2 invoices
    d1 = daily[daily["Date"] == "2010-01-01"].iloc[0]
    assert d1["DailyUnits"] == 15.0
    assert d1["TransactionCount"] == 2
    assert d1["IsObserved"] == 1

    # Inserted zero days
    d2 = daily[daily["Date"] == "2010-01-02"].iloc[0]
    assert d2["DailyUnits"] == 0.0
    assert d2["DailyRevenueGBP"] == 0.0
    assert d2["IsObserved"] == 0

    d3 = daily[daily["Date"] == "2010-01-03"].iloc[0]
    assert d3["DailyUnits"] == 0.0
    assert d3["IsObserved"] == 0

    # Day 4
    d4 = daily[daily["Date"] == "2010-01-04"].iloc[0]
    assert d4["DailyUnits"] == 20.0
    assert d4["IsObserved"] == 1
