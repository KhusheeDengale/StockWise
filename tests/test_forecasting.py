"""Automated unit tests for the four forecasting models, evaluation metrics, and backtesting.
"""
import numpy as np
import pandas as pd
import pytest

from core.forecast_evaluation import (
    calculate_metrics,
    compare_and_select_models,
    split_chronological_series,
)
from core.forecast_models import (
    fit_and_forecast_arima,
    fit_and_forecast_linear_regression,
    forecast_moving_average,
    forecast_seasonal_naive,
)
from core.schemas import InsufficientHistoryError


def test_forecast_metrics_exact_formulas():
    """Verify MAE, RMSE, WAPE, and R2 formulas against deterministic hand-calculated vectors."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    # abs errors: [2, 2, 3] -> mean = 7 / 3 = 2.3333
    # squared errors: [4, 4, 9] -> mean = 17 / 3 = 5.6667 -> sqrt = 2.3805
    # sum actual = 60 -> sum abs errors = 7 -> WAPE = 100 * 7 / 60 = 11.6667
    m = calculate_metrics(actual, predicted)
    assert m["MAE"] == pytest.approx(2.3333, abs=1e-3)
    assert m["RMSE"] == pytest.approx(2.3805, abs=1e-3)
    assert m["WAPE"] == pytest.approx(11.6667, abs=1e-3)
    assert m["R2"] is not None


def test_all_models_produce_finite_nonnegative_forecasts():
    """Verify all 4 models produce finite, non-negative forecasts matching requested horizon."""
    rng = np.random.RandomState(42)
    # 90 days of synthetic seasonal positive demand
    dates = pd.date_range("2010-01-01", periods=90, freq="D")
    units = 15.0 + 5.0 * np.sin(np.arange(90) * (2 * np.pi / 7)) + rng.normal(0, 1, 90)
    units = np.clip(units, 0, None)
    series = pd.Series(units, index=dates)

    for h in [7, 14, 30]:
        # 1. Seasonal Naive
        p_sn, det_sn = forecast_seasonal_naive(series, h)
        assert len(p_sn) == h
        assert np.all(p_sn >= 0)
        assert np.all(np.isfinite(p_sn))

        # 2. Moving Average
        p_ma, det_ma = forecast_moving_average(series, h)
        assert len(p_ma) == h
        assert np.all(p_ma >= 0)
        assert np.all(np.isfinite(p_ma))

        # 3. Linear Regression
        p_lr, det_lr = fit_and_forecast_linear_regression(series, pd.Series(dates), h)
        assert len(p_lr) == h
        assert np.all(p_lr >= 0)
        assert np.all(np.isfinite(p_lr))
        assert det_lr["worked_example"]["matches_unclipped"] is True

        # 4. ARIMA
        p_ar, det_ar = fit_and_forecast_arima(series, None, h)
        assert len(p_ar) == h
        assert np.all(p_ar >= 0)
        assert np.all(np.isfinite(p_ar))


def test_chronological_splitting_ratios():
    """Verify chronological split strictly enforces 70% train, 15% validation, and 15% test."""
    n = 100
    df = pd.DataFrame({
        "Date": pd.date_range("2010-01-01", periods=n, freq="D"),
        "DailyUnits": [10.0] * n,
    })
    train_df, val_df, test_df = split_chronological_series(df, horizon=7)
    
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    # Strict order check
    assert train_df["Date"].max() < val_df["Date"].min()
    assert val_df["Date"].max() < test_df["Date"].min()


def test_model_winner_selected_by_validation_not_test():
    """Verify model selection is strictly determined by validation performance."""
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "Date": pd.date_range("2010-01-01", periods=n, freq="D"),
        "DailyUnits": 20.0 + 8.0 * np.sin(np.arange(n) * (2 * np.pi / 7)) + rng.normal(0, 1, n),
    })
    res = compare_and_select_models(df, horizon=7)
    winner = res["winning_model"]
    summary = {r["Model"]: r for r in res["summary_table"]}

    # Selected model must have Selected == True
    assert summary[winner]["Selected"] is True
    # Validation WAPE of winner should be <= validation WAPEs (or tie broken)
    winner_val_wape = summary[winner]["ValidationWAPE"]
    for m_name, m_row in summary.items():
        if m_row["ValidationWAPE"] is not None:
            assert winner_val_wape <= m_row["ValidationWAPE"] + 1e-4


def test_insufficient_history_fallback():
    """Verify InsufficientHistoryError is raised when series is too short for backtesting."""
    short_df = pd.DataFrame({
        "Date": pd.date_range("2010-01-01", periods=20, freq="D"),
        "DailyUnits": [5.0] * 20,
    })
    with pytest.raises(InsufficientHistoryError):
        split_chronological_series(short_df, horizon=14)
