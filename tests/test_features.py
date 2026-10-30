"""Automated deterministic tests for feature engineering, lags, rolling statistics, and leakage prevention.
"""
import numpy as np
import pandas as pd
import pytest

from core.features import build_supervised_features


def test_exact_lag_values_on_deterministic_series():
    """Verify exact lag values on known arithmetic series y = [1, 2, ..., 40].
    
    Contract Section 20 Requirement:
    - y = [1, 2, 3, ..., 40]
    - Verify:
        Lag1 at day 29 = 28
        Lag7 at day 29 = 22
        Lag28 at day 29 = 1
    (Using consistent 1-based day labels).
    """
    # 1-based days 1 to 40 with y[t] = t
    days = list(range(1, 41))
    y = pd.Series(days, index=pd.date_range("2010-01-01", periods=40, freq="D"))

    features, target, _ = build_supervised_features(y)

    # In our output, the first 28 observations (days 1 to 28) are warm-up.
    # Therefore, the first surviving row in features corresponds to day 29.
    day_29_row = features.iloc[0]
    day_29_target = target.iloc[0]

    assert day_29_target == 29
    assert day_29_row["Lag1"] == 28
    assert day_29_row["Lag7"] == 22
    assert day_29_row["Lag14"] == 15
    assert day_29_row["Lag28"] == 1


def test_rolling_mean_strictly_preceding_observations():
    """Verify that RollingMean7 uses strictly preceding observations (days t-7 to t-1)."""
    days = list(range(1, 41))
    y = pd.Series(days, index=pd.date_range("2010-01-01", periods=40, freq="D"))

    features, _, _ = build_supervised_features(y)
    day_29_row = features.iloc[0]

    # Preceding 7 observations before day 29 are days 22, 23, 24, 25, 26, 27, 28
    expected_mean = np.mean([22, 23, 24, 25, 26, 27, 28])
    assert day_29_row["RollingMean7"] == expected_mean
    assert day_29_row["RollingMean7"] == 25.0


def test_no_target_leakage():
    """Verify that shifting target values in future rows does not impact features of previous rows."""
    y = pd.Series([10.0] * 35, index=pd.date_range("2010-01-01", periods=35, freq="D"))
    features1, _, _ = build_supervised_features(y)

    # Modify the last day's value
    y_perturbed = y.copy()
    y_perturbed.iloc[-1] = 999999.0
    features2, _, _ = build_supervised_features(y_perturbed)

    # Features for earlier rows (e.g. index 0 to -2) must be completely identical
    pd.testing.assert_frame_equal(features1.iloc[:-1], features2.iloc[:-1])
