"""Feature engineering, supervised transformation, and feature selection for demand forecasting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURE_NAMES = [
    "Lag1",
    "Lag7",
    "Lag14",
    "Lag28",
    "RollingMean7",
    "RollingMean14",
    "RollingStd7",
]

CALENDAR_FEATURE_NAMES = [
    "DayOfWeek",
    "Month",
    "IsWeekend",
]

FEATURE_DEFINITIONS = [
    {
        "name": "Lag1",
        "type": "numeric_lag",
        "formula": "y[t-1]",
        "description": "Demand from 1 day prior (immediate autoregressive term)",
    },
    {
        "name": "Lag7",
        "type": "numeric_lag",
        "formula": "y[t-7]",
        "description": "Demand from 7 days prior (same day of week last week)",
    },
    {
        "name": "Lag14",
        "type": "numeric_lag",
        "formula": "y[t-14]",
        "description": "Demand from 14 days prior (fortnightly lag)",
    },
    {
        "name": "Lag28",
        "type": "numeric_lag",
        "formula": "y[t-28]",
        "description": "Demand from 28 days prior (4-week cycle lag)",
    },
    {
        "name": "RollingMean7",
        "type": "numeric_rolling",
        "formula": "mean(y[t-7], ..., y[t-1])",
        "description": "7-day trailing moving average using strictly preceding days",
    },
    {
        "name": "RollingMean14",
        "type": "numeric_rolling",
        "formula": "mean(y[t-14], ..., y[t-1])",
        "description": "14-day trailing moving average using strictly preceding days",
    },
    {
        "name": "RollingStd7",
        "type": "numeric_rolling",
        "formula": "sample_std(y[t-7], ..., y[t-1], ddof=1)",
        "description": "7-day trailing sample standard deviation of demand volatility",
    },
    {
        "name": "DayOfWeek",
        "type": "categorical_calendar",
        "formula": "date.dayofweek (0=Monday, 6=Sunday)",
        "description": "Weekly cyclical seasonality encoded via one-hot encoder",
    },
    {
        "name": "Month",
        "type": "categorical_calendar",
        "formula": "date.month (1=January, 12=December)",
        "description": "Annual seasonality encoded via one-hot encoder",
    },
    {
        "name": "IsWeekend",
        "type": "binary_calendar",
        "formula": "1 if DayOfWeek in (5, 6) else 0",
        "description": "Binary indicator for Saturday and Sunday retail traffic patterns",
    },
]


def build_supervised_features(
    daily_series: pd.Series, dates: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series, List[Dict[str, str]]]:
    """Construct supervised tabular features from a univariate daily demand series.
    
    Drops the first 28 warm-up observations to ensure complete lag history.
    
    Args:
        daily_series: Daily demand units indexed chronologically.
        dates: Optional corresponding datetime series or index.
        
    Returns:
        Tuple of (feature_dataframe, target_series, feature_definitions).
    """
    y = pd.to_numeric(daily_series, errors="coerce").copy()
    if dates is not None:
        dt_series = pd.to_datetime(dates)
    elif isinstance(y.index, pd.DatetimeIndex):
        dt_series = pd.Series(y.index, index=y.index)
    else:
        # Default calendar progression starting from dummy date if unsupplied
        dt_series = pd.date_range("2010-01-01", periods=len(y), freq="D")
        dt_series = pd.Series(dt_series, index=y.index)

    df = pd.DataFrame(index=y.index)
    
    # 1. Target
    df["TargetDemand"] = y

    # 2. Lags
    df["Lag1"] = y.shift(1)
    df["Lag7"] = y.shift(7)
    df["Lag14"] = y.shift(14)
    df["Lag28"] = y.shift(28)

    # 3. Rolling metrics (strictly preceding observations using shift(1))
    shifted = y.shift(1)
    df["RollingMean7"] = shifted.rolling(window=7, min_periods=7).mean()
    df["RollingMean14"] = shifted.rolling(window=14, min_periods=14).mean()
    df["RollingStd7"] = shifted.rolling(window=7, min_periods=7).std(ddof=1).fillna(0.0)

    # 4. Calendar features
    df["DayOfWeek"] = dt_series.dt.dayofweek
    df["Month"] = dt_series.dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Drop the first 28 observations that lack full lag history
    clean_df = df.iloc[28:].copy()
    
    target = clean_df["TargetDemand"]
    features = clean_df.drop(columns=["TargetDemand"])
    
    return features, target, FEATURE_DEFINITIONS


class FeaturePipeline:
    """Pipelines one-hot encoding for calendar features and feature selection."""
    
    def __init__(self, selected_numeric_features: Optional[List[str]] = None):
        self.selected_numeric_features = (
            selected_numeric_features if selected_numeric_features is not None else NUMERIC_FEATURE_NAMES
        )
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.encoded_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePipeline":
        """Fit the categorical encoder on training data."""
        cal_data = X[["DayOfWeek", "Month"]]
        self.encoder.fit(cal_data)
        
        cal_encoded_names = list(self.encoder.get_feature_names_out(["DayOfWeek", "Month"]))
        self.encoded_feature_names_ = (
            list(self.selected_numeric_features) + ["IsWeekend"] + cal_encoded_names
        )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw tabular features into an encoded numeric array."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform.")

        num_part = X[self.selected_numeric_features].values
        weekend_part = X[["IsWeekend"]].values
        cal_data = X[["DayOfWeek", "Month"]]
        cal_encoded = self.encoder.transform(cal_data)

        return np.hstack([num_part, weekend_part, cal_encoded])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def select_features_for_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """Execute feature selection for Linear Regression based on training data and validation MAE.
    
    Procedure:
    - Separate numeric lag/rolling features from categorical calendar features.
    - Calculate absolute Pearson correlation with training target.
    - Remove numeric features with zero variance.
    - Identify pairs of numeric features with absolute inter-feature correlation > 0.95.
    - Prefer feature with higher target correlation in redundant pairs.
    - Retain at least three usable numeric predictors when possible.
    - Fit full and selected configurations; compare validation MAE.
    - Select winning feature configuration with lower validation MAE (tie-break: smaller set).
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        X_val: Validation features DataFrame.
        y_val: Validation target Series.
        
    Returns:
        Dictionary with candidate_features, selected_features, rejected_features,
        correlations, redundancy_pairs, selection_reasons, and validation MAEs.
    """
    candidate_features = list(NUMERIC_FEATURE_NAMES)
    rejected_features = []
    selection_reasons: Dict[str, str] = {}
    target_correlations: Dict[str, float] = {}

    # 1. Target correlations and zero variance check
    surviving_numeric = []
    for feat in candidate_features:
        s = X_train[feat]
        std_val = s.std(ddof=1)
        if std_val == 0 or np.isnan(std_val):
            rejected_features.append(feat)
            selection_reasons[feat] = "Zero variance in training period."
            target_correlations[feat] = 0.0
            continue

        r = np.corrcoef(s.values, y_train.values)[0, 1]
        abs_r = 0.0 if np.isnan(r) else abs(float(r))
        target_correlations[feat] = round(abs_r, 4)
        surviving_numeric.append(feat)

    # 2. Inter-feature correlation check (> 0.95)
    redundancy_pairs = []
    dropped_for_redundancy = set()

    for i in range(len(surviving_numeric)):
        for j in range(i + 1, len(surviving_numeric)):
            f1 = surviving_numeric[i]
            f2 = surviving_numeric[j]
            if f1 in dropped_for_redundancy or f2 in dropped_for_redundancy:
                continue

            r_inter = np.corrcoef(X_train[f1].values, X_train[f2].values)[0, 1]
            abs_inter = 0.0 if np.isnan(r_inter) else abs(float(r_inter))

            if abs_inter > 0.95:
                redundancy_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": round(abs_inter, 4),
                })
                # Check target correlations
                r1 = target_correlations.get(f1, 0.0)
                r2 = target_correlations.get(f2, 0.0)
                
                # Check if dropping would leave fewer than 3 features
                current_active = [f for f in surviving_numeric if f not in dropped_for_redundancy]
                if len(current_active) > 3:
                    if r1 >= r2:
                        dropped_for_redundancy.add(f2)
                        rejected_features.append(f2)
                        selection_reasons[f2] = (
                            f"Collinear with {f1} (r={abs_inter:.2f} > 0.95); "
                            f"{f1} has higher target correlation ({r1:.4f} vs {r2:.4f})."
                        )
                    else:
                        dropped_for_redundancy.add(f1)
                        rejected_features.append(f1)
                        selection_reasons[f1] = (
                            f"Collinear with {f2} (r={abs_inter:.2f} > 0.95); "
                            f"{f2} has higher target correlation ({r2:.4f} vs {r1:.4f})."
                        )

    selected_numeric = [f for f in surviving_numeric if f not in dropped_for_redundancy]
    
    # Ensure at least 3 usable numeric predictors if possible
    if len(selected_numeric) < 3 and len(surviving_numeric) >= 3:
        # Re-add best surviving by target correlation
        sorted_surv = sorted(surviving_numeric, key=lambda f: target_correlations.get(f, 0.0), reverse=True)
        selected_numeric = sorted_surv[:3]
        rejected_features = [f for f in candidate_features if f not in selected_numeric]
        for f in selected_numeric:
            selection_reasons.pop(f, None)

    for f in selected_numeric:
        selection_reasons[f] = f"Retained with training target correlation {target_correlations.get(f, 0.0):.4f}."

    # 3. Model validation comparison: Full vs Selected
    # Pipeline Full
    pipe_full = FeaturePipeline(selected_numeric_features=candidate_features)
    X_train_full = pipe_full.fit_transform(X_train)
    X_val_full = pipe_full.transform(X_val)
    
    lr_full = LinearRegression().fit(X_train_full, y_train)
    preds_full = np.clip(lr_full.predict(X_val_full), 0, None)
    mae_full = float(mean_absolute_error(y_val, preds_full))

    # Pipeline Selected
    pipe_sel = FeaturePipeline(selected_numeric_features=selected_numeric)
    X_train_sel = pipe_sel.fit_transform(X_train)
    X_val_sel = pipe_sel.transform(X_val)
    
    lr_sel = LinearRegression().fit(X_train_sel, y_train)
    preds_sel = np.clip(lr_sel.predict(X_val_sel), 0, None)
    mae_sel = float(mean_absolute_error(y_val, preds_sel))

    # Winner decision
    if mae_sel <= mae_full:
        final_selected_numeric = selected_numeric
    else:
        # Full performed strictly better
        final_selected_numeric = candidate_features
        rejected_features = []
        for f in candidate_features:
            selection_reasons[f] = "Retained in full feature set (validation MAE was superior)."

    return {
        "candidate_features": candidate_features,
        "selected_features": final_selected_numeric,
        "rejected_features": rejected_features,
        "target_correlations": target_correlations,
        "redundancy_pairs": redundancy_pairs,
        "selection_reasons": selection_reasons,
        "validation_mae_full": round(mae_full, 4),
        "validation_mae_selected": round(mae_sel, 4),
        "chosen_set": "selected" if final_selected_numeric == selected_numeric else "full",
    }
