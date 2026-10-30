"""Chronological backtesting, non-overlapping block evaluation, and model selection.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from core.features import select_features_for_regression
from core.forecast_models import (
    fit_and_forecast_arima,
    fit_and_forecast_linear_regression,
    forecast_moving_average,
    forecast_seasonal_naive,
)
from core.schemas import InsufficientHistoryError


def calculate_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> Dict[str, Optional[float]]:
    """Compute exact evaluation metrics: MAE, RMSE, WAPE, and R2.
    
    Args:
        actual: Array of actual demand units.
        predicted: Array of predicted demand units.
        
    Returns:
        Dictionary with MAE, RMSE, WAPE (%), and R2.
    """
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    if len(act) == 0:
        return {"MAE": None, "RMSE": None, "WAPE": None, "R2": None}

    errors = act - pred
    abs_errors = np.abs(errors)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # WAPE = 100 * sum(|act - pred|) / sum(act)
    sum_actual = float(np.sum(act))
    if sum_actual > 0:
        wape = float(100.0 * np.sum(abs_errors) / sum_actual)
    else:
        wape = 0.0 if np.all(abs_errors == 0) else None

    # R2 = 1 - sum((act - pred)^2) / sum((act - mean(act))^2)
    mean_act = np.mean(act)
    ss_tot = float(np.sum((act - mean_act) ** 2))
    ss_res = float(np.sum(errors ** 2))
    if ss_tot > 1e-9:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = None  # Ineligible when variance is zero

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4) if wape is not None else None,
        "R2": round(r2, 4) if r2 is not None else None,
    }


def split_chronological_series(
    daily_df: pd.DataFrame, horizon: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a product's daily demand series into 70% train, 15% validation, and 15% test.
    
    Requires that validation and test segments each contain at least one complete horizon H.
    
    Args:
        daily_df: Daily demand dataframe sorted by Date.
        horizon: Forecast horizon H.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
        
    Raises:
        InsufficientHistoryError: If history is too short to evaluate.
    """
    n = len(daily_df)
    min_required = 28 + (2 * horizon)  # warm-up + at least 1 block val + 1 block test
    if n < min_required:
        raise InsufficientHistoryError(
            f"Series has {n} days; requires at least {min_required} days "
            f"for warm-up (28) and non-overlapping validation/test blocks (horizon={horizon})."
        )

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    # Ensure validation and test have at least horizon H
    if n_val < horizon:
        n_val = horizon
    n_test = n - (n_train + n_val)
    if n_test < horizon:
        n_test = horizon
        n_train = n - n_val - n_test

    if n_train < 28:
        raise InsufficientHistoryError(
            f"Training segment has {n_train} days; requires at least 28 days for feature warm-up."
        )

    train_df = daily_df.iloc[:n_train].copy()
    val_df = daily_df.iloc[n_train : n_train + n_val].copy()
    test_df = daily_df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df


def evaluate_model_on_blocks(
    model_name: str,
    full_history_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    horizon: int,
    selected_features: Optional[List[str]] = None,
    arima_order: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Evaluate a forecasting model using non-overlapping H-day blocks.
    
    For each block of length H:
    - Origin moves forward by H days.
    - Model uses strictly data available up to the origin.
    - Model does not see any actual demand inside the forecast block.
    
    Args:
        model_name: 'Seasonal Naive', 'Moving Average (7-day)', 'Linear Regression', or 'ARIMA'.
        full_history_df: Complete historical dataframe.
        start_idx: Starting index of the evaluation segment (e.g. start of val or test).
        end_idx: Ending index of the evaluation segment.
        horizon: Block length H.
        selected_features: Features for Linear Regression.
        arima_order: Pre-selected order for ARIMA.
        
    Returns:
        Tuple of (concatenated_actuals, concatenated_predictions, block_details).
    """
    all_actuals = []
    all_predictions = []
    block_records = []

    # Calculate candidate block origins
    candidate_origins = list(range(start_idx, end_idx - horizon + 1, horizon))
    # Evaluate up to 4 most recent non-overlapping blocks for fast and representative testing
    if len(candidate_origins) > 4:
        candidate_origins = candidate_origins[-4:]

    for current_origin in candidate_origins:
        block_actuals = full_history_df.iloc[current_origin : current_origin + horizon]["DailyUnits"].values
        hist_so_far = full_history_df.iloc[:current_origin]
        hist_units = hist_so_far["DailyUnits"]
        hist_dates = hist_so_far["Date"]
        future_start_dt = full_history_df.iloc[current_origin]["Date"]

        if model_name == "Seasonal Naive":
            preds, _ = forecast_seasonal_naive(hist_units, horizon)
        elif model_name == "Moving Average (7-day)":
            preds, _ = forecast_moving_average(hist_units, horizon)
        elif model_name == "Linear Regression":
            preds, _ = fit_and_forecast_linear_regression(
                hist_units,
                hist_dates,
                horizon,
                selected_numeric_features=selected_features,
                future_start_date=future_start_dt,
            )
        elif model_name == "ARIMA":
            # For fast backtesting blocks, use recent history (last 120 days) and preselected order
            clean_units = hist_units.values[-120:]
            if arima_order is not None and len(clean_units) >= 15:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    fit_res = ARIMA(clean_units, order=arima_order).fit()
                    preds = np.clip(fit_res.forecast(steps=horizon), 0, None)
                except Exception:
                    preds, _ = forecast_moving_average(hist_units, horizon)
            else:
                preds, _ = fit_and_forecast_arima(hist_units, None, horizon)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        all_actuals.extend(block_actuals)
        all_predictions.extend(preds)
        block_records.append({
            "origin_date": str(hist_dates.iloc[-1].strftime("%Y-%m-%d")),
            "block_start": str(future_start_dt.strftime("%Y-%m-%d")),
            "actuals": [round(float(v), 2) for v in block_actuals],
            "predictions": [round(float(v), 2) for v in preds],
        })

    return np.array(all_actuals), np.array(all_predictions), block_records


def compare_and_select_models(
    daily_df: pd.DataFrame, horizon: int, include_arima: bool = True
) -> Dict[str, Any]:
    """Execute complete chronological evaluation and select the winning model.
    
    Procedure:
    - 70% train, 15% validation, 15% test.
    - Feature selection for Linear Regression on train/validation.
    - Evaluate candidate models on non-overlapping validation blocks.
    - Winner selected by validation WAPE (or MAE).
    - Tie-breaking: lower MAE, lower RMSE, simpler model in order:
      Seasonal Naive, Moving Average, Linear Regression, ARIMA.
    - Refit winner on train + val and evaluate on held-out test blocks.
    
    Args:
        daily_df: Product daily demand series with 'Date' and 'DailyUnits'.
        horizon: Forecast horizon H.
        include_arima: Whether to include ARIMA in model tournament (default True).
        
    Returns:
        Comprehensive comparison dictionary with metrics table, winner, and block predictions.
    """
    train_df, val_df, test_df = split_chronological_series(daily_df, horizon)
    
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    
    val_start_idx = n_train
    val_end_idx = n_train + n_val
    test_start_idx = val_end_idx
    test_end_idx = len(daily_df)

    # 1. Feature selection for Linear Regression
    from core.features import build_supervised_features
    X_train_sup, y_train_sup, _ = build_supervised_features(train_df["DailyUnits"], train_df["Date"])
    X_val_sup, y_val_sup, _ = build_supervised_features(val_df["DailyUnits"], val_df["Date"])
    
    if len(X_val_sup) >= 5 and len(X_train_sup) >= 10:
        fs_results = select_features_for_regression(X_train_sup, y_train_sup, X_val_sup, y_val_sup)
        selected_lr_features = fs_results["selected_features"]
    else:
        from core.features import NUMERIC_FEATURE_NAMES
        selected_lr_features = list(NUMERIC_FEATURE_NAMES)
        fs_results = {
            "candidate_features": list(NUMERIC_FEATURE_NAMES),
            "selected_features": selected_lr_features,
            "rejected_features": [],
            "target_correlations": {},
            "redundancy_pairs": [],
            "selection_reasons": {f: "Retained (default fallback)" for f in selected_lr_features},
            "validation_mae_full": 0.0,
            "validation_mae_selected": 0.0,
        }

    # 2. Fit candidate ARIMA on train and find best order if enabled
    if include_arima:
        _, arima_details = fit_and_forecast_arima(train_df["DailyUnits"], val_df["DailyUnits"], horizon)
        arima_order = arima_details.get("selected_order")
        models_to_test = [
            "Seasonal Naive",
            "Moving Average (7-day)",
            "Linear Regression",
            "ARIMA",
        ]
    else:
        arima_details = {"status": "skipped_fast_mode", "selected_order": None}
        arima_order = None
        models_to_test = [
            "Seasonal Naive",
            "Moving Average (7-day)",
            "Linear Regression",
        ]

    model_evaluations: Dict[str, Any] = {}
    
    for name in models_to_test:
        val_act, val_pred, val_blocks = evaluate_model_on_blocks(
            model_name=name,
            full_history_df=daily_df,
            start_idx=val_start_idx,
            end_idx=val_end_idx,
            horizon=horizon,
            selected_features=selected_lr_features,
            arima_order=arima_order,
        )
        val_metrics = calculate_metrics(val_act, val_pred)
        model_evaluations[name] = {
            "validation_metrics": val_metrics,
            "validation_actuals": [round(float(v), 2) for v in val_act],
            "validation_predictions": [round(float(v), 2) for v in val_pred],
            "validation_blocks": val_blocks,
            "status": "evaluated",
        }

    # 4. Model Selection Rules
    # Metric: validation WAPE if total demand > 0, else MAE
    # Tie-breaks: MAE, RMSE, simplicity order
    simplicity_rank = {
        "Seasonal Naive": 1,
        "Moving Average (7-day)": 2,
        "Linear Regression": 3,
        "ARIMA": 4,
    }

    def selection_key(m_name: str) -> Tuple[float, float, float, int]:
        m = model_evaluations[m_name]["validation_metrics"]
        wape = m["WAPE"] if (m["WAPE"] is not None and not np.isnan(m["WAPE"])) else float("inf")
        mae = m["MAE"] if (m["MAE"] is not None and not np.isnan(m["MAE"])) else float("inf")
        rmse = m["RMSE"] if (m["RMSE"] is not None and not np.isnan(m["RMSE"])) else float("inf")
        rank = simplicity_rank.get(m_name, 99)
        return (wape, mae, rmse, rank)

    sorted_models = sorted(models_to_test, key=selection_key)
    winning_model = sorted_models[0]

    # 5. Evaluate winning model (and others for comparison) on held-out test blocks
    # Note: Test performance NEVER changes the winner!
    for name in models_to_test:
        test_act, test_pred, test_blocks = evaluate_model_on_blocks(
            model_name=name,
            full_history_df=daily_df,
            start_idx=test_start_idx,
            end_idx=test_end_idx,
            horizon=horizon,
            selected_features=selected_lr_features,
            arima_order=arima_order,
        )
        test_metrics = calculate_metrics(test_act, test_pred)
        model_evaluations[name]["test_metrics"] = test_metrics
        model_evaluations[name]["test_actuals"] = [round(float(v), 2) for v in test_act]
        model_evaluations[name]["test_predictions"] = [round(float(v), 2) for v in test_pred]
        model_evaluations[name]["test_blocks"] = test_blocks

    # 6. Build comparison summary table
    summary_rows = []
    for name in models_to_test:
        vm = model_evaluations[name]["validation_metrics"]
        tm = model_evaluations[name]["test_metrics"]
        summary_rows.append({
            "Model": name,
            "ValidationMAE": vm["MAE"],
            "ValidationRMSE": vm["RMSE"],
            "ValidationWAPE": vm["WAPE"],
            "TestMAE": tm["MAE"],
            "TestRMSE": tm["RMSE"],
            "TestWAPE": tm["WAPE"],
            "TestR2": tm["R2"],
            "Selected": name == winning_model,
            "Status": "Held-out evaluation completed",
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "winning_model": winning_model,
        "summary_table": summary_df.to_dict(orient="records"),
        "model_evaluations": model_evaluations,
        "feature_selection": fs_results,
        "arima_details": arima_details,
        "selected_features": selected_lr_features,
        "train_days": n_train,
        "validation_days": n_val,
        "test_days": n_test,
    }
