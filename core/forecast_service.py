"""Forecast service orchestrating model comparison, final future forecast generation, and uncertainty scoring.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import DEFAULT_HORIZON
from core.forecast_evaluation import compare_and_select_models
from core.forecast_models import (
    fit_and_forecast_arima,
    fit_and_forecast_linear_regression,
    forecast_moving_average,
    forecast_seasonal_naive,
)
from core.schemas import ForecastResult, InsufficientHistoryError


class ForecastService:
    """Service to coordinate evaluation, model selection, refitting, and future prediction."""

    def __init__(self):
        self._cache: Dict[str, ForecastResult] = {}

    def forecast_product(
        self,
        daily_df: pd.DataFrame,
        stock_code: str,
        horizon: int = DEFAULT_HORIZON,
        product_description: str = "",
        min_lead_time_days: int = 0,
        use_cache: bool = True,
        include_arima: bool = True,
    ) -> ForecastResult:
        """Evaluate models, select the winner, refit on all history, and forecast future demand.
        
        Args:
            daily_df: Product daily demand series with 'Date' and 'DailyUnits'.
            stock_code: Product stock code.
            horizon: Primary forecast horizon H.
            product_description: Product name/description.
            min_lead_time_days: Supplier lead time days; if > horizon, extends forecast length.
            use_cache: Whether to return cached results if available.
            include_arima: Whether to include ARIMA in model tournament (default True).
            
        Returns:
            ForecastResult dataclass instance.
        """
        cache_key = f"{stock_code}_{horizon}_{min_lead_time_days}_{include_arima}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prod_series = daily_df.sort_values("Date").reset_index(drop=True)
        if len(prod_series) < 28 + (2 * horizon):
            raise InsufficientHistoryError(
                f"Product {stock_code} has insufficient history ({len(prod_series)} days) "
                f"for horizon {horizon} backtesting."
            )

        # 1. Backtest and select winner
        eval_result = compare_and_select_models(prod_series, horizon, include_arima=include_arima)
        winning_model = eval_result["winning_model"]
        selected_eval = eval_result["model_evaluations"][winning_model]
        val_metrics = selected_eval["validation_metrics"]
        test_metrics = selected_eval["test_metrics"]
        fs_details = eval_result["feature_selection"]

        # 2. Uncertainty score proxy
        # RelativeError = ValidationMAE / max(MeanDailyTrainingDemand, 0.1)
        # UncertaintyScore = clip(RelativeError, 0, 1)
        n_train = eval_result["train_days"]
        train_mean_demand = float(prod_series.iloc[:n_train]["DailyUnits"].mean())
        val_mae = val_metrics["MAE"] if val_metrics["MAE"] is not None else 1.0
        
        relative_error = val_mae / max(train_mean_demand, 0.1)
        uncertainty_score = float(np.clip(relative_error, 0.0, 1.0))

        warnings_list: List[str] = []
        if train_mean_demand < 0.5:
            warnings_list.append(
                "Extremely low average daily demand (<0.5 units/day); relative uncertainty score may be unstable."
            )

        # 3. Determine actual forecast length needed (at least max(horizon, min_lead_time_days))
        effective_horizon = max(horizon, min_lead_time_days)
        last_obs_date = pd.to_datetime(prod_series["Date"].max())
        future_dates = [
            (last_obs_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range(1, effective_horizon + 1)
        ]

        # 4. Refit selected model on ALL available history
        all_units = prod_series["DailyUnits"]
        all_dates = prod_series["Date"]
        model_params: Dict[str, Any] = {}

        if winning_model == "Seasonal Naive":
            final_preds, details = forecast_seasonal_naive(all_units, effective_horizon)
            model_params = details.get("parameters", {})
        elif winning_model == "Moving Average (7-day)":
            final_preds, details = forecast_moving_average(all_units, effective_horizon)
            model_params = details.get("parameters", {})
        elif winning_model == "Linear Regression":
            final_preds, details = fit_and_forecast_linear_regression(
                train_history=all_units,
                train_dates=all_dates,
                horizon=effective_horizon,
                selected_numeric_features=eval_result["selected_features"],
                future_start_date=last_obs_date + pd.Timedelta(days=1),
            )
            model_params = {
                "intercept": details.get("intercept"),
                "coefficients": details.get("coefficients"),
                "worked_example": details.get("worked_example"),
            }
        elif winning_model == "ARIMA":
            arima_order = eval_result["arima_details"].get("selected_order", (1, 1, 1))
            try:
                from statsmodels.tsa.arima.model import ARIMA
                fit_all = ARIMA(all_units.values[-120:], order=arima_order).fit()
                raw_arima = fit_all.forecast(steps=effective_horizon)
                final_preds = np.clip(raw_arima, 0, None)
                model_params = {
                    "order": arima_order,
                    "aic": round(float(fit_all.aic), 4),
                    "bic": round(float(fit_all.bic), 4),
                }
            except Exception as e:
                warnings_list.append(f"ARIMA refit on full history failed ({e}); used moving average.")
                final_preds, _ = forecast_moving_average(all_units, effective_horizon)
                model_params = {"fallback": "Moving Average (7-day)"}
        else:
            final_preds, _ = forecast_moving_average(all_units, effective_horizon)

        # Truncate primary predictions to the requested horizon for the primary target,
        # but keep full length if lead time needed
        primary_preds = [round(float(v), 4) for v in final_preds[:horizon]]
        total_demand = round(float(sum(primary_preds)), 4)

        result = ForecastResult(
            stock_code=stock_code,
            product_description=product_description,
            historical_start=str(prod_series["Date"].min().strftime("%Y-%m-%d")),
            forecast_origin=str(last_obs_date.strftime("%Y-%m-%d")),
            forecast_horizon=horizon,
            forecast_dates=future_dates[:horizon],
            daily_predictions=primary_preds,
            total_predicted_demand=total_demand,
            selected_model=winning_model,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            all_model_results=eval_result["summary_table"],
            selected_features=fs_details.get("selected_features", []),
            rejected_features=fs_details.get("rejected_features", []),
            feature_selection_details=fs_details,
            model_parameters=model_params,
            uncertainty_score=round(uncertainty_score, 4),
            warnings=warnings_list,
        )

        # Store extended lead-time forecast inside model_parameters for downstream use
        if effective_horizon > horizon:
            result.model_parameters["extended_lead_time_predictions"] = [
                round(float(v), 4) for v in final_preds
            ]

        if use_cache:
            self._cache[cache_key] = result
        return result
