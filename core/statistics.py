"""Statistical analysis engine for retail and demand time series.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def describe_numeric(series: pd.Series) -> Dict[str, Any]:
    """Calculate comprehensive descriptive statistics for a numeric series.
    
    Uses sample variance and standard deviation with ddof=1.
    
    Args:
        series: Numeric pandas Series.
        
    Returns:
        Dictionary containing metric values and formulas metadata.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean_s)
    
    formulas = {
        "mean": "sum(x_i) / n",
        "sample_variance": "sum((x_i - mean)^2) / (n - 1) [ddof=1]",
        "sample_std": "sqrt(sample_variance) [ddof=1]",
        "q1": "25th percentile (linear interpolation)",
        "q3": "75th percentile (linear interpolation)",
        "IQR": "q3 - q1",
        "skewness": "Fisher-Pearson standardized skewness coefficient",
    }
    
    if n == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "sample_variance": 0.0,
            "sample_std": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "IQR": 0.0,
            "skewness": 0.0,
            "formulas": formulas,
        }
        
    mean_val = float(clean_s.mean())
    median_val = float(clean_s.median())
    variance_val = float(clean_s.var(ddof=1)) if n > 1 else 0.0
    std_val = float(clean_s.std(ddof=1)) if n > 1 else 0.0
    min_val = float(clean_s.min())
    max_val = float(clean_s.max())
    q1_val = float(clean_s.quantile(0.25))
    q3_val = float(clean_s.quantile(0.75))
    iqr_val = float(q3_val - q1_val)
    skew_val = float(clean_s.skew()) if (n > 2 and std_val > 0) else 0.0

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "sample_variance": round(variance_val, 4),
        "sample_std": round(std_val, 4),
        "minimum": round(min_val, 4),
        "maximum": round(max_val, 4),
        "q1": round(q1_val, 4),
        "q3": round(q3_val, 4),
        "IQR": round(iqr_val, 4),
        "skewness": round(skew_val, 4),
        "formulas": formulas,
    }


def _interpret_correlation(r: float) -> str:
    """Provide descriptive label for a correlation coefficient magnitude."""
    if np.isnan(r):
        return "Undetermined / Constant Series"
    abs_r = abs(r)
    sign = "Positive" if r > 0 else ("Negative" if r < 0 else "Zero")
    if abs_r >= 0.8:
        strength = "Very Strong"
    elif abs_r >= 0.6:
        strength = "Strong"
    elif abs_r >= 0.4:
        strength = "Moderate"
    elif abs_r >= 0.2:
        strength = "Weak"
    else:
        strength = "Negligible"
    return f"{strength} {sign}" if abs_r >= 0.05 else "Negligible / Uncorrelated"


def calculate_correlations(
    dataframe: pd.DataFrame, columns: List[str]
) -> Dict[str, Any]:
    """Calculate Pearson and Spearman correlations with p-values and interpretations.
    
    Args:
        dataframe: Input DataFrame.
        columns: List of numeric column names to analyze.
        
    Returns:
        Dictionary with Pearson and Spearman correlation matrices, p-values, and labels.
    """
    valid_cols = [c for c in columns if c in dataframe.columns]
    data = dataframe[valid_cols].copy()
    
    n_cols = len(valid_cols)
    pearson_corr = np.zeros((n_cols, n_cols))
    pearson_pval = np.zeros((n_cols, n_cols))
    spearman_corr = np.zeros((n_cols, n_cols))
    spearman_pval = np.zeros((n_cols, n_cols))
    interpretations = {}

    for i, col1 in enumerate(valid_cols):
        for j, col2 in enumerate(valid_cols):
            if i == j:
                pearson_corr[i, j] = 1.0
                spearman_corr[i, j] = 1.0
                pearson_pval[i, j] = 0.0
                spearman_pval[i, j] = 0.0
                interpretations[f"{col1}_vs_{col2}"] = "Perfect Identity"
                continue

            sub = data[[col1, col2]].dropna()
            # Check constant column
            if len(sub) < 3 or sub[col1].std(ddof=1) == 0 or sub[col2].std(ddof=1) == 0:
                pearson_corr[i, j] = np.nan
                pearson_pval[i, j] = np.nan
                spearman_corr[i, j] = np.nan
                spearman_pval[i, j] = np.nan
                interpretations[f"{col1}_vs_{col2}"] = "Constant column (Zero variance)"
            else:
                p_r, p_p = stats.pearsonr(sub[col1], sub[col2])
                s_r, s_p = stats.spearmanr(sub[col1], sub[col2])
                pearson_corr[i, j] = p_r
                pearson_pval[i, j] = p_p
                spearman_corr[i, j] = s_r
                spearman_pval[i, j] = s_p
                interpretations[f"{col1}_vs_{col2}"] = _interpret_correlation(p_r)

    p_df = pd.DataFrame(pearson_corr, index=valid_cols, columns=valid_cols)
    p_p_df = pd.DataFrame(pearson_pval, index=valid_cols, columns=valid_cols)
    s_df = pd.DataFrame(spearman_corr, index=valid_cols, columns=valid_cols)
    s_p_df = pd.DataFrame(spearman_pval, index=valid_cols, columns=valid_cols)

    return {
        "columns": valid_cols,
        "n_observations": int(len(data.dropna(subset=valid_cols))),
        "pearson_matrix": p_df.to_dict(),
        "pearson_pvalues": p_p_df.to_dict(),
        "spearman_matrix": s_df.to_dict(),
        "spearman_pvalues": s_p_df.to_dict(),
        "interpretations": interpretations,
    }


def calculate_rolling_statistics(series: pd.Series) -> pd.DataFrame:
    """Calculate 7-day and 30-day rolling means and 7-day rolling sample standard deviation.
    
    Args:
        series: Daily numeric demand series.
        
    Returns:
        DataFrame with RollingMean7, RollingMean30, and RollingStd7 columns.
    """
    clean_s = pd.to_numeric(series, errors="coerce")
    rolling_df = pd.DataFrame(index=clean_s.index)
    rolling_df["RollingMean7"] = clean_s.rolling(window=7, min_periods=1).mean()
    rolling_df["RollingMean30"] = clean_s.rolling(window=30, min_periods=1).mean()
    rolling_df["RollingStd7"] = clean_s.rolling(window=7, min_periods=2).std(ddof=1).fillna(0.0)
    return rolling_df


def perform_adf_test(series: pd.Series) -> Dict[str, Any]:
    """Execute Augmented Dickey-Fuller (ADF) stationarity test.
    
    Args:
        series: Time series demand observations.
        
    Returns:
        Dictionary with test statistic, p-value, lag used, critical values, and interpretation.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean_s)

    if n < 15 or clean_s.std(ddof=1) == 0:
        return {
            "status": "ineligible",
            "reason": "Series too short (<15 observations) or has zero variance.",
            "statistic": None,
            "p_value": None,
            "used_lag": None,
            "n_observations": n,
            "critical_values": {},
            "interpretation": "Ineligible for ADF stationarity test due to insufficient length or variance.",
        }

    try:
        result = adfuller(clean_s, autolag="AIC")
        stat, pval, lag, nobs, crit, _ = result
        is_stationary = pval < 0.05
        interpretation = (
            f"Series is stationary (p-value = {pval:.4f} < 0.05). Null hypothesis rejected."
            if is_stationary
            else f"Series is non-stationary (p-value = {pval:.4f} >= 0.05). Null hypothesis cannot be rejected."
        )
        return {
            "status": "success",
            "statistic": round(float(stat), 4),
            "p_value": round(float(pval), 4),
            "used_lag": int(lag),
            "n_observations": int(nobs),
            "critical_values": {k: round(float(v), 4) for k, v in crit.items()},
            "is_stationary": bool(is_stationary),
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "statistic": None,
            "p_value": None,
            "used_lag": None,
            "n_observations": n,
            "critical_values": {},
            "interpretation": f"ADF test failed: {e}",
        }


def calculate_acf_pacf(series: pd.Series, max_lags: int = 30) -> Dict[str, Any]:
    """Calculate Autocorrelation (ACF) and Partial Autocorrelation (PACF) values.
    
    Args:
        series: Time series observations.
        max_lags: Maximum lags to calculate (default 30).
        
    Returns:
        Dictionary with ACF and PACF lists and corresponding lag indices.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean_s)

    if n < 10 or clean_s.std(ddof=1) == 0:
        return {
            "status": "ineligible",
            "acf": [],
            "pacf": [],
            "lags": [],
            "reason": "Series too short or constant.",
        }

    # Safe maximum lag for sample size
    safe_lags = min(max_lags, max(n // 2 - 1, 1))
    
    try:
        acf_vals = acf(clean_s, nlags=safe_lags, fft=True)
        # pacf requires nlags < n/2
        pacf_lags = min(safe_lags, max(n // 2 - 2, 1))
        pacf_vals = pacf(clean_s, nlags=pacf_lags, method="ywm")
        
        return {
            "status": "success",
            "lags": list(range(len(acf_vals))),
            "acf": [round(float(v), 4) for v in acf_vals],
            "pacf_lags": list(range(len(pacf_vals))),
            "pacf": [round(float(v), 4) for v in pacf_vals],
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "acf": [],
            "pacf": [],
            "lags": [],
        }


def decompose_time_series(series: pd.Series, period: int = 7) -> Dict[str, Any]:
    """Perform additive weekly seasonal decomposition when sufficient history exists.
    
    Args:
        series: Daily time series demand.
        period: Seasonal period (default 7 for weekly).
        
    Returns:
        Dictionary with trend, seasonal, and residual components.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean_s)

    # Seasonal decomposition requires at least 2 full periods
    if n < 2 * period or clean_s.std(ddof=1) == 0:
        return {
            "status": "ineligible",
            "reason": f"Insufficient observations for decomposition (got {n}, need at least {2 * period}).",
            "observed": clean_s.tolist(),
            "trend": [],
            "seasonal": [],
            "residual": [],
        }

    try:
        decomp = seasonal_decompose(clean_s, model="additive", period=period)
        return {
            "status": "success",
            "period": period,
            "observed": [round(float(v), 4) for v in clean_s],
            "trend": [round(float(v), 4) if pd.notna(v) else None for v in decomp.trend],
            "seasonal": [round(float(v), 4) if pd.notna(v) else None for v in decomp.seasonal],
            "residual": [round(float(v), 4) if pd.notna(v) else None for v in decomp.resid],
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "observed": clean_s.tolist(),
            "trend": [],
            "seasonal": [],
            "residual": [],
        }


def calculate_sales_summary(
    transactions: pd.DataFrame, filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute high-level transaction sales summary with optional filtering.
    
    Args:
        transactions: Cleaned transaction dataframe.
        filters: Optional dict with keys: 'stock_code', 'start_date', 'end_date', 'country'.
        
    Returns:
        Dictionary with aggregated sales metrics.
    """
    df = transactions.copy()
    if filters:
        if "stock_code" in filters and filters["stock_code"]:
            df = df[df["StockCode"] == str(filters["stock_code"])]
        if "start_date" in filters and filters["start_date"]:
            df = df[df["Date"] >= pd.to_datetime(filters["start_date"])]
        if "end_date" in filters and filters["end_date"]:
            df = df[df["Date"] <= pd.to_datetime(filters["end_date"])]
        if "country" in filters and filters["country"]:
            df = df[df["Country"] == str(filters["country"])]

    if df.empty:
        return {
            "total_rows": 0,
            "total_units": 0,
            "total_revenue_gbp": 0.0,
            "unique_invoices": 0,
            "unique_customers": 0,
            "average_order_value_gbp": 0.0,
            "min_date": None,
            "max_date": None,
        }

    total_units = int(df["Quantity"].sum())
    total_rev = float(df["RevenueGBP"].sum())
    n_invoices = int(df["InvoiceNo"].nunique())
    n_customers = int(df["CustomerID"].dropna().nunique())
    aov = total_rev / max(n_invoices, 1)

    return {
        "total_rows": len(df),
        "total_units": total_units,
        "total_revenue_gbp": round(total_rev, 2),
        "unique_invoices": n_invoices,
        "unique_customers": n_customers,
        "average_order_value_gbp": round(aov, 2),
        "min_date": str(df["Date"].min().strftime("%Y-%m-%d")),
        "max_date": str(df["Date"].max().strftime("%Y-%m-%d")),
    }
