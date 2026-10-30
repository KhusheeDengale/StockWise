"""Reporting, export, and tabular presentation utilities for StockWise.
"""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from core.schemas import RecommendationReport


def export_results_csv(
    result_type: str,
    data: Union[RecommendationReport, pd.DataFrame, List[Dict[str, Any]]],
    filepath: Optional[Union[str, Path]] = None,
) -> str:
    """Export pipeline results to CSV format or file.
    
    Supported result_types:
    - 'recommendations': Product purchasing recommendations
    - 'model_comparison': Backtesting performance comparison
    - 'inventory': Inventory configuration
    - 'features': Supervised feature definitions
    
    Args:
        result_type: Type of results to export.
        data: Data object or DataFrame.
        filepath: Optional path to save CSV.
        
    Returns:
        CSV string representation.
    """
    if isinstance(data, RecommendationReport):
        if result_type == "recommendations":
            rows = [
                {
                    "StockCode": r.StockCode,
                    "Description": r.Description,
                    "ForecastDemand": r.ForecastDemand,
                    "CurrentStock": r.CurrentStock,
                    "LeadTimeDays": r.LeadTimeDays,
                    "RiskScore": r.RiskScore,
                    "RiskCategory": r.RiskCategory,
                    "RecommendedQuantity": r.RecommendedQuantity,
                    "SupplierCostGBP": r.SupplierCostGBP,
                    "TotalOrderCostGBP": r.TotalOrderCostGBP,
                    "ExpectedShortageBefore": r.ExpectedShortageBefore,
                    "ExpectedShortageAfter": r.ExpectedShortageAfter,
                    "ExpectedEndingStock": r.ExpectedEndingStock,
                }
                for r in data.recommendations
            ]
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame([data.summary_metrics])
    elif isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported data type for export: {type(data)}")

    csv_str = df.to_csv(index=False)
    if filepath:
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(csv_str, encoding="utf-8")

    return csv_str


def format_recommendation_summary_table(report: RecommendationReport) -> pd.DataFrame:
    """Format baseline comparison table (Zero-order vs Greedy vs GA)."""
    rows = [
        {
            "Algorithm": "Zero-Order Baseline",
            "Objective": report.zero_order_metrics.ObjectiveValue,
            "Total Spend (£)": report.zero_order_metrics.TotalPurchaseCost,
            "Budget Remaining (£)": report.zero_order_metrics.BudgetRemaining,
            "Expected Shortage": report.zero_order_metrics.TotalExpectedShortage,
            "Expected Excess": report.zero_order_metrics.TotalExpectedExcess,
            "Budget Util (%)": report.zero_order_metrics.BudgetUtilizationPercent,
            "Runtime (s)": report.zero_order_metrics.RuntimeSeconds,
            "Feasible": report.zero_order_metrics.Feasible,
        },
        {
            "Algorithm": "Greedy Heuristic",
            "Objective": report.greedy_metrics.ObjectiveValue,
            "Total Spend (£)": report.greedy_metrics.TotalPurchaseCost,
            "Budget Remaining (£)": report.greedy_metrics.BudgetRemaining,
            "Expected Shortage": report.greedy_metrics.TotalExpectedShortage,
            "Expected Excess": report.greedy_metrics.TotalExpectedExcess,
            "Budget Util (%)": report.greedy_metrics.BudgetUtilizationPercent,
            "Runtime (s)": report.greedy_metrics.RuntimeSeconds,
            "Feasible": report.greedy_metrics.Feasible,
        },
        {
            "Algorithm": "Genetic Algorithm",
            "Objective": report.ga_metrics.ObjectiveValue,
            "Total Spend (£)": report.ga_metrics.TotalPurchaseCost,
            "Budget Remaining (£)": report.ga_metrics.BudgetRemaining,
            "Expected Shortage": report.ga_metrics.TotalExpectedShortage,
            "Expected Excess": report.ga_metrics.TotalExpectedExcess,
            "Budget Util (%)": report.ga_metrics.BudgetUtilizationPercent,
            "Runtime (s)": report.ga_metrics.RuntimeSeconds,
            "Feasible": report.ga_metrics.Feasible,
        },
    ]
    return pd.DataFrame(rows)
