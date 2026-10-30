"""Recommendation Service synthesizing forecasts, fuzzy risk, and optimization into actionable decisions.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from config import GA_CONFIG, OBJECTIVE_WEIGHTS
from core.fuzzy_engine import evaluate_fuzzy_urgency
from core.greedy_baseline import (
    compute_normalization_scales,
    compute_objective_components,
    simulate_inventory_for_product,
    solve_greedy_baseline,
)
from core.inventory import validate_inventory_config
from core.optimization import GeneticAlgorithmOptimizer
from core.schemas import (
    ForecastResult,
    FuzzyResult,
    OptimizationMetrics,
    ProductRecommendation,
    RecommendationReport,
    SchemaValidationError,
)


def evaluate_model_paired_errors(
    errors_model_a: List[float], errors_model_b: List[float]
) -> Dict[str, Any]:
    """Perform statistical paired comparison of validation forecast errors.
    
    Reports mean/median paired difference and Wilcoxon signed-rank test when appropriate.
    Documents limitations for small samples or zero variance.
    
    Args:
        errors_model_a: Absolute forecast errors of Model A.
        errors_model_b: Absolute forecast errors of Model B.
        
    Returns:
        Dictionary containing test statistics and documented limitations.
    """
    ea = np.asarray(errors_model_a, dtype=float)
    eb = np.asarray(errors_model_b, dtype=float)
    n = len(ea)

    if n != len(eb):
        raise ValueError(f"Error lists must be of equal length (got {n} and {len(eb)}).")

    if n == 0:
        return {
            "status": "empty",
            "n_observations": 0,
            "mean_paired_difference": 0.0,
            "median_paired_difference": 0.0,
            "test_used": "None",
            "statistic": None,
            "p_value": None,
            "interpretation": "No matched evaluation errors available.",
        }

    diffs = ea - eb
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    non_zero_diffs = diffs[diffs != 0]

    if len(non_zero_diffs) < 5:
        return {
            "status": "insufficient_non_zero_differences",
            "n_observations": n,
            "mean_paired_difference": round(mean_diff, 4),
            "median_paired_difference": round(median_diff, 4),
            "test_used": "Wilcoxon signed-rank test",
            "statistic": None,
            "p_value": None,
            "interpretation": (
                "Insufficient non-zero paired differences (<5) to conduct a valid Wilcoxon signed-rank test. "
                "Small sample size limitation: aggregate block differences should be interpreted with caution."
            ),
        }

    try:
        w_stat, p_val = stats.wilcoxon(ea, eb, zero_method="wilcox", alternative="two-sided")
        stat_sig = p_val < 0.05
        interp = (
            f"Statistically significant error difference (p={p_val:.4f} < 0.05). "
            f"{'Model A has lower error' if mean_diff < 0 else 'Model B has lower error'}."
            if stat_sig
            else f"No statistically significant difference in forecast errors (p={p_val:.4f} >= 0.05)."
        )
        return {
            "status": "success",
            "n_observations": n,
            "mean_paired_difference": round(mean_diff, 4),
            "median_paired_difference": round(median_diff, 4),
            "test_used": "Wilcoxon signed-rank test",
            "statistic": round(float(w_stat), 4),
            "p_value": round(float(p_val), 4),
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "status": "error",
            "n_observations": n,
            "mean_paired_difference": round(mean_diff, 4),
            "median_paired_difference": round(median_diff, 4),
            "test_used": "Wilcoxon signed-rank test",
            "statistic": None,
            "p_value": None,
            "interpretation": f"Statistical test could not be computed: {e}",
        }


def evaluate_ga_stability(
    products_data: List[Dict[str, Any]],
    budget_gbp: float,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Execute the Genetic Algorithm across multiple random seeds to measure solution stability.
    
    Default seeds from Section 17: 11, 21, 31, 41, 51.
    
    Args:
        products_data: List of product specifications.
        budget_gbp: Purchasing budget.
        seeds: Optional list of seeds.
        
    Returns:
        Dictionary with mean, std, best, worst objective, runtime, and feasibility rate.
    """
    eval_seeds = seeds if seeds is not None else [11, 21, 31, 41, 51]
    objectives = []
    runtimes = []
    feasibility = []

    for s in eval_seeds:
        opt = GeneticAlgorithmOptimizer(random_seed=s)
        res = opt.optimize(products_data, budget_gbp)
        objectives.append(res.metrics.ObjectiveValue)
        runtimes.append(res.metrics.RuntimeSeconds)
        feasibility.append(res.metrics.Feasible)

    return {
        "seeds_evaluated": eval_seeds,
        "mean_objective": round(float(np.mean(objectives)), 6),
        "std_objective": round(float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0, 6),
        "best_objective": round(float(np.min(objectives)), 6),
        "worst_objective": round(float(np.max(objectives)), 6),
        "mean_runtime": round(float(np.mean(runtimes)), 4),
        "feasibility_rate": round(float(sum(feasibility) / len(feasibility)) * 100.0, 2),
    }


def generate_recommendations(
    forecast_results: Dict[str, ForecastResult],
    inventory_config: pd.DataFrame,
    budget_gbp: float,
    planning_horizon: int,
    run_stability_check: bool = False,
) -> RecommendationReport:
    """Execute complete end-to-end recommendation workflow from forecasts to purchasing orders.
    
    Pipeline:
    1. Validate inventory configuration.
    2. Compute fuzzy urgency risk for each product.
    3. Run zero-order baseline.
    4. Run greedy optimization.
    5. Run Genetic Algorithm.
    6. Validate final chromosome against all constraints.
    7. Generate detailed product recommendations.
    
    Args:
        forecast_results: Dictionary mapping stock_code to ForecastResult.
        inventory_config: Inventory DataFrame.
        budget_gbp: Budget limit.
        planning_horizon: Horizon H (7, 14, 30).
        run_stability_check: Whether to execute the 5-seed stability test.
        
    Returns:
        RecommendationReport dataclass instance.
    """
    # 1. Validate configuration
    validate_inventory_config(inventory_config)

    # 2. Match products and compute Fuzzy Urgencies
    products_data: List[Dict[str, Any]] = []
    fuzzy_risks: Dict[str, FuzzyResult] = {}

    for _, row in inventory_config.iterrows():
        code = str(row["StockCode"])
        if code not in forecast_results:
            continue

        fc_res = forecast_results[code]
        desc = str(row["Description"])
        curr_stock = int(row["CurrentStock"])
        lead_time = int(row["LeadTimeDays"])
        cost = float(row["SupplierCostGBP"])
        cap = int(row["StorageCapacity"])
        moq = int(row["MinimumOrderQuantity"])
        multiple = int(row["OrderMultiple"])

        # Extended forecast if lead time > horizon
        lt_forecast = fc_res.model_parameters.get(
            "extended_lead_time_predictions", fc_res.daily_predictions
        )

        fuzzy_res = evaluate_fuzzy_urgency(
            current_stock=curr_stock,
            lead_time_days=lead_time,
            planning_horizon_days=planning_horizon,
            lead_time_forecast=lt_forecast,
            uncertainty_score=fc_res.uncertainty_score,
        )
        fuzzy_risks[code] = fuzzy_res

        products_data.append({
            "StockCode": code,
            "Description": desc,
            "CurrentStock": curr_stock,
            "SupplierCostGBP": cost,
            "LeadTimeDays": lead_time,
            "StorageCapacity": cap,
            "MinimumOrderQuantity": moq,
            "OrderMultiple": multiple,
            "UrgencyScore": fuzzy_res.centroid,
            "DailyForecast": fc_res.daily_predictions[:planning_horizon],
        })

    if not products_data:
        raise ValueError("No matching products with valid forecasts and inventory configuration.")

    # 3. Compute normalization scales
    scales = compute_normalization_scales(products_data, budget_gbp)

    # 4. Zero-Order Baseline
    zero_chrom = [0] * len(products_data)
    zero_obj, zero_comp = compute_objective_components(
        zero_chrom, products_data, budget_gbp, scales, OBJECTIVE_WEIGHTS
    )
    total_short_zero = sum(
        simulate_inventory_for_product(
            0, p["CurrentStock"], p["SupplierCostGBP"], p["LeadTimeDays"], p["DailyForecast"]
        )[0]
        for p in products_data
    )
    zero_metrics = OptimizationMetrics(
        ObjectiveValue=round(zero_obj, 6),
        TotalPurchaseCost=0.0,
        BudgetRemaining=round(budget_gbp, 2),
        TotalExpectedShortage=round(total_short_zero, 2),
        TotalExpectedExcess=round(zero_comp["TotalExcess"], 2),
        CoverageImbalance=round(zero_comp["CoverageImbalance"], 4),
        BudgetUtilizationPercent=0.0,
        RuntimeSeconds=0.0001,
        Feasible=True,
    )

    # 5. Greedy Optimizer
    greedy_res = solve_greedy_baseline(products_data, budget_gbp, scales, OBJECTIVE_WEIGHTS)

    # 6. Genetic Algorithm Optimizer
    ga_opt = GeneticAlgorithmOptimizer()
    ga_res = ga_opt.optimize(products_data, budget_gbp, greedy_baseline_result=greedy_res)

    # 7. Select Final Solution (GA is invariant-guaranteed to be <= greedy)
    final_res = ga_res if ga_res.metrics.ObjectiveValue <= greedy_res.metrics.ObjectiveValue else greedy_res
    best_chrom = final_res.chromosome

    # 8. Product-level Recommendations
    recommendations: List[ProductRecommendation] = []
    
    for i, p in enumerate(products_data):
        code = p["StockCode"]
        rec_q = best_chrom[i]
        cost_unit = p["SupplierCostGBP"]
        total_order_cost = round(rec_q * cost_unit, 2)

        # Before order (q=0)
        short_before, _, _, _ = simulate_inventory_for_product(
            q=0,
            current_stock=p["CurrentStock"],
            supplier_cost=cost_unit,
            lead_time_days=p["LeadTimeDays"],
            daily_forecast=p["DailyForecast"],
        )

        # After order (q=rec_q)
        short_after, ending_stock, _, _ = simulate_inventory_for_product(
            q=rec_q,
            current_stock=p["CurrentStock"],
            supplier_cost=cost_unit,
            lead_time_days=p["LeadTimeDays"],
            daily_forecast=p["DailyForecast"],
        )

        f_risk = fuzzy_risks[code]
        fc_demand = float(sum(p["DailyForecast"]))

        recommendations.append(
            ProductRecommendation(
                StockCode=code,
                Description=p["Description"],
                ForecastDemand=round(fc_demand, 2),
                CurrentStock=p["CurrentStock"],
                LeadTimeDays=p["LeadTimeDays"],
                RiskScore=f_risk.centroid,
                RiskCategory=f_risk.category,
                RecommendedQuantity=rec_q,
                SupplierCostGBP=cost_unit,
                TotalOrderCostGBP=total_order_cost,
                ExpectedShortageBefore=round(short_before, 2),
                ExpectedShortageAfter=round(short_after, 2),
                ExpectedEndingStock=round(ending_stock, 2),
            )
        )

    # 9. Stability analysis if requested
    stability_data = None
    if run_stability_check:
        stability_data = evaluate_ga_stability(products_data, budget_gbp)

    summary_metrics = {
        "products_evaluated": len(products_data),
        "total_budget_gbp": budget_gbp,
        "total_recommended_spend_gbp": final_res.metrics.TotalPurchaseCost,
        "budget_remaining_gbp": final_res.metrics.BudgetRemaining,
        "budget_utilization_pct": final_res.metrics.BudgetUtilizationPercent,
        "total_shortage_reduction": round(
            zero_metrics.TotalExpectedShortage - final_res.metrics.TotalExpectedShortage, 2
        ),
        "shortage_reduction_pct": round(
            (
                (zero_metrics.TotalExpectedShortage - final_res.metrics.TotalExpectedShortage)
                / max(zero_metrics.TotalExpectedShortage, 0.01)
            )
            * 100.0,
            2,
        ),
        "ga_vs_greedy_improvement_pct": round(
            (
                (greedy_res.metrics.ObjectiveValue - final_res.metrics.ObjectiveValue)
                / max(greedy_res.metrics.ObjectiveValue, 1e-6)
            )
            * 100.0,
            4,
        ),
    }

    return RecommendationReport(
        budget_gbp=budget_gbp,
        planning_horizon=planning_horizon,
        summary_metrics=summary_metrics,
        recommendations=recommendations,
        zero_order_metrics=zero_metrics,
        greedy_metrics=greedy_res.metrics,
        ga_metrics=ga_res.metrics,
        fuzzy_risks=fuzzy_risks,
        stability_analysis=stability_data,
        explanations={
            "normalization_scales": scales,
            "ga_explanation_events": ga_res.explanation_events,
            "generation_history": ga_res.generation_history,
        },
    )
