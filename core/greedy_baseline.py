"""Greedy baseline heuristic optimizer for retail inventory replenishment.
"""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.schemas import OptimizationMetrics, OptimizationResult


def simulate_inventory_for_product(
    q: int,
    current_stock: int,
    supplier_cost: float,
    lead_time_days: int,
    daily_forecast: List[float],
) -> Tuple[float, float, float, float]:
    """Simulate day-by-day deterministic inventory dynamics for a single product over horizon H.
    
    Args:
        q: Order quantity.
        current_stock: Initial stock on hand.
        supplier_cost: Unit purchase cost.
        lead_time_days: Days before order arrives.
        daily_forecast: Forecasted daily demand series.
        
    Returns:
        Tuple of (total_shortage, ending_stock, purchase_cost, mean_daily_forecast).
    """
    stock = float(current_stock)
    total_shortage = 0.0
    h = len(daily_forecast)

    for d in range(1, h + 1):
        # Order arrival: if d == lead_time_days (or if lead_time == 0, arrives before day 1)
        if lead_time_days == 0 and d == 1:
            stock += float(q)
        elif d == lead_time_days:
            stock += float(q)

        demand = max(0.0, float(daily_forecast[d - 1]))
        shortage_d = max(0.0, demand - stock)
        total_shortage += shortage_d
        stock = max(0.0, stock - demand)

    purchase_cost = float(q) * supplier_cost
    mean_daily = float(np.mean(daily_forecast)) if h > 0 else 1.0

    return total_shortage, stock, purchase_cost, mean_daily


def compute_objective_components(
    chromosome: List[int],
    products_data: List[Dict[str, Any]],
    budget_gbp: float,
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Compute exact normalized multi-objective value and constituent components.
    
    Components:
    - NS: Normalized urgency-weighted shortage
    - NE: Normalized excess inventory
    - NC: Normalized purchase cost
    - NI: Normalized ending coverage imbalance
    
    Objective = w_shortage * NS + w_excess * NE + w_cost * NC + w_coverage * NI
    
    Args:
        chromosome: Vector of non-negative integer order quantities.
        products_data: List of product specification dictionaries.
        budget_gbp: Total purchasing budget.
        scales: Fixed normalization denominators.
        weights: Objective component weights.
        
    Returns:
        Tuple of (objective_value, components_dict).
    """
    total_weighted_shortage = 0.0
    total_excess = 0.0
    total_cost = 0.0
    coverage_deviations = []

    for q, p in zip(chromosome, products_data):
        shortage, end_stock, cost, mean_forecast = simulate_inventory_for_product(
            q=q,
            current_stock=p["CurrentStock"],
            supplier_cost=p["SupplierCostGBP"],
            lead_time_days=p["LeadTimeDays"],
            daily_forecast=p["DailyForecast"],
        )

        urgency_score = float(p.get("UrgencyScore", 50.0))
        urgency_weight = 1.0 + (urgency_score / 100.0)
        total_weighted_shortage += urgency_weight * shortage

        target_safety = 3.0 * mean_forecast
        excess = max(0.0, end_stock - target_safety)
        total_excess += excess

        total_cost += cost

        # Ending coverage deviation capped to 10
        ending_coverage = end_stock / max(mean_forecast, 0.1)
        cov_dev = min(abs(ending_coverage - 3.0), 10.0)
        coverage_deviations.append(cov_dev)

    mean_coverage_imbalance = float(np.mean(coverage_deviations)) if coverage_deviations else 0.0

    # Normalized metrics using fixed scales
    ns = total_weighted_shortage / scales["shortage_scale"]
    ne = total_excess / scales["excess_scale"]
    nc = total_cost / scales["cost_scale"]
    ni = mean_coverage_imbalance / scales["coverage_scale"]

    objective = (
        weights["shortage"] * ns
        + weights["excess"] * ne
        + weights["cost"] * nc
        + weights["coverage"] * ni
    )

    components = {
        "WeightedShortage": round(total_weighted_shortage, 4),
        "TotalExcess": round(total_excess, 4),
        "PurchaseCost": round(total_cost, 2),
        "CoverageImbalance": round(mean_coverage_imbalance, 4),
        "NS": round(ns, 6),
        "NE": round(ne, 6),
        "NC": round(nc, 6),
        "NI": round(ni, 6),
        "Objective": round(objective, 6),
    }

    return objective, components


def compute_normalization_scales(
    products_data: List[Dict[str, Any]], budget_gbp: float
) -> Dict[str, float]:
    """Calculate fixed normalization scales prior to evaluating any chromosomes."""
    # 1. Shortage scale: zero-order solution weighted shortage
    zero_chrom = [0] * len(products_data)
    zero_shortage = 0.0
    total_forecast_units = 0.0

    for p in products_data:
        shortage, _, _, _ = simulate_inventory_for_product(
            q=0,
            current_stock=p["CurrentStock"],
            supplier_cost=p["SupplierCostGBP"],
            lead_time_days=p["LeadTimeDays"],
            daily_forecast=p["DailyForecast"],
        )
        urgency_score = float(p.get("UrgencyScore", 50.0))
        urgency_weight = 1.0 + (urgency_score / 100.0)
        zero_shortage += urgency_weight * shortage
        total_forecast_units += float(sum(p["DailyForecast"]))

    return {
        "shortage_scale": max(zero_shortage, 1.0),
        "excess_scale": max(total_forecast_units, 1.0),
        "cost_scale": max(budget_gbp, 1.0),
        "coverage_scale": 3.0,
    }


def repair_chromosome(
    chromosome: List[int],
    products_data: List[Dict[str, Any]],
    budget_gbp: float,
) -> Tuple[List[int], List[str]]:
    """Repair chromosome to guarantee 100% adherence to all hard constraints.
    
    Hard constraints:
    - q_i >= 0
    - q_i == 0 or q_i >= MOQ_i
    - q_i % OrderMultiple_i == 0
    - CurrentStock_i + q_i <= StorageCapacity_i
    - sum(q_i * SupplierCost_i) <= BudgetGBP
    
    Args:
        chromosome: Raw integer order vector.
        products_data: List of product specifications.
        budget_gbp: Budget bound.
        
    Returns:
        Tuple of (repaired_chromosome, list_of_repair_actions).
    """
    repaired = list(chromosome)
    actions = []

    # 1. Storage capacity, MOQ, and Order Multiple per product
    for i, p in enumerate(products_data):
        curr_stock = p["CurrentStock"]
        storage_cap = p["StorageCapacity"]
        moq = p["MinimumOrderQuantity"]
        multiple = max(1, p["OrderMultiple"])

        # Storage capacity clamp
        max_allowed_by_storage = max(0, storage_cap - curr_stock)
        if repaired[i] > max_allowed_by_storage:
            actions.append(f"Product {p['StockCode']}: reduced {repaired[i]} -> {max_allowed_by_storage} due to storage capacity.")
            repaired[i] = max_allowed_by_storage

        # Round down to nearest valid order multiple
        if repaired[i] % multiple != 0:
            repaired[i] = (repaired[i] // multiple) * multiple

        # MOQ constraint
        if 0 < repaired[i] < moq:
            actions.append(f"Product {p['StockCode']}: order {repaired[i]} < MOQ {moq}; reset to 0.")
            repaired[i] = 0

        if repaired[i] < 0:
            repaired[i] = 0

    # 2. Global budget constraint
    total_cost = sum(q * p["SupplierCostGBP"] for q, p in zip(repaired, products_data))
    
    if total_cost > budget_gbp:
        # Over budget: reduce products in order multiple increments
        # Prioritize reducing products with lowest urgency or highest unit cost
        priority_indices = sorted(
            range(len(products_data)),
            key=lambda idx: (
                products_data[idx].get("UrgencyScore", 50.0) / max(products_data[idx]["SupplierCostGBP"], 0.1)
            ),
        )

        while total_cost > budget_gbp:
            reduced_any = False
            for idx in priority_indices:
                multiple = max(1, products_data[idx]["OrderMultiple"])
                moq = products_data[idx]["MinimumOrderQuantity"]

                if repaired[idx] >= multiple:
                    repaired[idx] -= multiple
                    if 0 < repaired[idx] < moq:
                        repaired[idx] = 0
                    reduced_any = True
                    total_cost = sum(q * p["SupplierCostGBP"] for q, p in zip(repaired, products_data))
                    if total_cost <= budget_gbp:
                        break

            if not reduced_any:
                # Force reset all to 0 as emergency fallback
                repaired = [0] * len(products_data)
                actions.append("Budget constraint emergency fallback: reset all orders to 0.")
                break

        actions.append(f"Budget repaired to £{total_cost:.2f} <= £{budget_gbp:.2f}")

    return repaired, actions


def solve_greedy_baseline(
    products_data: List[Dict[str, Any]],
    budget_gbp: float,
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> OptimizationResult:
    """Greedy baseline optimizer: iteratively adds best improvement per GBP.
    
    Starts with zero orders. At each step:
    - Finds smallest feasible increment for each product.
    - Evaluates delta objective / delta cost.
    - Selects the best positive marginal improvement.
    - Halts when no feasible positive improvement remains.
    
    Args:
        products_data: List of product specs.
        budget_gbp: Total budget.
        scales: Normalization denominators.
        weights: Objective weights.
        
    Returns:
        OptimizationResult dataclass.
    """
    start_time = time.time()
    n = len(products_data)
    current_chrom = [0] * n
    
    current_obj, current_components = compute_objective_components(
        current_chrom, products_data, budget_gbp, scales, weights
    )

    iteration = 0
    max_iterations = 2000

    while iteration < max_iterations:
        iteration += 1
        best_candidate = None
        best_efficiency = 0.0  # (obj_drop) / cost_increase

        for i, p in enumerate(products_data):
            multiple = max(1, p["OrderMultiple"])
            moq = p["MinimumOrderQuantity"]
            cost_unit = p["SupplierCostGBP"]
            storage_cap = p["StorageCapacity"]
            curr_stock = p["CurrentStock"]

            # Smallest feasible increment
            if current_chrom[i] == 0:
                inc = max(multiple, (moq + multiple - 1) // multiple * multiple) if moq > 0 else multiple
            else:
                inc = multiple

            candidate_q = current_chrom[i] + inc
            
            # Check individual storage constraint
            if curr_stock + candidate_q > storage_cap:
                continue

            # Check budget
            added_cost = inc * cost_unit
            current_total_cost = sum(q * pr["SupplierCostGBP"] for q, pr in zip(current_chrom, products_data))
            if current_total_cost + added_cost > budget_gbp:
                continue

            # Candidate chromosome
            candidate_chrom = list(current_chrom)
            candidate_chrom[i] = candidate_q

            cand_obj, _ = compute_objective_components(
                candidate_chrom, products_data, budget_gbp, scales, weights
            )

            delta_obj = current_obj - cand_obj  # positive means improvement (reduction in objective)
            if delta_obj > 1e-6 and added_cost > 0:
                efficiency = delta_obj / added_cost
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_candidate = (i, candidate_q, cand_obj)

        if best_candidate is not None:
            idx, new_q, new_obj = best_candidate
            current_chrom[idx] = new_q
            current_obj = new_obj
        else:
            # No feasible positive improvement remains
            break

    runtime = time.time() - start_time
    final_obj, final_components = compute_objective_components(
        current_chrom, products_data, budget_gbp, scales, weights
    )

    total_cost = sum(q * p["SupplierCostGBP"] for q, p in zip(current_chrom, products_data))
    budget_rem = max(0.0, budget_gbp - total_cost)
    util_pct = (total_cost / max(budget_gbp, 1.0)) * 100.0

    # Total expected shortage and excess
    total_short = 0.0
    total_exc = 0.0
    for q, p in zip(current_chrom, products_data):
        short, end_stk, _, m_fc = simulate_inventory_for_product(
            q=q,
            current_stock=p["CurrentStock"],
            supplier_cost=p["SupplierCostGBP"],
            lead_time_days=p["LeadTimeDays"],
            daily_forecast=p["DailyForecast"],
        )
        total_short += short
        total_exc += max(0.0, end_stk - (3.0 * m_fc))

    metrics = OptimizationMetrics(
        ObjectiveValue=round(final_obj, 6),
        TotalPurchaseCost=round(total_cost, 2),
        BudgetRemaining=round(budget_rem, 2),
        TotalExpectedShortage=round(total_short, 2),
        TotalExpectedExcess=round(total_exc, 2),
        CoverageImbalance=round(final_components["CoverageImbalance"], 4),
        BudgetUtilizationPercent=round(util_pct, 2),
        RuntimeSeconds=round(runtime, 4),
        Feasible=True,
    )

    prod_orders = {
        p["StockCode"]: q for q, p in zip(current_chrom, products_data)
    }

    return OptimizationResult(
        algorithm="Greedy Baseline",
        metrics=metrics,
        chromosome=current_chrom,
        product_orders=prod_orders,
        objective_components=final_components,
    )
