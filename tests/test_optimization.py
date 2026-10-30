"""Automated unit tests for Genetic Algorithm, Greedy Baseline, constraints, and optimization invariants.
"""
import numpy as np
import pytest

from config import OBJECTIVE_WEIGHTS
from core.greedy_baseline import (
    compute_normalization_scales,
    compute_objective_components,
    repair_chromosome,
    solve_greedy_baseline,
)
from core.optimization import GeneticAlgorithmOptimizer


@pytest.fixture
def sample_products_data():
    """Create deterministic fixture with 3 products of varying urgency and costs."""
    return [
        {
            "StockCode": "P1",
            "Description": "Product 1 High Urgency",
            "CurrentStock": 5,
            "SupplierCostGBP": 10.0,
            "LeadTimeDays": 3,
            "StorageCapacity": 100,
            "MinimumOrderQuantity": 5,
            "OrderMultiple": 5,
            "UrgencyScore": 90.0,
            "DailyForecast": [4.0] * 14,
        },
        {
            "StockCode": "P2",
            "Description": "Product 2 Moderate Urgency",
            "CurrentStock": 20,
            "SupplierCostGBP": 5.0,
            "LeadTimeDays": 5,
            "StorageCapacity": 150,
            "MinimumOrderQuantity": 10,
            "OrderMultiple": 2,
            "UrgencyScore": 50.0,
            "DailyForecast": [2.0] * 14,
        },
        {
            "StockCode": "P3",
            "Description": "Product 3 Low Urgency",
            "CurrentStock": 60,
            "SupplierCostGBP": 2.0,
            "LeadTimeDays": 4,
            "StorageCapacity": 80,
            "MinimumOrderQuantity": 0,
            "OrderMultiple": 1,
            "UrgencyScore": 20.0,
            "DailyForecast": [1.0] * 14,
        },
    ]


def test_hard_constraints_repair(sample_products_data):
    """Verify that repair_chromosome strictly enforces storage, MOQ, multiples, and budget."""
    budget = 100.0  # £100 budget
    # Violating chromosome: P1 ordered 50 (50*10 = £500 > £100), P2 ordered 7 (not multiple of 2, < MOQ 10)
    violating = [50, 7, 100]  # P3 storage cap is 80, curr is 60 -> max allowed is 20
    repaired, actions = repair_chromosome(violating, sample_products_data, budget)

    # 1. Non-negative
    assert all(q >= 0 for q in repaired)

    # 2. Storage capacity
    for q, p in zip(repaired, sample_products_data):
        assert p["CurrentStock"] + q <= p["StorageCapacity"]

    # 3. MOQ and OrderMultiple
    for q, p in zip(repaired, sample_products_data):
        if q > 0:
            assert q >= p["MinimumOrderQuantity"]
        assert q % p["OrderMultiple"] == 0

    # 4. Global Budget
    total_spend = sum(q * p["SupplierCostGBP"] for q, p in zip(repaired, sample_products_data))
    assert total_spend <= budget


def test_zero_budget_produces_zero_orders(sample_products_data):
    """Verify that a budget of zero strictly produces zero orders for all products."""
    scales = compute_normalization_scales(sample_products_data, 0.0)
    greedy_res = solve_greedy_baseline(sample_products_data, 0.0, scales, OBJECTIVE_WEIGHTS)
    assert all(q == 0 for q in greedy_res.chromosome)

    ga_opt = GeneticAlgorithmOptimizer(random_seed=42)
    ga_res = ga_opt.optimize(sample_products_data, 0.0, greedy_baseline_result=greedy_res)
    assert all(q == 0 for q in ga_res.chromosome)
    assert ga_res.metrics.TotalPurchaseCost == 0.0


def test_reproducibility_with_same_seed(sample_products_data):
    """Verify identical chromosomes are produced when using the exact same random seed."""
    budget = 500.0
    ga1 = GeneticAlgorithmOptimizer(random_seed=42)
    res1 = ga1.optimize(sample_products_data, budget)

    ga2 = GeneticAlgorithmOptimizer(random_seed=42)
    res2 = ga2.optimize(sample_products_data, budget)

    assert res1.chromosome == res2.chromosome
    assert res1.metrics.ObjectiveValue == res2.metrics.ObjectiveValue


def test_ga_invariant_objective_no_worse_than_greedy(sample_products_data):
    """Verify invariant: GA final objective must be <= greedy baseline objective."""
    budget = 400.0
    scales = compute_normalization_scales(sample_products_data, budget)
    greedy_res = solve_greedy_baseline(sample_products_data, budget, scales, OBJECTIVE_WEIGHTS)

    ga_opt = GeneticAlgorithmOptimizer(random_seed=42)
    ga_res = ga_opt.optimize(sample_products_data, budget, greedy_baseline_result=greedy_res)

    assert ga_res.metrics.ObjectiveValue <= greedy_res.metrics.ObjectiveValue + 1e-6
    assert ga_res.metrics.Feasible is True
    assert ga_res.metrics.TotalPurchaseCost <= budget


def test_objective_components_recomputation(sample_products_data):
    """Verify that objective components recompute accurately and sum to objective."""
    budget = 300.0
    scales = compute_normalization_scales(sample_products_data, budget)
    chrom = [10, 10, 10]
    rep_chrom, _ = repair_chromosome(chrom, sample_products_data, budget)

    obj_val, comp = compute_objective_components(
        rep_chrom, sample_products_data, budget, scales, OBJECTIVE_WEIGHTS
    )
    expected_obj = (
        OBJECTIVE_WEIGHTS["shortage"] * comp["NS"]
        + OBJECTIVE_WEIGHTS["excess"] * comp["NE"]
        + OBJECTIVE_WEIGHTS["cost"] * comp["NC"]
        + OBJECTIVE_WEIGHTS["coverage"] * comp["NI"]
    )
    assert obj_val == pytest.approx(expected_obj, abs=1e-5)
