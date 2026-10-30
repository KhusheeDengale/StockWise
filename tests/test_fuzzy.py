"""Automated unit tests for the Mamdani Fuzzy Inference System and all 27 rules.
"""
import numpy as np
import pytest

from core.fuzzy_engine import (
    FUZZY_RULES,
    calculate_input_memberships,
    evaluate_fuzzy_urgency,
    map_urgency_category,
)


def test_all_27_rules_defined():
    """Verify that exactly 27 unique rules exist covering all 3x3x3 combinations."""
    assert len(FUZZY_RULES) == 27
    combos = set()
    for r in FUZZY_RULES:
        key = (r["coverage"], r["lead_pressure"], r["uncertainty"])
        assert key not in combos, f"Duplicate rule found: {key}"
        combos.add(key)
    assert len(combos) == 27


def test_membership_degrees_bounded_in_unit_interval():
    """Verify all membership degrees are in [0, 1] across various input domain points."""
    for cov in [0.0, 0.25, 0.75, 1.25, 2.0, 2.5, 3.0]:
        for lp in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for unc in [0.0, 0.2, 0.5, 0.8, 1.0]:
                m = calculate_input_memberships(cov, lp, unc)
                for var_name, terms in m.items():
                    for term_name, val in terms.items():
                        assert 0.0 <= val <= 1.0, f"Out of bounds: {var_name}.{term_name} = {val}"


def test_low_coverage_high_lead_pressure_critical_urgency():
    """Verify low coverage and high lead pressure produce High or Critical reorder urgency."""
    res = evaluate_fuzzy_urgency(
        current_stock=2,
        lead_time_days=10,
        planning_horizon_days=14,
        lead_time_forecast=[5.0] * 10,  # 50 demand vs 2 stock -> coverage ~ 0.04
        uncertainty_score=0.8,
    )
    assert res.centroid >= 70.0
    assert res.category in ["High", "Critical"]


def test_high_coverage_low_lead_pressure_low_urgency():
    """Verify high coverage and low lead pressure produce Low reorder urgency."""
    res = evaluate_fuzzy_urgency(
        current_stock=100,
        lead_time_days=2,
        planning_horizon_days=14,
        lead_time_forecast=[2.0, 2.0],  # 4 demand vs 100 stock -> coverage > 3.0
        uncertainty_score=0.1,
    )
    assert res.centroid < 35.0
    assert res.category == "Low"


def test_edge_values_handled_gracefully():
    """Verify edge values (zero demand, zero lead time, extreme stocks) are handled smoothly."""
    # Zero lead time
    res_zero_lt = evaluate_fuzzy_urgency(
        current_stock=10,
        lead_time_days=0,
        planning_horizon_days=14,
        lead_time_forecast=[],
        uncertainty_score=0.0,
    )
    assert 0.0 <= res_zero_lt.centroid <= 100.0

    # Zero stock and zero demand
    res_zeros = evaluate_fuzzy_urgency(
        current_stock=0,
        lead_time_days=5,
        planning_horizon_days=14,
        lead_time_forecast=[0.0] * 5,
        uncertainty_score=0.5,
    )
    assert 0.0 <= res_zeros.centroid <= 100.0


def test_centroid_numerical_integration():
    """Verify centroid calculation matches numerical formula."""
    res = evaluate_fuzzy_urgency(
        current_stock=10,
        lead_time_days=5,
        planning_horizon_days=14,
        lead_time_forecast=[3.0] * 5,
        uncertainty_score=0.3,
    )
    u_vals = np.arange(0.0, 100.5, 0.5)
    curve = np.array(res.aggregated_output_curve)
    expected_centroid = np.sum(u_vals * curve) / np.sum(curve)
    assert res.centroid == pytest.approx(expected_centroid, abs=0.05)
