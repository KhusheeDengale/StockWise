"""Mamdani Fuzzy Inference System for Inventory Reorder Urgency Assessment.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import skfuzzy as fuzz

from core.schemas import FuzzyResult


# Trapezoidal and Triangular membership functions
def trapmf(x: Union[float, np.ndarray], abcd: Tuple[float, float, float, float]) -> Union[float, np.ndarray]:
    """Evaluate trapezoidal membership function."""
    a, b, c, d = abcd
    return fuzz.trapmf(np.atleast_1d(x), [a, b, c, d]) if isinstance(x, np.ndarray) else float(fuzz.trapmf(np.array([x]), [a, b, c, d])[0])


def trimf(x: Union[float, np.ndarray], abc: Tuple[float, float, float]) -> Union[float, np.ndarray]:
    """Evaluate triangular membership function."""
    a, b, c = abc
    return fuzz.trimf(np.atleast_1d(x), [a, b, c]) if isinstance(x, np.ndarray) else float(fuzz.trimf(np.array([x]), [a, b, c])[0])


# Exact Membership Parameters from Contract Section 14.2
MEMBERSHIP_PARAMS = {
    "coverage": {
        "Low": (0.0, 0.0, 0.5, 1.0),       # trapezoid
        "Medium": (0.5, 1.25, 2.0),         # triangle
        "High": (1.5, 2.0, 3.0, 3.0),       # trapezoid
    },
    "lead_pressure": {
        "Low": (0.0, 0.0, 0.2, 0.4),       # trapezoid
        "Medium": (0.2, 0.5, 0.8),          # triangle
        "High": (0.6, 0.8, 1.0, 1.0),       # trapezoid
    },
    "uncertainty": {
        "Low": (0.0, 0.0, 0.15, 0.35),     # trapezoid
        "Medium": (0.2, 0.5, 0.8),          # triangle
        "High": (0.65, 0.85, 1.0, 1.0),     # trapezoid
    },
    "urgency": {
        "Low": (0.0, 0.0, 15.0, 35.0),      # trapezoid
        "Moderate": (25.0, 45.0, 65.0),     # triangle
        "High": (55.0, 75.0, 90.0),         # triangle
        "Critical": (80.0, 90.0, 100.0, 100.0), # trapezoid
    },
}

# The 27 Mandatory Fuzzy Rules from Section 14.3
FUZZY_RULES = [
    # LOW UNCERTAINTY
    {"coverage": "Low", "lead_pressure": "Low", "uncertainty": "Low", "consequent": "High"},
    {"coverage": "Low", "lead_pressure": "Medium", "uncertainty": "Low", "consequent": "High"},
    {"coverage": "Low", "lead_pressure": "High", "uncertainty": "Low", "consequent": "Critical"},
    {"coverage": "Medium", "lead_pressure": "Low", "uncertainty": "Low", "consequent": "Low"},
    {"coverage": "Medium", "lead_pressure": "Medium", "uncertainty": "Low", "consequent": "Moderate"},
    {"coverage": "Medium", "lead_pressure": "High", "uncertainty": "Low", "consequent": "High"},
    {"coverage": "High", "lead_pressure": "Low", "uncertainty": "Low", "consequent": "Low"},
    {"coverage": "High", "lead_pressure": "Medium", "uncertainty": "Low", "consequent": "Low"},
    {"coverage": "High", "lead_pressure": "High", "uncertainty": "Low", "consequent": "Moderate"},

    # MEDIUM UNCERTAINTY
    {"coverage": "Low", "lead_pressure": "Low", "uncertainty": "Medium", "consequent": "High"},
    {"coverage": "Low", "lead_pressure": "Medium", "uncertainty": "Medium", "consequent": "Critical"},
    {"coverage": "Low", "lead_pressure": "High", "uncertainty": "Medium", "consequent": "Critical"},
    {"coverage": "Medium", "lead_pressure": "Low", "uncertainty": "Medium", "consequent": "Moderate"},
    {"coverage": "Medium", "lead_pressure": "Medium", "uncertainty": "Medium", "consequent": "Moderate"},
    {"coverage": "Medium", "lead_pressure": "High", "uncertainty": "Medium", "consequent": "High"},
    {"coverage": "High", "lead_pressure": "Low", "uncertainty": "Medium", "consequent": "Low"},
    {"coverage": "High", "lead_pressure": "Medium", "uncertainty": "Medium", "consequent": "Moderate"},
    {"coverage": "High", "lead_pressure": "High", "uncertainty": "Medium", "consequent": "Moderate"},

    # HIGH UNCERTAINTY
    {"coverage": "Low", "lead_pressure": "Low", "uncertainty": "High", "consequent": "Critical"},
    {"coverage": "Low", "lead_pressure": "Medium", "uncertainty": "High", "consequent": "Critical"},
    {"coverage": "Low", "lead_pressure": "High", "uncertainty": "High", "consequent": "Critical"},
    {"coverage": "Medium", "lead_pressure": "Low", "uncertainty": "High", "consequent": "Moderate"},
    {"coverage": "Medium", "lead_pressure": "Medium", "uncertainty": "High", "consequent": "High"},
    {"coverage": "Medium", "lead_pressure": "High", "uncertainty": "High", "consequent": "Critical"},
    {"coverage": "High", "lead_pressure": "Low", "uncertainty": "High", "consequent": "Moderate"},
    {"coverage": "High", "lead_pressure": "Medium", "uncertainty": "High", "consequent": "Moderate"},
    {"coverage": "High", "lead_pressure": "High", "uncertainty": "High", "consequent": "High"},
]


def calculate_input_memberships(
    coverage: float, lead_pressure: float, uncertainty: float
) -> Dict[str, Dict[str, float]]:
    """Compute membership degrees for all three input linguistic variables."""
    cov = float(np.clip(coverage, 0.0, 3.0))
    lp = float(np.clip(lead_pressure, 0.0, 1.0))
    unc = float(np.clip(uncertainty, 0.0, 1.0))

    return {
        "coverage": {
            "Low": float(trapmf(cov, MEMBERSHIP_PARAMS["coverage"]["Low"])),
            "Medium": float(trimf(cov, MEMBERSHIP_PARAMS["coverage"]["Medium"])),
            "High": float(trapmf(cov, MEMBERSHIP_PARAMS["coverage"]["High"])),
        },
        "lead_pressure": {
            "Low": float(trapmf(lp, MEMBERSHIP_PARAMS["lead_pressure"]["Low"])),
            "Medium": float(trimf(lp, MEMBERSHIP_PARAMS["lead_pressure"]["Medium"])),
            "High": float(trapmf(lp, MEMBERSHIP_PARAMS["lead_pressure"]["High"])),
        },
        "uncertainty": {
            "Low": float(trapmf(unc, MEMBERSHIP_PARAMS["uncertainty"]["Low"])),
            "Medium": float(trimf(unc, MEMBERSHIP_PARAMS["uncertainty"]["Medium"])),
            "High": float(trapmf(unc, MEMBERSHIP_PARAMS["uncertainty"]["High"])),
        },
    }


def map_urgency_category(score: float) -> str:
    """Map crisp urgency score [0, 100] to risk category."""
    if score < 35.0:
        return "Low"
    elif score < 60.0:
        return "Moderate"
    elif score < 80.0:
        return "High"
    else:
        return "Critical"


def evaluate_fuzzy_urgency(
    current_stock: float,
    lead_time_days: int,
    planning_horizon_days: int,
    lead_time_forecast: List[float],
    uncertainty_score: float,
) -> FuzzyResult:
    """Evaluate Mamdani Fuzzy Inference System for Reorder Urgency.
    
    Args:
        current_stock: Stock currently on hand.
        lead_time_days: Supplier lead time in days.
        planning_horizon_days: Planning horizon H.
        lead_time_forecast: Daily forecasted demand for days 1 to lead_time_days.
        uncertainty_score: Normalized forecast error proxy in [0, 1].
        
    Returns:
        FuzzyResult with crisp inputs, membership values, fired rules, output curve, and centroid.
    """
    # 1. Input Calculations (Section 14.1)
    if lead_time_days <= 0:
        demand_lt = 0.0
    else:
        # Sum demand during lead time
        demand_lt = float(sum(lead_time_forecast[:lead_time_days]))

    if demand_lt <= 1e-4:
        # Zero lead-time demand: safe coverage
        coverage_ratio = 3.0
    else:
        coverage_ratio = current_stock / max(demand_lt, 0.1)

    coverage_input = float(np.clip(coverage_ratio, 0.0, 3.0))
    lead_pressure_input = float(np.clip(lead_time_days / max(planning_horizon_days, 1), 0.0, 1.0))
    uncertainty_input = float(np.clip(uncertainty_score, 0.0, 1.0))

    # 2. Fuzzification
    memberships = calculate_input_memberships(
        coverage_input, lead_pressure_input, uncertainty_input
    )

    # 3. Output universe: 0 to 100 in 0.5 increments
    u_urgency = np.arange(0.0, 100.5, 0.5)
    
    # Precompute output membership curves on the universe
    mf_curves = {
        "Low": trapmf(u_urgency, MEMBERSHIP_PARAMS["urgency"]["Low"]),
        "Moderate": trimf(u_urgency, MEMBERSHIP_PARAMS["urgency"]["Moderate"]),
        "High": trimf(u_urgency, MEMBERSHIP_PARAMS["urgency"]["High"]),
        "Critical": trapmf(u_urgency, MEMBERSHIP_PARAMS["urgency"]["Critical"]),
    }

    # 4. Rule Evaluation (AND = min, Implication = min, Aggregation = max)
    aggregated_curve = np.zeros_like(u_urgency)
    fired_rules = []
    firing_strengths: Dict[str, float] = {}

    for i, rule in enumerate(FUZZY_RULES):
        mu_cov = memberships["coverage"][rule["coverage"]]
        mu_lp = memberships["lead_pressure"][rule["lead_pressure"]]
        mu_unc = memberships["uncertainty"][rule["uncertainty"]]
        
        strength = min(mu_cov, mu_lp, mu_unc)
        consequent = rule["consequent"]
        firing_strengths[f"Rule_{i+1}"] = round(float(strength), 4)

        if strength > 1e-4:
            # Implication (minimum clipping)
            implied_curve = np.minimum(strength, mf_curves[consequent])
            # Aggregation (maximum)
            aggregated_curve = np.maximum(aggregated_curve, implied_curve)
            
            fired_rules.append({
                "rule_id": i + 1,
                "antecedent": f"Coverage={rule['coverage']} AND LeadPressure={rule['lead_pressure']} AND Uncertainty={rule['uncertainty']}",
                "consequent": consequent,
                "strength": round(float(strength), 4),
            })

    # 5. Defuzzification (Centroid)
    denom = np.sum(aggregated_curve)
    if denom > 1e-6:
        centroid = float(np.sum(u_urgency * aggregated_curve) / denom)
    else:
        # Fallback if no rule activated due to numerical edge case
        centroid = 50.0

    centroid = round(float(np.clip(centroid, 0.0, 100.0)), 2)
    category = map_urgency_category(centroid)

    return FuzzyResult(
        coverage_ratio=round(coverage_ratio, 4),
        lead_time_pressure=round(lead_pressure_input, 4),
        uncertainty=round(uncertainty_input, 4),
        membership_degrees=memberships,
        fired_rules=fired_rules,
        firing_strengths=firing_strengths,
        aggregated_output_curve=[round(float(v), 4) for v in aggregated_curve],
        centroid=centroid,
        category=category,
    )
