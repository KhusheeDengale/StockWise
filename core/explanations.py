"""Explainability data structures and helpers for transparent model inspections.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LinearRegressionExplanation:
    """Detailed mathematical breakdown of a linear regression prediction."""
    step: int
    date: str
    feature_names: List[str]
    feature_values: Dict[str, float]
    coefficients: Dict[str, float]
    intercept: float
    individual_contributions: Dict[str, float]
    raw_prediction: float
    clipped_prediction: float
    formula_matches_raw: bool


@dataclass
class FeatureSelectionExplanation:
    """Audit report explaining candidate filtering, collinearity pruning, and validation MAE comparison."""
    candidate_features: List[str]
    selected_features: List[str]
    rejected_features: List[str]
    correlation_scores: Dict[str, float]
    redundancy_decisions: List[Dict[str, Any]]
    selection_reasons: Dict[str, str]
    validation_comparison: Dict[str, Any]


@dataclass
class ArimaExplanation:
    """Step-by-step audit of ARIMA parameter grid search and residual diagnostics."""
    candidate_orders: List[Dict[str, Any]]
    selected_order: Optional[List[int]]
    aic: Optional[float]
    bic: Optional[float]
    residuals: List[float]
    selection_criterion: str


@dataclass
class FuzzyLogicExplanation:
    """Comprehensive trace of Mamdani fuzzification, rule firing, and centroid integration."""
    crisp_inputs: Dict[str, float]
    membership_degrees: Dict[str, Dict[str, float]]
    activated_rules: List[Dict[str, Any]]
    rule_strengths: Dict[str, float]
    aggregated_output_curve: List[float]
    centroid: float
    category: str


@dataclass
class GeneticAlgorithmExplanation:
    """Evolution trace containing operator events, convergence history, and objective breakdown."""
    initial_population_summary: Dict[str, Any]
    sample_selection_event: Dict[str, Any]
    sample_crossover_event: Dict[str, Any]
    sample_mutation_event: Dict[str, Any]
    sample_repair_event: Dict[str, Any]
    generation_history: List[Dict[str, Any]]
    final_objective_components: Dict[str, float]
