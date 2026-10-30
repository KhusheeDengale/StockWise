"""StockWise Core Typed Schemas and Dataclasses.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class SchemaValidationError(ValueError):
    """Raised when dataset or input schema fails validation."""
    pass


class InsufficientHistoryError(ValueError):
    """Raised when a product lacks sufficient history for evaluation or modeling."""
    pass


class OptimizationInfeasibleError(RuntimeError):
    """Raised when an optimization problem cannot find a feasible solution."""
    pass


@dataclass
class ForecastResult:
    """Standardized forecast output object for all downstream services."""
    stock_code: str
    product_description: str
    historical_start: str
    forecast_origin: str
    forecast_horizon: int
    forecast_dates: List[str]
    daily_predictions: List[float]
    total_predicted_demand: float
    selected_model: str
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    all_model_results: Dict[str, Any]
    selected_features: List[str]
    rejected_features: List[str]
    feature_selection_details: Dict[str, Any]
    model_parameters: Dict[str, Any]
    uncertainty_score: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class InventoryItem:
    """Simulated inventory record for a product."""
    StockCode: str
    Description: str
    CurrentStock: int
    SupplierCostGBP: float
    LeadTimeDays: int
    StorageCapacity: int
    MinimumOrderQuantity: int = 0
    OrderMultiple: int = 1


@dataclass
class FuzzyResult:
    """Mamdani fuzzy inference evaluation for reorder urgency."""
    coverage_ratio: float
    lead_time_pressure: float
    uncertainty: float
    membership_degrees: Dict[str, Dict[str, float]]
    fired_rules: List[Dict[str, Any]]
    firing_strengths: Dict[str, float]
    aggregated_output_curve: List[float]
    centroid: float
    category: str  # 'Low', 'Moderate', 'High', 'Critical'


@dataclass
class OptimizationMetrics:
    """Summary metrics of an optimization solution."""
    ObjectiveValue: float
    TotalPurchaseCost: float
    BudgetRemaining: float
    TotalExpectedShortage: float
    TotalExpectedExcess: float
    CoverageImbalance: float
    BudgetUtilizationPercent: float
    RuntimeSeconds: float
    Feasible: bool


@dataclass
class OptimizationResult:
    """Complete results from an inventory optimization run."""
    algorithm: str
    metrics: OptimizationMetrics
    chromosome: List[int]
    product_orders: Dict[str, int]
    objective_components: Dict[str, float]
    generation_history: List[Dict[str, Any]] = field(default_factory=list)
    explanation_events: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductRecommendation:
    """Final actionable purchasing recommendation for a single product."""
    StockCode: str
    Description: str
    ForecastDemand: float
    CurrentStock: int
    LeadTimeDays: int
    RiskScore: float
    RiskCategory: str
    RecommendedQuantity: int
    SupplierCostGBP: float
    TotalOrderCostGBP: float
    ExpectedShortageBefore: float
    ExpectedShortageAfter: float
    ExpectedEndingStock: float


@dataclass
class RecommendationReport:
    """Complete recommendation report including all baseline comparisons."""
    budget_gbp: float
    planning_horizon: int
    summary_metrics: Dict[str, Any]
    recommendations: List[ProductRecommendation]
    zero_order_metrics: OptimizationMetrics
    greedy_metrics: OptimizationMetrics
    ga_metrics: OptimizationMetrics
    fuzzy_risks: Dict[str, FuzzyResult]
    stability_analysis: Optional[Dict[str, Any]] = None
    explanations: Dict[str, Any] = field(default_factory=dict)
