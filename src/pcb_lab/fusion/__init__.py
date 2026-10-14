"""PCB Lab Fusion Package (Step 5): Calibration, Grid Search, Rules, and Evaluation."""
from .rules import RoutingDecision, predict_cascade, predict_or
from .metrics import compute_binary_metrics, compute_confusion, compute_hybrid_metrics, compute_delta_metrics
from .sweep import grid_search_2d, GridPointResult, SweepOutcome, GridSearchResult
from .evaluate import evaluate_split, evaluate_fusion

__all__ = [
    "RoutingDecision",
    "predict_cascade",
    "predict_or",
    "compute_binary_metrics",
    "compute_confusion",
    "compute_hybrid_metrics",
    "compute_delta_metrics",
    "grid_search_2d",
    "GridPointResult",
    "SweepOutcome",
    "GridSearchResult",
    "evaluate_split",
    "evaluate_fusion",
]
