"""2-D Grid Search optimization on calibration split for (tau_yolo, tau_ad)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_binary_metrics
from .rules import is_sample_defect, predict_cascade, validate_sample_alignment


@dataclass
class GridPointResult:
    tau_yolo: float
    tau_ad: float
    tp: int
    tn: int
    fp: int
    fn: int
    defect_recall: float
    frr: float
    balanced_accuracy: float
    defect_precision: float
    f1: float
    mcc: float
    ecpb: float
    route_rate: float
    route_rate_good: float
    route_rate_defect: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepOutcome:
    optimal_primary: Optional[GridPointResult]        # Selected at FRR <= 0.05
    optimal_secondary: Optional[GridPointResult]  # Selected at FRR <= 0.01 (if available)
    grid_points: List[GridPointResult]
    ad_stats: Dict[str, float]
    config_used: Dict[str, Any]


# Contract name (§step5-contract sweep.py): grid_search_2d -> GridSearchResult.
GridSearchResult = SweepOutcome


def compute_ad_distribution_stats(ad_defect_scores: np.ndarray) -> Dict[str, float]:
    """Compute defect calibration anomaly score distribution statistics."""
    mean_val = float(np.mean(ad_defect_scores))
    std_val = float(np.std(ad_defect_scores, ddof=1)) if len(ad_defect_scores) > 1 else 0.0
    p10 = float(np.percentile(ad_defect_scores, 10))
    p50 = float(np.median(ad_defect_scores))
    p90 = float(np.percentile(ad_defect_scores, 90))
    p99_5 = float(np.percentile(ad_defect_scores, 99.5))
    min_val = float(np.min(ad_defect_scores))
    max_val = float(np.max(ad_defect_scores))

    return {
        "mean": mean_val,
        "std": std_val,
        "median": p50,
        "p10": p10,
        "p90": p90,
        "p99_5": p99_5,
        "min": min_val,
        "max": max_val,
    }


def grid_search_2d(
    yolo_rows: List[Dict[str, Any]],
    ad_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> SweepOutcome:
    """Execute 2-D Grid Search over (tau_yolo, tau_ad) on calibration split."""
    validate_sample_alignment(yolo_rows, ad_rows)
    ad_by_id = {r["sample_id"]: r for r in ad_rows}

    aligned_yolo = []
    aligned_ad = []
    labels = []
    y_scores = []
    ad_scores = []

    for yr in yolo_rows:
        sid = yr["sample_id"]
        ar = ad_by_id[sid]

        aligned_yolo.append(yr)
        aligned_ad.append(ar)
        is_def = is_sample_defect(sid, yr)
        labels.append(1 if is_def else 0)
        y_scores.append(float(yr.get("yolo_image_score", 0.0)))
        ad_scores.append(float(ar.get("anomaly_score", 0.0)))

    y_arr = np.asarray(y_scores, dtype=np.float64)
    ad_arr = np.asarray(ad_scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    n_samples = len(label_arr)
    n_defects = int(np.sum(label_arr == 1))
    n_goods = int(np.sum(label_arr == 0))

    if n_defects == 0 or n_goods == 0:
        raise ValueError(f"Calibration must have both good and defect samples (found {n_goods} good, {n_defects} defect)")

    # 1. Establish tau_yolo grid
    grid_cfg = config.get("grid_search", {})
    y_min = float(grid_cfg.get("yolo_min", 0.10))
    y_max = float(grid_cfg.get("yolo_max", 0.90))
    y_step = float(grid_cfg.get("yolo_step", 0.02))
    tau_yolo_grid = np.round(np.arange(y_min, y_max + 1e-6, y_step), 4)

    # 2. Establish tau_ad grid from calibration defect distribution (per PLAN §5 Bước 5: [mu, mu + 3*sigma])
    defect_ad_scores = ad_arr[label_arr == 1]
    stats = compute_ad_distribution_stats(defect_ad_scores)
    mu_ad = stats["mean"]
    sigma_ad = stats["std"]

    ad_points_count = int(grid_cfg.get("ad_num_points", 50))
    ad_lower = mu_ad
    ad_upper = mu_ad + 3.0 * sigma_ad
    tau_ad_grid = np.round(np.linspace(ad_lower, ad_upper, ad_points_count), 4)

    # Pre-calculate boolean comparison matrices for speed
    # y_passes: [len(y_grid), N]
    y_passes = y_arr[None, :] >= tau_yolo_grid[:, None]
    # ad_passes: [len(ad_grid), N]
    ad_passes = ad_arr[None, :] >= tau_ad_grid[:, None]

    cost_matrix = config.get("cost_matrix", {})
    results: List[GridPointResult] = []

    # Vectorized grid sweep
    for i, ty in enumerate(tau_yolo_grid):
        y_pass = y_passes[i]  # shape [N]
        # In cascade, if y_pass is False, we route to AD.
        # So routed_mask = ~y_pass
        route_count_total = int(np.sum(~y_pass))
        route_count_good = int(np.sum((~y_pass) & (label_arr == 0)))
        route_count_defect = int(np.sum((~y_pass) & (label_arr == 1)))

        rr_total = route_count_total / n_samples
        rr_good = route_count_good / n_goods
        rr_defect = route_count_defect / n_defects

        for j, ta in enumerate(tau_ad_grid):
            ad_pass = ad_passes[j]  # shape [N]
            # Final decision: DEFECT if (y_pass or (~y_pass and ad_pass)) == (y_pass | ad_pass)
            pred_defect = y_pass | ad_pass

            tp = int(np.sum(pred_defect & (label_arr == 1)))
            fp = int(np.sum(pred_defect & (label_arr == 0)))
            fn = n_defects - tp
            tn = n_goods - fp

            metrics = compute_binary_metrics(tp, tn, fp, fn, cost_matrix)

            res = GridPointResult(
                tau_yolo=round(float(ty), 4),
                tau_ad=round(float(ta), 4),
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                defect_recall=metrics["defect_recall"],
                frr=metrics["frr"],
                balanced_accuracy=metrics["balanced_accuracy"],
                defect_precision=metrics["defect_precision"],
                f1=metrics["f1"],
                mcc=metrics["mcc"],
                ecpb=metrics["ecpb"],
                route_rate=rr_total,
                route_rate_good=rr_good,
                route_rate_defect=rr_defect,
            )
            results.append(res)

    # Selection per PLAN §5 Bước 5:
    # Maximize Defect Recall subject to FRR <= constraint
    # Tie-breaker: lower tau_yolo (early exit / lower latency), then lower ECPB
    constraints = config.get("constraints", {})
    primary_frr = float(constraints.get("primary_frr_max", 0.05))
    secondary_frr = float(constraints.get("secondary_frr_max", 0.01))

    def pick_best(max_frr: float) -> Optional[GridPointResult]:
        eligible = [r for r in results if r.frr <= max_frr + 1e-9]
        if not eligible:
            return None
        # Sort key:
        # 1. defect_recall (descending)
        # 2. tau_yolo (ascending -> lower tau_yolo means more early-exit!)
        # 3. ecpb (ascending -> lower cost)
        # 4. frr (ascending)
        eligible.sort(key=lambda r: (-r.defect_recall, r.tau_yolo, r.ecpb, r.frr))
        return eligible[0]

    best_primary = pick_best(primary_frr)
    best_secondary = pick_best(secondary_frr)

    return SweepOutcome(
        optimal_primary=best_primary,
        optimal_secondary=best_secondary,
        grid_points=results,
        ad_stats=stats,
        config_used=config,
    )
