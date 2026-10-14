"""Evaluation functions for fusion experiments on development_selection (fusion split)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .metrics import compute_binary_metrics, compute_confusion, compute_delta_metrics, compute_hybrid_metrics
from .rules import RoutingDecision, is_sample_defect, predict_cascade, predict_or, validate_sample_alignment


def evaluate_split(
    yolo_rows: List[Dict[str, Any]],
    ad_rows: List[Dict[str, Any]],
    tau_yolo: float,
    tau_ad: float,
    cost_params: Optional[Dict[str, float]] = None,
    *,
    allow_test: bool = False,
) -> Dict[str, Any]:
    """Evaluate both H01 (Cascade) and H00 (OR) on a split, plus baselines and deltas."""
    validate_sample_alignment(yolo_rows, ad_rows, allow_test=allow_test)
    ad_by_id = {r["sample_id"]: r for r in ad_rows}

    cascade_decisions: List[RoutingDecision] = []
    or_decisions: List[RoutingDecision] = []

    # Also track single-model decisions at the given thresholds
    yolo_only_tp = yolo_only_tn = yolo_only_fp = yolo_only_fn = 0
    ad_only_tp = ad_only_tn = ad_only_fp = ad_only_fn = 0

    for yr in yolo_rows:
        sid = yr["sample_id"]
        ar = ad_by_id[sid]

        is_def = is_sample_defect(sid, yr)
        y_score = float(yr.get("yolo_image_score", 0.0))
        ad_score = float(ar.get("anomaly_score", 0.0))

        # Single-model checks
        y_pred = y_score >= tau_yolo
        if is_def:
            if y_pred:
                yolo_only_tp += 1
            else:
                yolo_only_fn += 1
        else:
            if y_pred:
                yolo_only_fp += 1
            else:
                yolo_only_tn += 1

        ad_pred = ad_score >= tau_ad
        if is_def:
            if ad_pred:
                ad_only_tp += 1
            else:
                ad_only_fn += 1
        else:
            if ad_pred:
                ad_only_fp += 1
            else:
                ad_only_tn += 1

        # Decisions
        cas = predict_cascade(yr, ar, tau_yolo, tau_ad, allow_test=allow_test)
        cascade_decisions.append(cas)

        or_d = predict_or(yr, ar, tau_yolo, tau_ad, allow_test=allow_test)
        or_decisions.append(or_d)

    # 1. H01 (Cascade) Metrics
    cas_conf = compute_confusion(cascade_decisions)
    cas_metrics = compute_binary_metrics(
        cas_conf["tp"], cas_conf["tn"], cas_conf["fp"], cas_conf["fn"], cost_params
    )
    cas_hybrid = compute_hybrid_metrics(cascade_decisions, tau_yolo)
    h01_summary = dict(cas_metrics, **cas_hybrid)

    # 2. H00 (OR) Metrics
    or_conf = compute_confusion(or_decisions)
    or_metrics = compute_binary_metrics(
        or_conf["tp"], or_conf["tn"], or_conf["fp"], or_conf["fn"], cost_params
    )
    or_hybrid = compute_hybrid_metrics(or_decisions, tau_yolo)
    h00_summary = dict(or_metrics, **or_hybrid)

    # 3. Baseline summaries
    yolo_only_summary = compute_binary_metrics(
        yolo_only_tp, yolo_only_tn, yolo_only_fp, yolo_only_fn, cost_params
    )
    ad_only_summary = compute_binary_metrics(
        ad_only_tp, ad_only_tn, ad_only_fp, ad_only_fn, cost_params
    )

    # 4. Deltas (H01 vs H00)
    deltas = compute_delta_metrics(h01_summary, h00_summary)

    return {
        "tau_yolo": tau_yolo,
        "tau_ad": tau_ad,
        "h01_cascade": h01_summary,
        "h00_or": h00_summary,
        "yolo_only": yolo_only_summary,
        "ad_only": ad_only_summary,
        "deltas_H01_minus_H00": deltas,
        "num_samples": len(cascade_decisions),
    }


def evaluate_fusion(
    yolo_fusion_rows: List[Dict[str, Any]],
    ad_fusion_rows: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Contract entry point (§step5-contract evaluate.py).

    Evaluate H01 + H00 on the 306-image ``fusion`` (development_selection)
    split at the locked thresholds from ``policy`` (``tau_yolo``/``tau_ad``).
    ``policy`` may also carry a ``cost_matrix`` mapping for ECPB.
    """
    tau_yolo = float(policy["tau_yolo"])
    tau_ad = float(policy["tau_ad"])
    cost_params = policy.get("cost_matrix")
    return evaluate_split(yolo_fusion_rows, ad_fusion_rows, tau_yolo, tau_ad, cost_params)
