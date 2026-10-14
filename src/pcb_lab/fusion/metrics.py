"""Evaluation and comparison metrics for fusion rules, including ECPB and hybrid metrics."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence
from .rules import RoutingDecision


def compute_confusion(decisions: Sequence[RoutingDecision]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for d in decisions:
        if d.is_defect:
            if d.decision == "DEFECT":
                tp += 1
            else:
                fn += 1
        else:
            if d.decision == "DEFECT":
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": len(decisions)}


def compute_binary_metrics(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    cost_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    total = tp + tn + fp + fn
    if total == 0:
        return {}

    n_defect = tp + fn
    n_good = tn + fp

    tpr = (tp / n_defect) if n_defect > 0 else 0.0
    tnr = (tn / n_good) if n_good > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0

    frr = (fp / n_good) if n_good > 0 else 0.0
    miss_rate = (fn / n_defect) if n_defect > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0

    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom > 0:
        mcc = float((tp * tn - fp * fn) / math.sqrt(denom))
    else:
        mcc = 0.0

    # ECPB (Expected Cost per Board)
    # Default: N_sim = 10,000, prevalence = 1% (100 defect, 9900 good), C_FN = 100, C_FP = 1
    cp = cost_params or {}
    c_fn = float(cp.get("c_fn", 100.0))
    c_fp = float(cp.get("c_fp", 1.0))
    prev = float(cp.get("defect_prevalence", 0.01))
    n_sim = float(cp.get("n_simulation", 10000.0))

    sim_defects = n_sim * prev
    sim_goods = n_sim * (1.0 - prev)
    fn_sim = sim_defects * miss_rate
    fp_sim = sim_goods * frr
    ecpb = (c_fn * fn_sim + c_fp * fp_sim) / n_sim

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "n_defect": n_defect,
        "n_good": n_good,
        "balanced_accuracy": balanced_acc,
        "defect_recall": tpr,
        "defect_precision": precision,
        "frr": frr,
        "defect_to_good": miss_rate,
        "f1": f1,
        "mcc": mcc,
        "ecpb": ecpb,
    }


def compute_hybrid_metrics(
    decisions: Sequence[RoutingDecision],
    tau_yolo: Optional[float] = None,
    yolo_only_decisions: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Metrics tracking cascade routing behaviour and hybrid gains.

    Contract signature is ``(decisions, yolo_only_decisions=None)``; the
    implementation additionally needs ``tau_yolo`` to derive YOLO-alone
    misses (``yolo_score < tau_yolo``) for rescue/false-reject accounting.
    Both calling conventions are accepted:

    - ``compute_hybrid_metrics(decisions, tau_yolo=0.5)`` (primary);
    - ``compute_hybrid_metrics(decisions, yolo_only_decisions=[...])``
      (contract form — rescue fields fall back to ``None``/``0`` when no
      threshold is available);
    - a list passed positionally as 2nd arg is treated as
      ``yolo_only_decisions`` for backward compatibility.
    """
    # Backward-compat: list passed positionally means yolo_only_decisions.
    if isinstance(tau_yolo, (list, tuple)):
        yolo_only_decisions = tau_yolo
        tau_yolo = None
    total = len(decisions)
    if total == 0:
        return {}

    goods = [d for d in decisions if not d.is_defect]
    defects = [d for d in decisions if d.is_defect]

    n_good = len(goods)
    n_defect = len(defects)

    routed = [d for d in decisions if d.route_ad]
    routed_good = [d for d in goods if d.route_ad]
    routed_defect = [d for d in defects if d.route_ad]

    route_rate = len(routed) / total if total > 0 else 0.0
    route_rate_good = len(routed_good) / n_good if n_good > 0 else 0.0
    route_rate_defect = len(routed_defect) / n_defect if n_defect > 0 else 0.0

    # Rescued defects & added false rejects:
    # 1. If yolo_only_decisions is provided, compare against its decisions.
    # 2. If tau_yolo is provided, compare yolo_score < tau_yolo.
    # 3. Otherwise, use route_ad (in cascade, route_ad==True means YOLO passed / missed).
    if yolo_only_decisions is not None:
        yolo_map = {getattr(d, "sample_id", d.get("sample_id") if isinstance(d, dict) else None): d
                    for d in yolo_only_decisions}
        def is_yolo_miss(d: RoutingDecision) -> bool:
            yd = yolo_map.get(d.sample_id)
            dec = getattr(yd, "decision", yd.get("decision") if isinstance(yd, dict) else None)
            return dec != "DEFECT"
        def is_yolo_good(d: RoutingDecision) -> bool:
            yd = yolo_map.get(d.sample_id)
            dec = getattr(yd, "decision", yd.get("decision") if isinstance(yd, dict) else None)
            return dec != "DEFECT"
    elif tau_yolo is not None:
        def is_yolo_miss(d: RoutingDecision) -> bool:
            return d.yolo_score < tau_yolo
        def is_yolo_good(d: RoutingDecision) -> bool:
            return d.yolo_score < tau_yolo
    else:
        def is_yolo_miss(d: RoutingDecision) -> bool:
            return d.route_ad
        def is_yolo_good(d: RoutingDecision) -> bool:
            return d.route_ad

    yolo_misses = [d for d in defects if is_yolo_miss(d)]
    yolo_miss_count = len(yolo_misses)
    rescued = [d for d in yolo_misses if d.decision == "DEFECT"]
    rescued_defects = len(rescued)
    rescue_rate = (rescued_defects / yolo_miss_count) if yolo_miss_count > 0 else None

    added_fr = [d for d in goods if is_yolo_good(d) and d.decision == "DEFECT"]
    added_false_rejects = len(added_fr)

    # Unclassified positive rate: AD-only positives / all predicted defects
    predicted_defects = [d for d in decisions if d.decision == "DEFECT"]
    ad_only_positives = [d for d in predicted_defects if d.reason == "ad_hit"]
    unclass_rate = (len(ad_only_positives) / len(predicted_defects)) if predicted_defects else 0.0

    # Timing
    valid_timings = [d.timing_ms for d in decisions if d.timing_ms is not None]
    mean_latency_ms = (sum(valid_timings) / len(valid_timings)) if valid_timings else None

    return {
        "route_rate": route_rate,
        "route_rate_good": route_rate_good,
        "route_rate_defect": route_rate_defect,
        "yolo_fn_count": yolo_miss_count,
        "rescued_defects": rescued_defects,
        "rescue_rate": rescue_rate,
        "added_false_rejects": added_false_rejects,
        "unclassified_positive_rate": unclass_rate,
        "mean_latency_ms": mean_latency_ms,
    }


def compute_delta_metrics(
    cascade_summary: Dict[str, Any],
    or_summary: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Delta metrics comparing H01 (Cascade) against H00 (OR)."""
    delta_recall = cascade_summary["defect_recall"] - or_summary["defect_recall"]
    delta_frr = cascade_summary["frr"] - or_summary["frr"]
    delta_ecpb = cascade_summary["ecpb"] - or_summary["ecpb"]

    cas_lat = cascade_summary.get("mean_latency_ms")
    or_lat = or_summary.get("mean_latency_ms")
    delta_latency = (cas_lat - or_lat) if (cas_lat is not None and or_lat is not None) else None

    return {
        "delta_recall_H01_minus_H00": delta_recall,
        "delta_FRR_H01_minus_H00": delta_frr,
        "delta_ECPB_H01_minus_H00": delta_ecpb,
        "delta_latency_ms_H01_minus_H00": delta_latency,
        # Short contract aliases (§step5-contract metrics.py)
        "delta_recall": delta_recall,
        "delta_frr": delta_frr,
        "delta_ecpb": delta_ecpb,
    }
