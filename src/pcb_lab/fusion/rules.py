"""Routing and fusion decision rules: H01 (Cascade) and H00 (OR)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RoutingDecision:
    sample_id: str
    split: str
    is_defect: bool
    decision: str  # "GOOD" | "DEFECT"
    mode: str      # "CASCADE" | "OR"
    yolo_score: float
    ad_score: Optional[float]
    route_ad: bool
    reason: str    # "yolo_hit" | "ad_hit" | "both_hit" | "all_pass"
    yolo_boxes: List[Any] = field(default_factory=list)
    ad_boxes: List[Any] = field(default_factory=list)
    timing_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "is_defect": self.is_defect,
            "decision": self.decision,
            "mode": self.mode,
            "yolo_score": self.yolo_score,
            "ad_score": self.ad_score,
            "route_ad": self.route_ad,
            "reason": self.reason,
            "yolo_boxes": self.yolo_boxes,
            "ad_boxes": self.ad_boxes,
            "timing_ms": self.timing_ms,
        }


def _filter_yolo_boxes(boxes: List[Any], tau_yolo: float) -> List[Any]:
    """Filter YOLO boxes by confidence >= tau.

    Supports dict-format detections ({"confidence": c, ...}) from Step 3
    extractor and legacy list/tuple format ([x1,y1,x2,y2,conf,...]).
    """
    kept = []
    for b in boxes or []:
        try:
            if isinstance(b, dict):
                conf = b.get("confidence", 0.0)
            elif isinstance(b, (list, tuple)) and len(b) > 4:
                conf = b[4]
            else:
                continue
            if float(conf) >= tau_yolo:
                kept.append(b)
        except (TypeError, ValueError, KeyError):
            continue
    return kept


def is_sample_defect(sample_id: str, row: Optional[Dict[str, Any]] = None) -> bool:
    if row and "is_defect" in row:
        return bool(row["is_defect"])
    return sample_id.endswith("_defect")


def validate_sample_alignment(
    yolo_rows: List[Dict[str, Any]],
    ad_rows: List[Dict[str, Any]],
    *,
    allow_test: bool = False,
) -> None:
    """Validate zero test partition access (unless authorized) and bijective sample ID matching."""
    if not allow_test:
        for yr in yolo_rows:
            if yr.get("split") == "test":
                raise PermissionError("Fusion must not access test partition without authorization")
        for ar in ad_rows:
            if ar.get("split") == "test":
                raise PermissionError("Fusion must not access test partition without authorization")

    yolo_ids = [yr["sample_id"] for yr in yolo_rows]
    ad_ids = [ar["sample_id"] for ar in ad_rows]

    if len(yolo_ids) != len(set(yolo_ids)):
        raise ValueError("Duplicate sample_id found in YOLO predictions")
    if len(ad_ids) != len(set(ad_ids)):
        raise ValueError("Duplicate sample_id found in EfficientAD predictions")
    if set(yolo_ids) != set(ad_ids):
        raise ValueError("Sample IDs between YOLO and EfficientAD are not bijective")


def predict_cascade(
    yolo_row: Dict[str, Any],
    ad_row: Dict[str, Any],
    tau_yolo: float,
    tau_ad: float,
    *,
    allow_test: bool = False,
) -> RoutingDecision:
    """H01: Sequential cascade.

    If YOLO detects a defect (score >= tau_yolo), return DEFECT early.
    EfficientAD is not run; ad_score is None, route_ad is False.
    Otherwise, route to EfficientAD. If ad_score >= tau_ad, return DEFECT;
    otherwise return GOOD.
    """
    sample_id = yolo_row["sample_id"]
    if sample_id != ad_row["sample_id"]:
        raise ValueError(f"Sample ID mismatch: {sample_id} vs {ad_row['sample_id']}")

    if not allow_test and (yolo_row.get("split") == "test" or ad_row.get("split") == "test"):
        raise PermissionError("Fusion must not process test partition without authorization")

    split = yolo_row.get("split") or ad_row.get("split", "unknown")

    if yolo_row.get("error_reason"):
        raise ValueError(f"YOLO inference error: {yolo_row['error_reason']}")

    is_defect = is_sample_defect(sample_id, yolo_row)
    yolo_score = float(yolo_row.get("yolo_image_score", 0.0))
    raw_ad_score = float(ad_row.get("anomaly_score", 0.0))

    y_time = yolo_row.get("timing_ms")
    ad_time = ad_row.get("timing_ms")

    raw_yolo_boxes = yolo_row.get("detections", [])
    raw_ad_boxes = ad_row.get("boxes", [])

    if yolo_score >= tau_yolo:
        # Early-exit: YOLO positive
        filtered_boxes = _filter_yolo_boxes(raw_yolo_boxes, tau_yolo)
        return RoutingDecision(
            sample_id=sample_id,
            split=split,
            is_defect=is_defect,
            decision="DEFECT",
            mode="CASCADE",
            yolo_score=yolo_score,
            ad_score=None,
            route_ad=False,
            reason="yolo_hit",
            yolo_boxes=filtered_boxes if filtered_boxes else raw_yolo_boxes,
            ad_boxes=[],
            timing_ms=float(y_time) if y_time is not None else None,
        )

    # Routed to EfficientAD
    if ad_row.get("error_reason"):
        raise ValueError(f"EfficientAD inference error: {ad_row['error_reason']}")

    route_time = None
    if y_time is not None and ad_time is not None:
        route_time = float(y_time) + float(ad_time)

    if raw_ad_score >= tau_ad:
        return RoutingDecision(
            sample_id=sample_id,
            split=split,
            is_defect=is_defect,
            decision="DEFECT",
            mode="CASCADE",
            yolo_score=yolo_score,
            ad_score=raw_ad_score,
            route_ad=True,
            reason="ad_hit",
            yolo_boxes=[],
            ad_boxes=raw_ad_boxes,
            timing_ms=route_time,
        )

    return RoutingDecision(
        sample_id=sample_id,
        split=split,
        is_defect=is_defect,
        decision="GOOD",
        mode="CASCADE",
        yolo_score=yolo_score,
        ad_score=raw_ad_score,
        route_ad=True,
        reason="all_pass",
        yolo_boxes=[],
        ad_boxes=[],
        timing_ms=route_time,
    )


def predict_or(
    yolo_row: Dict[str, Any],
    ad_row: Dict[str, Any],
    tau_yolo: float,
    tau_ad: float,
    *,
    allow_test: bool = False,
) -> RoutingDecision:
    """H00: Unconditional OR rule.

    Both YOLO and EfficientAD are always evaluated (route_ad is True).
    If either model detects a defect, return DEFECT; otherwise GOOD.
    """
    sample_id = yolo_row["sample_id"]
    if sample_id != ad_row["sample_id"]:
        raise ValueError(f"Sample ID mismatch: {sample_id} vs {ad_row['sample_id']}")

    if not allow_test and (yolo_row.get("split") == "test" or ad_row.get("split") == "test"):
        raise PermissionError("Fusion must not process test partition without authorization")

    split = yolo_row.get("split") or ad_row.get("split", "unknown")

    if yolo_row.get("error_reason"):
        raise ValueError(f"YOLO inference error: {yolo_row['error_reason']}")
    if ad_row.get("error_reason"):
        raise ValueError(f"EfficientAD inference error: {ad_row['error_reason']}")

    is_defect = is_sample_defect(sample_id, yolo_row)
    yolo_score = float(yolo_row.get("yolo_image_score", 0.0))
    ad_score = float(ad_row.get("anomaly_score", 0.0))

    y_time = yolo_row.get("timing_ms")
    ad_time = ad_row.get("timing_ms")
    total_time = (float(y_time) + float(ad_time)) if (y_time is not None and ad_time is not None) else None

    raw_yolo_boxes = yolo_row.get("detections", [])
    raw_ad_boxes = ad_row.get("boxes", [])

    yolo_hit = yolo_score >= tau_yolo
    ad_hit = ad_score >= tau_ad

    filtered_yolo = _filter_yolo_boxes(raw_yolo_boxes, tau_yolo)

    if yolo_hit:
        return RoutingDecision(
            sample_id=sample_id,
            split=split,
            is_defect=is_defect,
            decision="DEFECT",
            mode="OR",
            yolo_score=yolo_score,
            ad_score=ad_score,
            route_ad=True,
            reason="yolo_hit",
            yolo_boxes=filtered_yolo if filtered_yolo else raw_yolo_boxes,
            ad_boxes=raw_ad_boxes if ad_hit else [],
            timing_ms=total_time,
        )
    elif ad_hit:
        return RoutingDecision(
            sample_id=sample_id,
            split=split,
            is_defect=is_defect,
            decision="DEFECT",
            mode="OR",
            yolo_score=yolo_score,
            ad_score=ad_score,
            route_ad=True,
            reason="ad_hit",
            yolo_boxes=[],
            ad_boxes=raw_ad_boxes,
            timing_ms=total_time,
        )

    return RoutingDecision(
        sample_id=sample_id,
        split=split,
        is_defect=is_defect,
        decision="GOOD",
        mode="OR",
        yolo_score=yolo_score,
        ad_score=ad_score,
        route_ad=True,
        reason="all_pass",
        yolo_boxes=[],
        ad_boxes=[],
        timing_ms=total_time,
    )
