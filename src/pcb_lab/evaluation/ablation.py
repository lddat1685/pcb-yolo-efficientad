"""Step 6 ablations: the 3 mandatory comparisons (contract scope)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .benchmark import BenchmarkResult


def _delta(a: Dict[str, Any], b: Dict[str, Any], metric: str) -> float | None:
    va, vb = a.get(metric), b.get(metric)
    if va is None or vb is None:
        return None
    return float(va) - float(vb)


class AblationResult(dict):
    """Structured ablation comparisons on the fusion (selection) split."""

    def __init__(
        self,
        selection_split: str = "fusion",
        ablations: Optional[Dict[str, Dict[str, Any]]] = None,
        notes: Optional[List[str]] = None,
        **kwargs,
    ):
        data = {
            "schema_version": 1,
            "selection_split": selection_split,
            "test_access": False,
            "ablations": ablations if ablations is not None else {},
            "notes": notes if notes is not None else [],
        }
        data.update(kwargs)
        super().__init__(data)

    @property
    def selection_split(self) -> str:
        return self["selection_split"]

    @selection_split.setter
    def selection_split(self, value: str):
        self["selection_split"] = value

    @property
    def ablations(self) -> Dict[str, Dict[str, Any]]:
        return self["ablations"]

    @ablations.setter
    def ablations(self, value: Dict[str, Dict[str, Any]]):
        self["ablations"] = value

    @property
    def notes(self) -> List[str]:
        return self["notes"]

    @notes.setter
    def notes(self, value: List[str]):
        self["notes"] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def to_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ab_id, ab in self["ablations"].items():
            row = {"ablation": ab_id}
            for k, v in ab.items():
                if not isinstance(v, dict):
                    row[k] = v
            rows.append(row)
        return rows


def compute_ablations(benchmark_result: BenchmarkResult) -> AblationResult:
    """Compute the 3 mandatory ablations from a BenchmarkResult (fusion split)."""
    m = benchmark_result.splits["fusion"]
    cal = benchmark_result.splits["calibration"]
    out = AblationResult()
    ab: Dict[str, Dict[str, Any]] = {}

    b01, b02 = m["B01"], m["B02"]
    ab["A1_detector_capacity_B01_vs_B02"] = {
        "comparison": "B02 (YOLO11s) minus B01 (YOLO11n), single-model, fusion split",
        "tau_yolo_B01": b01["tau_yolo"],
        "tau_yolo_B02": b02["tau_yolo"],
        "delta_recall": _delta(b02, b01, "defect_recall"),
        "delta_frr": _delta(b02, b01, "frr"),
        "delta_balanced_accuracy": _delta(b02, b01, "balanced_accuracy"),
        "delta_ecpb": _delta(b02, b01, "ecpb"),
        "delta_mcc": _delta(b02, b01, "mcc"),
        "B01": {k: b01[k] for k in (
            "balanced_accuracy", "defect_recall", "defect_precision",
            "frr", "f1", "mcc", "ecpb", "tp", "tn", "fp", "fn")},
        "B02": {k: b02[k] for k in (
            "balanced_accuracy", "defect_recall", "defect_precision",
            "frr", "f1", "mcc", "ecpb", "tp", "tn", "fp", "fn")},
    }

    h00, h01 = m["H00"], m["H01"]
    route = h01.get("route_rate_total", h01.get("route_rate"))
    saving_rate = (1.0 - route) if route is not None else None
    ab["A2_fusion_logic_H00_vs_H01"] = {
        "comparison": "H01 (cascade) minus H00 (OR), same operating point, fusion split",
        "tau_yolo": h01["tau_yolo"],
        "tau_ad": h01["tau_ad"],
        "delta_recall": _delta(h01, h00, "defect_recall"),
        "delta_frr": _delta(h01, h00, "frr"),
        "delta_ecpb": _delta(h01, h00, "ecpb"),
        "delta_balanced_accuracy": _delta(h01, h00, "balanced_accuracy"),
        "delta_latency_ms": None,
        "route_rate_H01": h01.get("route_rate"),
        "route_rate_H00": h00.get("route_rate"),
        "saving_rate_early_exit": saving_rate,
        "route_rate_good_H01": h01.get("route_rate_good"),
        "route_rate_defect_H01": h01.get("route_rate_defect"),
        "rescued_defects_H01": h01.get("rescued_defects"),
        "rescue_rate_H01": h01.get("rescue_rate"),
        "added_false_rejects_H01": h01.get("added_false_rejects"),
    }

    h01_sec = m.get("H01_secondary")
    if h01_sec is not None:
        ab["A3_operating_threshold_primary_vs_secondary"] = {
            "comparison": "H01 secondary (FRR<=1%) minus H01 primary (FRR<=5%), fusion split",
            "primary": {"tau_yolo": h01["tau_yolo"], "tau_ad": h01["tau_ad"]},
            "secondary": {"tau_yolo": h01_sec["tau_yolo"], "tau_ad": h01_sec["tau_ad"]},
            "delta_recall": _delta(h01_sec, h01, "defect_recall"),
            "delta_frr": _delta(h01_sec, h01, "frr"),
            "delta_ecpb": _delta(h01_sec, h01, "ecpb"),
            "delta_balanced_accuracy": _delta(h01_sec, h01, "balanced_accuracy"),
            "H01_primary_fusion": {k: h01[k] for k in (
                "balanced_accuracy", "defect_recall", "defect_precision",
                "frr", "f1", "mcc", "ecpb", "route_rate")},
            "H01_secondary_fusion": {k: h01_sec[k] for k in (
                "balanced_accuracy", "defect_recall", "defect_precision",
                "frr", "f1", "mcc", "ecpb", "route_rate")},
            "H01_primary_calibration": {k: cal["H01"][k] for k in (
                "balanced_accuracy", "defect_recall", "frr", "ecpb", "route_rate")},
            "H01_secondary_calibration": {k: cal["H01_secondary"][k] for k in (
                "balanced_accuracy", "defect_recall", "frr", "ecpb", "route_rate")},
        }
    else:
        ab["A3_operating_threshold_primary_vs_secondary"] = {
            "comparison": "secondary operating point unavailable (no grid file)",
            "delta_recall": None, "delta_frr": None, "delta_ecpb": None,
        }

    out.ablations = ab
    out.notes = [
        "Latency deltas are null: prediction files are static with timing_ms=null.",
        "Saving rate = 1 - route_rate(H01 cascade): share of images skipping EfficientAD.",
        "Ablation tile-256 vs resize-256 (PLAN R7/V2) not feasible: only tile-256 B03 preds exist.",
    ]
    return out
