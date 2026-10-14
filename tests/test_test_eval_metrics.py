"""Step 7 Spec-first Metrics Tests: Verification of Metric Formulas and Invariants."""
from __future__ import annotations

import math
import pytest

SIX_CLASSES = [
    "open_circuit",
    "short",
    "mouse_bite",
    "spur",
    "spurious_copper",
    "pin_hole",
]


def test_six_defect_classes_canonical():
    """Verify standard 6 defect classes according to PLAN §1.3."""
    assert len(SIX_CLASSES) == 6
    assert set(SIX_CLASSES) == {
        "open_circuit",
        "short",
        "mouse_bite",
        "spur",
        "spurious_copper",
        "pin_hole",
    }


def test_ecpb_formula_contract():
    """Verify ECPB calculation matches PLAN §7.3: N=10000, 1% Defect, C_FN=100, C_FP=1."""
    # Defect: 100 boards, Good: 9900 boards
    # Case 1: Perfect classifier (Recall=1.0, FRR=0.0) -> ECPB = 0
    tpr = 1.0
    frr = 0.0
    fn_sim = 100 * (1.0 - tpr)
    fp_sim = 9900 * frr
    ecpb = (100 * fn_sim + 1 * fp_sim) / 10000.0
    assert ecpb == 0.0

    # Case 2: Recall=0.90 (FN=10%), FRR=0.05 (FP=5%)
    tpr = 0.90
    frr = 0.05
    fn_sim = 100 * (1.0 - 0.90)  # 10 FN boards
    fp_sim = 9900 * 0.05         # 495 FP boards
    expected_ecpb = (100 * 10 + 1 * 495) / 10000.0  # (1000 + 495) / 10000 = 0.1495
    assert abs(expected_ecpb - 0.1495) < 1e-9


def test_balanced_accuracy_formula():
    """Verify Balanced Accuracy = (TPR + TNR) / 2 = (TPR + (1 - FRR)) / 2."""
    tpr = 0.95
    frr = 0.05
    tnr = 1.0 - frr
    bal_acc = (tpr + tnr) / 2.0
    assert abs(bal_acc - 0.95) < 1e-9


def test_worst_group_metrics_logic():
    """Verify worst-group logic: worst recall is MIN across groups, worst FRR is MAX across groups."""
    groups = {
        "group44000": {"recall": 0.98, "frr": 0.03},
        "group92000": {"recall": 0.92, "frr": 0.08},
    }
    worst_recall = min(g["recall"] for g in groups.values())
    worst_frr = max(g["frr"] for g in groups.values())

    assert worst_recall == 0.92
    assert worst_frr == 0.08


def test_delta_metrics_logic():
    """Verify delta metrics: H01 - H00."""
    h00 = {"recall": 1.0, "frr": 0.05, "latency_mean_ms": 50.0}
    h01 = {"recall": 1.0, "frr": 0.05, "latency_mean_ms": 22.0}

    delta_recall = h01["recall"] - h00["recall"]
    delta_frr = h01["frr"] - h00["frr"]
    delta_latency = h01["latency_mean_ms"] - h00["latency_mean_ms"]

    assert delta_recall == 0.0
    assert delta_frr == 0.0
    assert delta_latency == -28.0  # Negative means early-exit speedup!
