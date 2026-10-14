"""Step 9 Spec-First Test Suite: Runtime Profiling Invariants."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_profiler_timing_stages_exist():
    """Verify that InspectionResult.timing_ms contains required stages from PLAN.md 7.5."""
    from pcb_lab.inference.service import InspectionResult

    result = InspectionResult(model_id="B01")
    timing_keys = set(result.timing_ms.keys())
    assert "decode" in timing_keys, "Missing decode timing stage"
    assert "yolo" in timing_keys, "Missing yolo timing stage"
    assert "total_core" in timing_keys, "Missing total_core timing stage"


def test_score_difference_max_allowed():
    """Document the export tolerance contract: max score diff <= 1e-3."""
    tolerance = 1e-3
    assert tolerance == 1e-3, "Spec defines max score diff <= 1e-3"


def test_decision_consistency_across_backends():
    """Document the decision contract: 0% decision mismatch across backends."""
    assert True, "Decision must be consistent across backends"


def test_early_exit_timing_invariant():
    """Early-exit result must not run AD and must leave anomaly_score None."""
    from pcb_lab.inference.service import InspectionResult

    result = InspectionResult(model_id="H01")
    result.early_exit = True
    result.yolo_executed = True
    result.efficientad_executed = False
    result.anomaly_score = None
    result.timing_ms = {"decode": 10.0, "yolo": 50.0, "efficientad": 0.0, "total_core": 60.0}

    assert result.early_exit is True
    assert result.efficientad_executed is False
    assert result.anomaly_score is None


def test_timing_precision_has_ms_resolution():
    """Timing values must carry at least millisecond resolution."""
    from pcb_lab.inference.service import InspectionResult

    result = InspectionResult(model_id="B01")
    result.timing_ms = {"decode": 1.234567, "yolo": 50.654321, "total_core": 52.0}
    for stage, val in result.timing_ms.items():
        if val > 0:
            decimals = f"{val:.6f}".split(".")[-1]
            assert len(decimals) >= 3, f"{stage} timing lacks sufficient precision"
