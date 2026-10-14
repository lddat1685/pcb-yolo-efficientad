"""Step 6 implementer self-check: evaluation package unit tests (no dataset needed)."""
import json

import pytest


def test_load_predictions_refuses_test_partition(tmp_path):
    from pcb_lab.evaluation.benchmark import load_predictions
    p = tmp_path / "preds_fusion.jsonl"
    p.write_text(
        json.dumps({"sample_id": "a_defect", "split": "fusion"}) + "\n"
        + json.dumps({"sample_id": "b_good", "split": "test"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        load_predictions(p)


def test_derive_optimum_rule():
    from pcb_lab.evaluation.benchmark import derive_optimum
    grid = [
        {"tau_yolo": 0.10, "tau_ad": 1.0, "defect_recall": 1.0, "frr": 0.06, "ecpb": 0.06},
        {"tau_yolo": 0.14, "tau_ad": 1.0, "defect_recall": 1.0, "frr": 0.04, "ecpb": 0.05},
        {"tau_yolo": 0.12, "tau_ad": 1.0, "defect_recall": 1.0, "frr": 0.04, "ecpb": 0.04},
    ]
    opt = derive_optimum(grid, 0.05)
    assert opt["tau_yolo"] == 0.12  # tie-break: lower tau_yolo wins
    with pytest.raises(ValueError):
        derive_optimum(grid, 0.001)


def _fake_benchmark():
    from pcb_lab.evaluation.benchmark import BenchmarkResult
    def m(rec, frr, ecpb, bal=0.9, prec=0.8, route=0.3):
        return {"balanced_accuracy": bal, "defect_recall": rec, "defect_precision": prec,
                "frr": frr, "f1": 0.8, "mcc": 0.7, "ecpb": ecpb, "tp": 1, "tn": 1,
                "fp": 1, "fn": 0, "route_rate": route, "route_rate_good": route,
                "route_rate_defect": 0.0, "yolo_fn_count": 0, "rescued_defects": 0,
                "rescue_rate": None, "added_false_rejects": 1,
                "unclassified_positive_rate": 0.0, "mean_latency_ms": None,
                "tau_yolo": 0.14, "tau_ad": 1.45}
    b = BenchmarkResult(recipe_id="r", target_frr=0.05, secondary_frr_max=0.01,
                        cost_matrix={}, operating_points={"B03": {"tau_ad": 1.45}})
    b.splits = {
        "fusion": {
            "B01": dict(m(1.0, 0.38, 0.38), tau_yolo=0.14, tau_ad=None),
            "B02": dict(m(1.0, 0.15, 0.15), tau_yolo=0.38, tau_ad=None),
            "B03": dict(m(0.28, 0.0, 0.71), tau_yolo=None, tau_ad=1.45),
            "H00": dict(m(1.0, 0.38, 0.38), route_rate=1.0, route_rate_good=1.0,
                        route_rate_defect=1.0),
            "H01": dict(m(1.0, 0.38, 0.38)),
            "H02": dict(m(1.0, 0.15, 0.15), tau_yolo=0.38),
            "H01_secondary": dict(m(1.0, 0.01, 0.02), tau_yolo=0.48),
        },
        "calibration": {
            "H01": {"balanced_accuracy": 0.97, "defect_recall": 1.0,
                    "frr": 0.047, "ecpb": 0.047, "route_rate": 0.47},
            "H01_secondary": {"balanced_accuracy": 0.99, "defect_recall": 1.0,
                              "frr": 0.008, "ecpb": 0.008, "route_rate": 0.49},
        },
    }
    return b


def test_ablations_three_mandatory():
    from pcb_lab.evaluation import compute_ablations
    abl = compute_ablations(_fake_benchmark())
    assert set(abl.ablations) == {"A1_detector_capacity_B01_vs_B02",
                                  "A2_fusion_logic_H00_vs_H01",
                                  "A3_operating_threshold_primary_vs_secondary"}
    assert abl.to_dict()["test_access"] is False
    a2 = abl.ablations["A2_fusion_logic_H00_vs_H01"]
    assert a2["delta_recall"] == 0.0
    assert abs(a2["saving_rate_early_exit"] - 0.7) < 1e-9
    assert len(abl.to_rows()) == 3


def test_shortlist_minimum_members_and_schema():
    from pcb_lab.evaluation import compute_ablations, select_shortlist
    bench = _fake_benchmark()
    short = select_shortlist(bench, compute_ablations(bench))
    d = short.to_dict()
    assert d["schema_version"] == 1
    assert d["selection_split"] == "fusion"
    assert d["test_access"] is False
    ids = [s["model_id"] for s in d["shortlist"]]
    assert len(ids) >= 5
    for required in ("B01", "B03", "H00", "H01", "H02"):
        assert any(required in mid for mid in ids), required
    for s in d["shortlist"]:
        assert s["rationale"] and s["selection_metrics"]
    assert "total_gpu_hours_measured_yolo" in d["budget_summary"]
