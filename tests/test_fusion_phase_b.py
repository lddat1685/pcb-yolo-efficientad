"""Adversarial verification added after the independent Phase A commit."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import pytest

from helpers.step5_spec import (ROOT, config, confusion_oracle, field,
                                load_artifact, rows, sweep_rows)


@pytest.mark.parametrize("function", ["predict_cascade", "predict_or"])
def test_or_and_cascade_reason_stays_in_contract_enum(function):
    from pcb_lab.fusion import rules
    result = getattr(rules, function)(*rows(yolo=0.9, ad=0.9), 0.5, 0.5)
    assert result.reason in {"yolo_hit", "ad_hit", "all_pass"}


@pytest.mark.parametrize("function", ["predict_cascade", "predict_or"])
@pytest.mark.parametrize("side", ["yolo", "ad"])
def test_failed_prediction_cannot_silently_become_good(function, side):
    from pcb_lab.fusion import rules
    yolo, ad = rows(yolo=0.0, ad=0.0)
    poisoned = yolo if side == "yolo" else ad
    poisoned["error_reason"] = "synthetic inference failure"
    # PLAN sections 8/11: failures must not become GOOD. A binary-only API can
    # reject the input; a future explicit ERROR result is also acceptable.
    try:
        result = getattr(rules, function)(yolo, ad, 0.5, 0.5)
    except (ValueError, RuntimeError):
        return
    assert result.decision == "ERROR", "Inference failure was silently classified"


@pytest.mark.parametrize("entry", ["sweep", "evaluate"])
@pytest.mark.parametrize("side", ["yolo", "ad"])
def test_batch_entrypoints_deny_injected_test_split(entry, side):
    from pcb_lab.fusion.evaluate import evaluate_fusion
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = sweep_rows()
    (yolo if side == "yolo" else ad)[0]["split"] = "test"
    with pytest.raises(PermissionError):
        if entry == "sweep":
            grid_search_2d(yolo, ad, config(small=True))
        else:
            evaluate_fusion(yolo, ad, {"tau_yolo": 0.5, "tau_ad": 0.5})


def test_jsonl_loader_denies_synthetic_test_row(tmp_path):
    spec = importlib.util.spec_from_file_location("step5_runner_under_test", ROOT / "scripts/calibrate_fusion.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    path = tmp_path / "synthetic_poison.jsonl"
    path.write_text('{"sample_id":"synthetic-only","split":"test"}\n', encoding="utf-8")
    with pytest.raises(PermissionError):
        runner.load_jsonl(path)


@pytest.mark.parametrize("entry", ["sweep", "evaluate"])
@pytest.mark.parametrize("mutation", ["duplicate_yolo", "duplicate_ad", "missing_ad", "extra_ad"])
def test_batch_pairing_rejects_non_bijective_sample_ids(entry, mutation):
    from pcb_lab.fusion.evaluate import evaluate_fusion
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = sweep_rows()
    if mutation == "duplicate_yolo":
        yolo.append(deepcopy(yolo[0]))
    elif mutation == "duplicate_ad":
        ad.append(deepcopy(ad[0]))
    elif mutation == "missing_ad":
        ad[0]["sample_id"] = "swapped-id"
    else:
        ad.append(rows("extra-ad-only", defect=False)[1])
    with pytest.raises(ValueError):
        if entry == "sweep":
            grid_search_2d(yolo, ad, config(small=True))
        else:
            evaluate_fusion(yolo, ad, {"tau_yolo": 0.5, "tau_ad": 0.5})


def test_hybrid_contract_baseline_argument_is_used():
    from pcb_lab.fusion.metrics import compute_hybrid_metrics
    from pcb_lab.fusion.rules import predict_cascade
    pairs = [rows("d-yolo", yolo=0.9, ad=0.1),
             rows("d-rescue", yolo=0.1, ad=0.9),
             rows("d-miss", yolo=0.1, ad=0.1),
             rows("g-ad-fp", defect=False, yolo=0.1, ad=0.9)]
    cascade = [predict_cascade(y, a, 0.5, 0.5) for y, a in pairs]
    baseline = [replace(d, decision="DEFECT" if d.yolo_score >= 0.5 else "GOOD")
                for d in cascade]
    for result in (compute_hybrid_metrics(cascade, baseline),
                   compute_hybrid_metrics(cascade, yolo_only_decisions=baseline)):
        assert result["rescued_defects"] == 1
        assert result["rescue_rate"] == pytest.approx(0.5)
        assert result["added_false_rejects"] == 1


def test_h00_calibration_artifact_records_unconditional_ad_routing():
    metrics = load_artifact("thresholds.json")["h00_yolo11n_or"]["calibration_metrics"]
    for key in ("route_rate", "route_rate_good", "route_rate_defect"):
        assert metrics[key] == 1.0, key


def test_grid_points_replay_at_their_serialized_thresholds():
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = sweep_rows()
    yolo[-2]["yolo_image_score"] = 0.3
    result = grid_search_2d(yolo, ad, config())
    for point in result.grid_points:
        expected = confusion_oracle(yolo, ad, point.tau_yolo, point.tau_ad)
        actual = {key: field(point, key) for key in expected}
        assert actual == expected, (point.tau_yolo, point.tau_ad)


def test_infeasible_primary_constraint_does_not_return_a_locked_optimum():
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = sweep_rows()
    for row in yolo:
        row["yolo_image_score"] = 0.99
    try:
        result = grid_search_2d(yolo, ad, config(small=True))
    except ValueError:
        return
    assert result.optimal_primary is None, "A 100% FRR fallback must not be called optimal at FRR <= 5%"
