"""Independent Step 5 interface/artifact checks; all impl imports are lazy."""
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import hashlib
import math
import re

import pytest

from helpers.step5_spec import (ROOT, METRICS, assert_routing_contract, config,
                                load_artifact, rows, walk)


def test_thresholds_json_schema_and_shared_operating_point():
    data = load_artifact("thresholds.json")
    assert type(data["schema_version"]) is int and data["schema_version"] == 1
    assert data["recipe_id"] == "fusion_recipe_v1"
    assert data["calibration_split"] == "calibration"
    assert data["target_frr"] == 0.05
    for name in ("h01_yolo11n_cascade", "h02_yolo11s_cascade", "h00_yolo11n_or"):
        model = data[name]
        assert 0.1 <= model["tau_yolo"] <= 0.9
        assert isinstance(model["tau_ad"], (float, int))
        assert math.isfinite(model["tau_ad"])
        # Phase B clarification: the artifact's {...} does not promise the
        # API-only defect_to_good field; the API test still requires it.
        assert (METRICS - {"defect_to_good"}) <= model["calibration_metrics"].keys()
        assert model["calibration_metrics"]["frr"] <= data["target_frr"]
    for threshold in ("tau_yolo", "tau_ad"):
        assert data["h00_yolo11n_or"][threshold] == data["h01_yolo11n_cascade"][threshold]


def test_routing_policy_json_locks_thresholds_provenance_and_test_access():
    policy = load_artifact("routing_policy.json")
    thresholds = load_artifact("thresholds.json")
    assert policy["test_access"] is False
    # Contract fixes the contents, but leaves policy nesting/key names open.
    nodes = list(walk(policy))
    strings = [node for node in nodes if isinstance(node, str)]
    hashes = {value for value in strings if re.fullmatch(r"[0-9a-fA-F]{64}", value)}
    # Phase B binding: model names need not occur literally in the policy.
    # Verify the promised B01/B02/B03 checksums against the six actual sources.
    for directory in ("yolo/B01_yolo11n", "yolo/B02_yolo11s",
                      "efficientad/B03_efficientad_s"):
        for split in ("calibration", "fusion"):
            path = ROOT / "artifacts" / directory / "seed42" / f"preds_{split}.jsonl"
            assert hashlib.sha256(path.read_bytes()).hexdigest() in hashes
    locked = thresholds["h01_yolo11n_cascade"]
    assert any(isinstance(node, dict) and
               node.get("tau_yolo") == locked["tau_yolo"] and
               node.get("tau_ad") == locked["tau_ad"] for node in nodes)
    assert any("cascade" in value.lower() for value in strings)
    assert any("early-exit" in value and "DEFECT" in value for value in strings)


def test_routing_decision_is_frozen_and_contains_contract_fields():
    from pcb_lab.fusion.rules import RoutingDecision, predict_cascade
    assert is_dataclass(RoutingDecision)
    assert {"sample_id", "split", "is_defect", "decision", "mode", "yolo_score",
            "ad_score", "route_ad", "reason", "yolo_boxes", "ad_boxes", "timing_ms"} <= {
                field.name for field in fields(RoutingDecision)}
    decision = predict_cascade(*rows(yolo=0.9), 0.5, 0.5)
    assert_routing_contract(decision)
    with pytest.raises(FrozenInstanceError):
        decision.ad_score = 0.8


@pytest.mark.parametrize("function", ["predict_cascade", "predict_or"])
@pytest.mark.parametrize("poison_side", ["yolo", "ad", "both"])
@pytest.mark.parametrize("yolo_score", [0.1, 0.9])
def test_test_split_is_denied_even_on_early_exit(function, poison_side, yolo_score):
    from pcb_lab.fusion import rules
    yolo, ad = rows(yolo=yolo_score)
    if poison_side in {"yolo", "both"}:
        yolo["split"] = "test"
    if poison_side in {"ad", "both"}:
        ad["split"] = "test"
    with pytest.raises(PermissionError):
        getattr(rules, function)(yolo, ad, 0.5, 0.5)


@pytest.mark.parametrize("function", ["predict_cascade", "predict_or"])
@pytest.mark.parametrize("yolo_score", [0.1, 0.9])
def test_sample_id_swap_is_rejected(function, yolo_score):
    from pcb_lab.fusion import rules
    yolo, ad = rows(yolo=yolo_score)
    ad["sample_id"] = "different-synthetic-id"
    with pytest.raises(ValueError):
        getattr(rules, function)(yolo, ad, 0.5, 0.5)


def test_grid_denies_injected_test_split():
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = rows(split="test")
    with pytest.raises(PermissionError):
        grid_search_2d([yolo], [ad], config(small=True))


def test_unrouted_score_sabotage_violates_contract_assertion():
    from pcb_lab.fusion.rules import predict_cascade
    decision = predict_cascade(*rows(yolo=0.9), 0.5, 0.5)
    assert_routing_contract(decision)
    # A frozen dataclass may validate replace itself; otherwise the independent
    # contract assertion must catch a forged numeric score on the skipped branch.
    with pytest.raises(AssertionError):
        poisoned = replace(decision, ad_score=0.73)
        assert_routing_contract(poisoned)
