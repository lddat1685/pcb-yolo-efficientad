"""Analytical controls for the independent Phase C AP50 implementation."""
import pytest
from helpers.step3_ap import ap50_101, linear_101_area


def test_ap_oracle_perfect_empty_duplicates_and_wrong_image():
    # Protect plan §7.3/Phase C §3: AP oracle must not count duplicates or cross-image matches as TP.
    gt = [{"sample_id": "a", "boxes": [{"class_id": 0, "xyxy": [0, 0, 10, 10]}]},
          {"sample_id": "b", "boxes": [{"class_id": 0, "xyxy": [0, 0, 10, 10]}]}]
    def detection(score):
        return {"class_id": 0, "confidence": score, "xyxy_original": [0, 0, 10, 10]}
    perfect = [{"sample_id": "a", "detections": [detection(.9)]}, {"sample_id": "b", "detections": [detection(.7)]}]
    assert ap50_101(perfect, gt, ["defect"])["overall_map50"] == 1.0
    assert ap50_101([], gt, ["defect"])["overall_map50"] == 0.0
    duplicated = [{"sample_id": "a", "detections": [detection(.9), detection(.8)]}, perfect[1]]
    result = ap50_101(duplicated, gt, ["defect"])
    assert result["per_class"]["defect"]["tp_at_conf_floor"] == 2
    assert result["overall_map50"] == pytest.approx((51 + 50 * 2 / 3) / 101)
    wrong_image = [{"sample_id": "not-a-or-b", "detections": [detection(.99)]}]
    assert ap50_101(wrong_image, gt, ["defect"])["overall_map50"] == 0.0


def test_linear_101_oracle_terminal_zero_and_empty_predictions():
    # Protect Phase C AP diagnosis: isolate interpolation convention without importing framework metrics.
    assert linear_101_area([], []) == 0.0
    assert linear_101_area([0.0], [0.0]) == 0.0
    assert linear_101_area([.5, 1.0], [1.0, 1.0]) == pytest.approx(.995)
    assert linear_101_area([.5], [1.0]) == pytest.approx(.495)
