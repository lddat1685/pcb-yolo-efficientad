"""Four decision quadrants and >= boundaries from Step 5 contract."""
from copy import deepcopy
import pytest
from helpers.step5_spec import assert_routing_contract, rows


@pytest.mark.parametrize("defect", [False, True])
@pytest.mark.parametrize("ys,ads,expected,reason,route", [
    (0.2, 0.2, "GOOD", "all_pass", True),
    (0.2, 0.8, "DEFECT", "ad_hit", True),
    (0.8, 0.2, "DEFECT", "yolo_hit", False),
    (0.8, 0.8, "DEFECT", "yolo_hit", False),
    (0.5, 0.2, "DEFECT", "yolo_hit", False),
    (0.2, 0.5, "DEFECT", "ad_hit", True),
    (0.5, 0.5, "DEFECT", "yolo_hit", False),
])
def test_cascade_and_or_quadrants_and_threshold_equality(defect, ys, ads, expected, reason, route):
    from pcb_lab.fusion.rules import predict_cascade, predict_or
    yolo, ad = rows(defect=defect, yolo=ys, ad=ads)
    before = deepcopy((yolo, ad))
    cascade = predict_cascade(yolo, ad, 0.5, 0.5)
    either = predict_or(yolo, ad, 0.5, 0.5)
    assert cascade.decision == either.decision == expected
    assert cascade.mode == "CASCADE" and either.mode == "OR"
    assert cascade.reason == reason
    assert cascade.route_ad is route
    assert cascade.ad_score == (ads if route else None)
    assert either.route_ad is True and either.ad_score == ads
    for decision in (cascade, either):
        assert decision.is_defect is defect  # GT is preserved, not a prediction.
        assert decision.sample_id == yolo["sample_id"]
        assert decision.split == yolo["split"]
        assert_routing_contract(decision)
    assert (yolo, ad) == before


def test_false_positive_yolo_cannot_be_overridden_by_ad():
    from pcb_lab.fusion.rules import predict_cascade
    decision = predict_cascade(*rows(defect=False, yolo=0.9, ad=0.0), 0.5, 0.5)
    assert decision.decision == "DEFECT"
    assert decision.route_ad is False and decision.ad_score is None
