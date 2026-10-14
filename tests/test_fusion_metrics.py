"""Hand-computed metric examples, including industrial ECPB weighting."""
import math
import pytest
from helpers.step5_spec import METRICS


@pytest.mark.parametrize("tp,tn,fp,fn,expected", [
    (10, 10, 0, 0, 0.0), (0, 0, 10, 10, 1.99),
    (0, 10, 0, 10, 1.0), (10, 0, 10, 0, 0.99),
    (3, 8, 2, 1, 0.448), (30, 800, 200, 10, 0.448),
])
def test_ecpb_boundaries_and_class_prevalence_invariance(tp, tn, fp, fn, expected):
    from pcb_lab.fusion.metrics import compute_binary_metrics
    result = compute_binary_metrics(tp, tn, fp, fn)
    assert METRICS <= result.keys()
    assert result["ecpb"] == pytest.approx(expected)


def test_binary_metrics_hand_computed():
    from pcb_lab.fusion.metrics import compute_binary_metrics
    result = compute_binary_metrics(3, 8, 2, 1)
    expected = {"balanced_accuracy": 0.775, "defect_recall": 0.75,
                "defect_precision": 0.6, "frr": 0.2, "defect_to_good": 0.25,
                "f1": 2 / 3, "mcc": 22 / math.sqrt(1800), "ecpb": 0.448}
    for name, value in expected.items():
        assert result[name] == pytest.approx(value), name


def test_custom_cost_matrix_uses_population_rates():
    from pcb_lab.fusion.metrics import compute_binary_metrics
    params = {"c_fn": 7, "c_fp": 2, "defect_prevalence": 0.2, "n_simulation": 5000}
    result = compute_binary_metrics(3, 8, 2, 1, params)
    assert result["ecpb"] == pytest.approx(0.25 * 0.2 * 7 + 0.2 * 0.8 * 2)


def decision(sample_id, *, defect, route, positive, reason):
    from pcb_lab.fusion.rules import RoutingDecision
    return RoutingDecision(sample_id=sample_id, split="fusion", is_defect=defect,
                           decision="DEFECT" if positive else "GOOD", mode="CASCADE",
                           yolo_score=0.1 if route else 0.9,
                           ad_score=(0.9 if positive else 0.1) if route else None,
                           route_ad=route, reason=reason, yolo_boxes=[], ad_boxes=[],
                           timing_ms=None)


def test_confusion_and_hybrid_metrics_hand_counted():
    from pcb_lab.fusion.metrics import compute_confusion, compute_hybrid_metrics
    decisions = [
        decision("d-yolo", defect=True, route=False, positive=True, reason="yolo_hit"),
        decision("d-rescue", defect=True, route=True, positive=True, reason="ad_hit"),
        decision("d-miss", defect=True, route=True, positive=False, reason="all_pass"),
        decision("g-yolo-fp", defect=False, route=False, positive=True, reason="yolo_hit"),
        decision("g-ad-fp", defect=False, route=True, positive=True, reason="ad_hit"),
        decision("g-pass", defect=False, route=True, positive=False, reason="all_pass"),
    ]
    counts = compute_confusion(decisions)
    for name, value in dict(tp=2, tn=1, fp=2, fn=1).items():
        assert counts[name] == value
    hybrid = compute_hybrid_metrics(decisions)
    expected = dict(route_rate=4 / 6, route_rate_good=2 / 3, route_rate_defect=2 / 3,
                    rescued_defects=1, rescue_rate=1 / 2, added_false_rejects=1,
                    unclassified_positive_rate=2 / 4)
    for name, value in expected.items():
        assert hybrid[name] == pytest.approx(value), name


def test_rescue_rate_zero_yolo_false_negatives_is_na():
    from pcb_lab.fusion.metrics import compute_hybrid_metrics
    decisions = [decision("d", defect=True, route=False, positive=True, reason="yolo_hit"),
                 decision("g", defect=False, route=True, positive=False, reason="all_pass")]
    result = compute_hybrid_metrics(decisions)
    assert result["rescued_defects"] == 0
    assert result["rescue_rate"] in (None, "N/A")


def test_delta_metrics_sign_is_cascade_minus_or():
    from pcb_lab.fusion.metrics import compute_delta_metrics
    result = compute_delta_metrics(dict(defect_recall=0.9, frr=0.02, ecpb=0.12),
                                   dict(defect_recall=0.8, frr=0.05, ecpb=0.25))
    assert result["delta_recall"] == pytest.approx(0.1)
    assert result["delta_frr"] == pytest.approx(-0.03)
    assert result["delta_ecpb"] == pytest.approx(-0.13)
