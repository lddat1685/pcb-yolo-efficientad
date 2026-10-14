"""Grid determinism, independent counts, and the ordered tie-break criteria."""
from copy import deepcopy
import statistics
import pytest
from helpers.step5_spec import (config, confusion_oracle, ec_pb, field, plain,
                                point_metrics, sweep_rows)


def run(*, yolo_false_positive=False, small=True):
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = sweep_rows(yolo_false_positive=yolo_false_positive)
    return grid_search_2d(yolo, ad, config(small=small)), yolo, ad


def test_grid_is_deterministic_without_mutating_inputs():
    from pcb_lab.fusion.sweep import grid_search_2d
    yolo, ad = sweep_rows()
    cfg = config(small=True)
    before = deepcopy((yolo, ad, cfg))
    first = grid_search_2d(yolo, ad, cfg)
    second = grid_search_2d(yolo, ad, cfg)
    assert plain(first) == plain(second)
    assert (yolo, ad, cfg) == before


def test_all_grid_points_saved_and_match_independent_confusion_counts():
    result, yolo, ad = run()
    points = field(result, "grid_points")
    assert len(points) == 3 * 5
    assert len({(field(p, "tau_yolo"), field(p, "tau_ad")) for p in points}) == 15
    for point in points:
        counts = confusion_oracle(yolo, ad, field(point, "tau_yolo"), field(point, "tau_ad"))
        metrics = point_metrics(point)
        assert metrics["defect_recall"] == pytest.approx(counts["tp"] / (counts["tp"] + counts["fn"]))
        assert metrics["frr"] == pytest.approx(counts["fp"] / (counts["fp"] + counts["tn"]))
        assert metrics["ecpb"] == pytest.approx(ec_pb(counts))


def test_default_grid_size_and_plan_ad_domain():
    # Explicit precedence: PLAN section 5 overrides the contract's wider
    # percentile envelope. See docs/handoff/questions-verify.md.
    result, _, ad = run(small=False)
    points = field(result, "grid_points")
    assert len(points) == 41 * 50
    yolo_values = sorted({field(point, "tau_yolo") for point in points})
    assert yolo_values == pytest.approx([0.1 + i * 0.02 for i in range(41)])
    scores = [row["anomaly_score"] for row in ad if row["is_defect"]]
    lower = statistics.mean(scores)
    # The spec does not fix ddof; either standard deviation convention is
    # acceptable. Compare rounded stored coordinates with a 5e-5 tolerance.
    upper_candidates = [lower + 3 * statistics.pstdev(scores),
                        lower + 3 * statistics.stdev(scores)]
    ad_values = sorted({field(point, "tau_ad") for point in points})
    assert ad_values[0] == pytest.approx(lower)
    assert any(ad_values[-1] == pytest.approx(upper, abs=5e-5) for upper in upper_candidates)


@pytest.mark.parametrize("name,limit", [("optimal_primary", 0.05), ("optimal_secondary", 0.01)])
def test_selection_matches_independent_ordered_objective(name, limit):
    result, yolo, ad = run(yolo_false_positive=True)
    scored = []
    for point in field(result, "grid_points"):
        ty, ta = field(point, "tau_yolo"), field(point, "tau_ad")
        counts = confusion_oracle(yolo, ad, ty, ta)
        frr = counts["fp"] / (counts["fp"] + counts["tn"])
        if frr <= limit:
            scored.append((-counts["tp"] / (counts["tp"] + counts["fn"]), ty, ec_pb(counts)))
    optimal = field(result, name)
    metrics = point_metrics(optimal)
    actual = (-metrics["defect_recall"], field(optimal, "tau_yolo"), metrics["ecpb"])
    assert actual == pytest.approx(min(scored))


def test_lower_yolo_threshold_takes_priority_over_lower_ecpb():
    result, _, _ = run(yolo_false_positive=True)
    primary = field(result, "optimal_primary")
    assert field(primary, "tau_yolo") == pytest.approx(0.1)
    assert point_metrics(primary)["ecpb"] == pytest.approx(0.99 / 20)
    assert any(point_metrics(point)["ecpb"] == 0 for point in field(result, "grid_points"))
    assert field(field(result, "optimal_secondary"), "tau_yolo") == pytest.approx(0.5)


def test_ecpb_breaks_equal_recall_and_equal_yolo_threshold_ties():
    result, _, _ = run()
    primary = field(result, "optimal_primary")
    assert field(primary, "tau_yolo") == pytest.approx(0.1)
    assert point_metrics(primary)["ecpb"] == pytest.approx(0.0)
    assert any(field(point, "tau_yolo") == pytest.approx(0.1) and
               point_metrics(point)["ecpb"] > 0 for point in field(result, "grid_points"))
