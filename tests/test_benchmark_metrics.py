"""Check independent hand formulas against the complete benchmark API."""
import pytest
from helpers.step6_spec import MODELS, METRICS, ROUTING, artifact, binary, oracle, synthetic_samples
from helpers.step6_bindings import benchmark_rows, build_case


@pytest.mark.parametrize("scenario,ecpb", [("perfect",0.0), ("total_fail",1.99)])
def test_ecpb_extremes_from_synthetic_predictions(scenario, ecpb, tmp_path):
    rows = benchmark_rows(build_case(tmp_path, scenario).run())
    for row in rows:
        if row["split"] == "fusion":
            assert row["ecpb"] == pytest.approx(ecpb)
            assert row["balanced_accuracy"] == pytest.approx(1.0 if scenario == "perfect" else 0.0)


@pytest.mark.parametrize("model", MODELS)
def test_each_model_matches_independent_metric_and_routing_oracle(model, tmp_path):
    case = build_case(tmp_path)
    rows = benchmark_rows(case.run())
    actual = next(r for r in rows if r["model_id"][:3] == model and r["split"] == "fusion")
    expected = oracle(case.samples, model, 0.5, 0.5)
    for key in METRICS + ROUTING:
        if expected[key] is None:
            # For single models, rescue/added-FR are not applicable; zero is
            # acceptable for counts, but a zero denominator rate remains N/A.
            acceptable = (None, "N/A", 0) if key in ("rescued_defects", "added_false_rejects") else (None, "N/A")
            assert actual[key] in acceptable, key
        else:
            assert actual[key] == pytest.approx(expected[key]), key


def test_table_metrics_recompute_from_confusion_without_using_fusion_module():
    for row in benchmark_rows(artifact("development_summary.json")):
        expected = binary(*(row[k] for k in ("tp", "tn", "fp", "fn")))
        for key in METRICS:
            assert row[key] == pytest.approx(expected[key]), (row["model_id"], row["split"], key)


def test_routing_rate_bounds_and_unconditional_models():
    for row in benchmark_rows(artifact("development_summary.json")):
        model = row["model_id"][:3]
        for key in ROUTING[:3]:
            assert 0 <= row[key] <= 1
            if model in {"B01", "B02"}:
                assert row[key] == 0
            if model in {"B03", "H00"}:
                assert row[key] == 1
