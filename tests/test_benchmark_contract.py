"""Step 6 contract tests, authored before inspecting implementation."""
import pytest
from helpers.step6_spec import MODELS, METRICS, ROUTING, artifact, csv_rows
from helpers.step6_bindings import benchmark_rows, build_case


def test_development_table_contains_all_models_on_both_allowed_splits():
    rows = benchmark_rows(artifact("development_summary.json"))
    pairs = [(row["model_id"][:3], row["split"]) for row in rows]
    assert len(pairs) == len(set(pairs)), "Duplicate model/split rows"
    assert set(pairs) == {(model, split) for model in MODELS for split in ("calibration", "fusion")}
    for row in rows:
        assert set(METRICS + ROUTING) <= row.keys()
        assert all(row[key] is not None for key in METRICS)
        assert row["split"] != "test"


def test_csv_preserves_table_metrics_and_plan_provenance():
    expected = {(r["model_id"][:3], r["split"]): r for r in benchmark_rows(artifact("development_summary.json"))}
    rows = benchmark_rows(csv_rows("development_summary.csv"))
    assert len(rows) == len(expected)
    for row in rows:
        original = expected[row["model_id"][:3], row["split"]]
        for key in METRICS:
            assert float(row[key]) == pytest.approx(original[key], abs=1e-10)
        for key in ("dataset_id", "protocol_id", "seed", "device"):
            assert row.get(key) not in (None, ""), key
        assert int(row["seed"]) == 42
        assert any("hash" in key.lower() and value for key, value in row.items())


@pytest.mark.parametrize("model", ["B01", "B02", "B03"])
@pytest.mark.parametrize("split", ["calibration", "fusion"])
def test_injected_test_row_is_denied_before_benchmark(model, split, tmp_path):
    case = build_case(tmp_path)
    case.mutate_prediction(model, split, {"split": "test"})
    with pytest.raises(PermissionError):
        case.run()


@pytest.mark.parametrize("model", ["B01", "B02", "B03"])
def test_wrong_model_id_is_rejected(model, tmp_path):
    case = build_case(tmp_path)
    case.mutate_prediction(model, "fusion", {"model_id": "UNRELATED_CHECKPOINT"})
    with pytest.raises(ValueError):
        case.run()


def test_benchmark_is_deterministic_and_keeps_locked_thresholds(tmp_path):
    case = build_case(tmp_path)
    first, second = benchmark_rows(case.run()), benchmark_rows(case.run())
    assert first == second
    for row in first:
        if row["model_id"].startswith("H"):
            assert row["tau_yolo"] == 0.5 and row["tau_ad"] == 0.5


def test_or_immediately_precedes_cascade_in_each_split():
    rows = benchmark_rows(artifact("development_summary.json"))
    for split in ("calibration", "fusion"):
        order = [row["model_id"][:3] for row in rows if row["split"] == split]
        assert order.index("H01") == order.index("H00") + 1
