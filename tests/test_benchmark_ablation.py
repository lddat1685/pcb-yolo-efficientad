"""Independent ablation direction, savings and threshold comparisons."""
import pytest
from helpers.step6_spec import artifact, csv_rows
from helpers.step6_bindings import ablation_rows, benchmark_rows, build_case


def test_all_three_required_ablations_present():
    rows = ablation_rows(artifact("ablation_summary.json"))
    assert {"detector_capacity", "fusion_logic", "operating_threshold"} <= {r["kind"] for r in rows}
    assert all(row["split"] in {"calibration", "fusion"} for row in rows)


def test_ablation_csv_preserves_every_numeric_comparison():
    expected = {(r["kind"],r["split"]): r for r in ablation_rows(artifact("ablation_summary.json"))}
    rows = ablation_rows(csv_rows("ablation_summary.csv"))
    assert len(rows) == len(expected)
    for row in rows:
        original = expected[row["kind"],row["split"]]
        for key in ("delta_recall", "delta_frr", "delta_ecpb"):
            assert float(row[key]) == pytest.approx(original[key], abs=1e-10)


def test_ablation_deltas_and_early_exit_are_derived_from_benchmark(tmp_path):
    from pcb_lab.evaluation.ablation import compute_ablations
    result = build_case(tmp_path).run()
    models = {(r["model_id"][:3],r["split"]): r for r in benchmark_rows(result)}
    for row in ablation_rows(compute_ablations(result)):
        if row["kind"] not in {"detector_capacity", "fusion_logic"}:
            continue
        first, second = ("B01", "B02") if row["kind"] == "detector_capacity" else ("H00", "H01")
        baseline, treatment = models[first,row["split"]], models[second,row["split"]]
        for delta, metric in (("delta_recall","defect_recall"), ("delta_frr","frr"), ("delta_ecpb","ecpb")):
            assert row[delta] == pytest.approx(treatment[metric]-baseline[metric])
        if row["kind"] == "fusion_logic":
            assert row["delta_recall"] == row["delta_frr"] == row["delta_ecpb"] == 0.0
            assert row["saving_rate"] == pytest.approx(1-treatment["route_rate_total"])


def test_threshold_ablation_uses_distinct_calibrated_operating_points(tmp_path):
    from pcb_lab.evaluation.ablation import compute_ablations
    rows = ablation_rows(compute_ablations(build_case(tmp_path).run()))
    threshold = [r for r in rows if r["kind"] == "operating_threshold"]
    assert threshold
    for row in threshold:
        assert row["primary_target_frr"] == 0.05
        assert row["secondary_target_frr"] == 0.01
        assert (row["primary_tau_yolo"],row["primary_tau_ad"]) != (row["secondary_tau_yolo"],row["secondary_tau_ad"])
