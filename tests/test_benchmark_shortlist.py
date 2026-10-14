"""Shortlist membership, fusion-based rationale, and budget disclosure."""
import math
import pytest
from helpers.step6_spec import artifact, recursive
from helpers.step6_bindings import benchmark_rows, build_case


def assert_schema(shortlist):
    assert type(shortlist["schema_version"]) is int and shortlist["schema_version"] == 1
    assert shortlist["selection_split"] == "fusion"
    assert shortlist["test_access"] is False
    entries = shortlist["shortlist"]
    ids = [r["model_id"][:3] for r in entries]
    assert len(ids) == len(set(ids))
    assert {"B01", "B03", "H00", "H01", "H02"} <= set(ids)
    for row in entries:
        assert isinstance(row["role"],str) and row["role"].strip()
        assert isinstance(row["rationale"],str) and row["rationale"].strip()
        assert isinstance(row["selection_metrics"],dict) and row["selection_metrics"]
        if row["model_id"].startswith("H"):
            assert math.isfinite(row["tau_yolo"]) and math.isfinite(row["tau_ad"])
    assert {"gpu_hours", "tuning_budget"} <= shortlist["budget_summary"].keys()


def test_shortlist_json_schema():
    assert_schema(artifact("shortlist.json"))


def test_shortlist_metrics_equal_fusion_rows_not_calibration():
    shortlist = artifact("shortlist.json")
    models = {r["model_id"][:3]: r for r in benchmark_rows(artifact("development_summary.json")) if r["split"] == "fusion"}
    for entry in shortlist["shortlist"]:
        source = models[entry["model_id"][:3]]
        for key,value in entry["selection_metrics"].items():
            if key == "saving_rate":
                assert value == pytest.approx(1-source["route_rate"])
                continue
            assert key in source, key
            if isinstance(value,(int,float)):
                assert value == pytest.approx(source[key]), (entry["model_id"],key)
            else:
                assert value == source[key]


def test_public_shortlist_selector_uses_real_benchmark_results(tmp_path):
    from pcb_lab.evaluation.ablation import compute_ablations
    from pcb_lab.evaluation.shortlist import select_shortlist
    result = build_case(tmp_path).run()
    shortlist = select_shortlist(result,compute_ablations(result))
    # Validate contents separately; Phase B has a dedicated strict Dict return test.
    if hasattr(shortlist,"to_dict"):
        shortlist=shortlist.to_dict()
    assert_schema(shortlist)
    models = {r["model_id"][:3]:r for r in benchmark_rows(result) if r["split"] == "fusion"}
    for entry in shortlist["shortlist"]:
        assert entry["selection_metrics"]["ecpb"] == pytest.approx(models[entry["model_id"][:3]]["ecpb"])


@pytest.mark.parametrize("bad_budget", [-1, float("nan"), float("inf")])
def test_invalid_resource_budget_is_rejected(bad_budget,tmp_path):
    from pcb_lab.evaluation.ablation import compute_ablations
    from pcb_lab.evaluation.shortlist import select_shortlist
    case = build_case(tmp_path)
    case.mutate_budget(bad_budget)
    with pytest.raises(ValueError):
        result = case.run()
        select_shortlist(result,compute_ablations(result))
