"""Adversarial API/provenance checks added only after the Phase A commit."""
import json
import pytest
from helpers.step6_bindings import build_case, write_json


@pytest.mark.parametrize("mutation", ["missing", "test_access", "checksum", "threshold"])
def test_locked_policy_is_actually_validated(tmp_path,mutation):
    case=build_case(tmp_path)
    policy=json.loads(case.policy.read_text('utf-8'))
    if mutation=="missing":
        case.policy=tmp_path/'absent-policy.json'
    else:
        if mutation=="test_access": policy['test_access']=True
        if mutation=="checksum": policy['source_checksums']['ad_fusion']='0'*64
        if mutation=="threshold": policy['tau_yolo']=.99
        write_json(case.policy,policy)
    with pytest.raises((PermissionError,ValueError,FileNotFoundError)):
        case.run()


@pytest.mark.parametrize("entry",["ablation","shortlist"])
def test_public_functions_return_dict_as_contract_states(tmp_path,entry):
    from pcb_lab.evaluation.ablation import compute_ablations
    from pcb_lab.evaluation.shortlist import select_shortlist
    result=build_case(tmp_path).run()
    ablation=compute_ablations(result)
    value=ablation if entry=="ablation" else select_shortlist(result,ablation)
    assert isinstance(value,dict)


def test_rationale_does_not_fabricate_perfect_recall(tmp_path):
    from pcb_lab.evaluation.ablation import compute_ablations
    from pcb_lab.evaluation.shortlist import select_shortlist
    case=build_case(tmp_path,"total_fail")
    bench=case.run()
    result=select_shortlist(bench,compute_ablations(bench))
    data=result.to_dict() if hasattr(result,'to_dict') else result
    for row in data['shortlist']:
        assert row['selection_metrics']['defect_recall']==0
        assert '100% defect recall' not in row['rationale']
        if row['model_id'].startswith('B03'):
            assert 'FRR=0.0%' not in row['rationale']
            assert 'precision=100%' not in row['rationale']


def test_budget_tracks_changed_source_manifest(tmp_path):
    from pcb_lab.evaluation.ablation import compute_ablations
    from pcb_lab.evaluation.shortlist import select_shortlist
    case=build_case(tmp_path)
    case.mutate_budget(2.0)
    bench=case.run()
    shortlist=select_shortlist(bench,compute_ablations(bench))
    data=shortlist.to_dict() if hasattr(shortlist,'to_dict') else shortlist
    b01=next(row for row in data['shortlist'] if row['model_id'].startswith('B01'))
    assert b01['gpu_hours']==2.0
