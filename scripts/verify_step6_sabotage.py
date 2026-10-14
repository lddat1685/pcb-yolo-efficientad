"""Synthetic-only Step 6 adversarial audit, using the real public APIs."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from helpers.step6_bindings import build_case, write_json
from pcb_lab.evaluation.ablation import compute_ablations
from pcb_lab.evaluation.shortlist import select_shortlist


def main():
    base=ROOT/'.cache/step6-sabotage-cases'; base.mkdir(parents=True,exist_ok=True)
    cases=[]
    def run(label,mutate,expected=ValueError,validate=None):
        root=Path(tempfile.mkdtemp(prefix='case-',dir=base)).resolve()
        assert root.is_relative_to(base.resolve())
        case=build_case(root)
        mutate(case)
        try:
            bench=case.run(); shortlist=select_shortlist(bench,compute_ablations(bench))
            data=shortlist.to_dict() if hasattr(shortlist,'to_dict') else shortlist
        except Exception as exc:
            cases.append(dict(case=label,expected=expected.__name__,observed=type(exc).__name__,message=str(exc),passed=isinstance(exc,expected)))
        else:
            actual=validate(data) if validate else None
            cases.append(dict(case=label,expected=expected.__name__ if not validate else 'source-derived B01 gpu_hours=2.0',
                              observed='accepted',value=actual,passed=(actual==2.0) if validate else False))
    for model in ('B01','B02','B03'):
        for split in ('calibration','fusion'):
            run(f'test row: {model}/{split}',lambda c,m=model,s=split:c.mutate_prediction(m,s,{'split':'test'}),PermissionError)
        run(f'wrong model ID: {model}',lambda c,m=model:c.mutate_prediction(m,'fusion',{'model_id':'WRONG_MODEL'}))
        run(f'wrong checkpoint: {model}',lambda c,m=model:c.mutate_prediction(m,'fusion',{'checkpoint_sha256':'0'*64}))
    for value in (-1.0,float('nan'),float('inf')):
        run(f'invalid training budget: {value}',lambda c,v=value:c.mutate_budget(v))
    run('changed valid training budget',lambda c:c.mutate_budget(2.0),
        validate=lambda data:next(r['gpu_hours'] for r in data['shortlist'] if r['model_id'].startswith('B01')))
    for mutation in ('missing','checksum','threshold','test_access'):
        def poison(case,mode=mutation):
            if mode=='missing': case.policy=case.root/'missing-policy.json'; return
            data=json.loads(case.policy.read_text('utf-8'))
            if mode=='checksum': data['source_checksums']['ad_fusion']='0'*64
            if mode=='threshold': data['tau_yolo']=.99
            if mode=='test_access': data['test_access']=True
            write_json(case.policy,data)
        run(f'policy {mutation}',poison,FileNotFoundError if mutation=='missing' else ValueError)
    output=dict(synthetic_only=True,real_test_access=False,observations=len(cases),passed=sum(c['passed'] for c in cases),cases=cases)
    path=ROOT/'docs/handoff/step6-sabotage-log.json'
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n','utf-8')
    print(f"{len(cases)} sabotage cases: {output['passed']} pass, {len(cases)-output['passed']} fail")
    print(path)
    return 1 if output['passed']!=len(cases) else 0


if __name__=='__main__':
    raise SystemExit(main())
