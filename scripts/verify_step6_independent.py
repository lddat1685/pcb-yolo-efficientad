"""Cross-check every numeric benchmark/ablation cell using Python stdlib only.

No pcb_lab/numpy/yaml imports, no images/checkpoints/dataset manifest/test split.
The formulas imported below were committed by the verifier in spec-first A.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from helpers.step6_spec import MODELS, METRICS, ROUTING, oracle

SOURCES={'yolo_n':'yolo/B01_yolo11n','yolo_s':'yolo/B02_yolo11s','ad':'efficientad/B03_efficientad_s'}
CHECKS=[]
MISSING=object()


def read(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path,actual,expected,*,tolerance=1e-10):
    if actual is MISSING:
        status='missing'; actual='<missing>'
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        status='pass' if isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=tolerance) else 'mismatch'
    else:
        status='pass' if actual==expected else 'mismatch'
    CHECKS.append(dict(path=path,actual=actual,expected=expected,status=status))


def walk_cells(path,actual,expected):
    """Every scalar in the actual numeric table is accounted for, never dropped."""
    for key,value in actual.items():
        label=f'{path}/{key}'
        if isinstance(value,dict):
            walk_cells(label,value,expected.get(key,{}))
        elif key in expected:
            check(label,value,expected[key])
        else:
            CHECKS.append(dict(path=label,actual=value,status='unresolved',expected='No independently specified mapping'))


def parse_csv(value):
    if value=='': return None
    try: return json.loads(value)
    except (ValueError,TypeError): return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/handoff/step6-verify-evidence.json')
    args=parser.parse_args()
    bench=read(ROOT/'artifacts/benchmark/development_summary.json')
    abl=read(ROOT/'artifacts/benchmark/ablation_summary.json')
    short=read(ROOT/'artifacts/benchmark/shortlist.json')
    threshold=read(ROOT/'artifacts/fusion/thresholds.json')
    policy=read(ROOT/'artifacts/fusion/routing_policy.json')
    grid=read(ROOT/'artifacts/fusion/grid_calibration.json')
    sources={}; loaded={}; manifests={}
    for key,directory in SOURCES.items():
        meta=read(ROOT/'artifacts'/directory/'seed42/artifact.json')
        for split in ('calibration','fusion'):
            path=ROOT/'artifacts'/directory/'seed42'/f'preds_{split}.jsonl'
            records=[json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
            by_id={}
            for row in records:
                if row['split']=='test': raise PermissionError('Oracle refuses test rows')
                assert row['split']==split and row['sample_id'] not in by_id
                assert row['sample_id'].rsplit('_',1)[1] in {'good','defect'}
                assert row['error_reason'] is None and row['model_id']==meta['model_id']
                assert row['checkpoint_sha256']==meta.get('checkpoint_sha256',meta.get('checkpoint_best_sha256'))
                assert row['config_hash']==meta['config_hash']
                assert row['timing_ms'] is None
                score=row['anomaly_score'] if key=='ad' else row['yolo_image_score']
                assert math.isfinite(score)
                by_id[row['sample_id']]=row
            identity=f'{key}_{split}'; loaded[identity]=by_id
            digest=sha(path)
            sources[identity]=dict(path=path.relative_to(ROOT).as_posix(),rows=len(records),sha256=digest)
            check(f'policy/source_checksums/{identity}',policy['source_checksums'].get(identity,MISSING),digest)
            check(f'benchmark/source_checksums/{identity}',bench['source_checksums'].get(identity,MISSING),digest)
            check(f'benchmark/counts/{identity}',bench['counts'].get(identity,MISSING),len(records))
        manifest_path=ROOT/meta['run_manifest']['path']
        check(f'manifest/{key}/sha256',sha(manifest_path),meta['run_manifest']['sha256'])
        manifest=read(manifest_path)
        manifests[key]=dict(path=manifest_path.relative_to(ROOT).as_posix(),sha256=sha(manifest_path),
                            train_time_s=manifest['train_time_s'],gpu_run_elapsed_hours=manifest['train_time_s']/3600,
                            device=manifest['device'],seed=manifest['seed'])
    check('shortlist/budget/B01',short['budget_summary']['yolo_b01_gpu_hours'],round(manifests['yolo_n']['gpu_run_elapsed_hours'],2))
    check('shortlist/budget/B02',short['budget_summary']['yolo_b02_gpu_hours'],round(manifests['yolo_s']['gpu_run_elapsed_hours'],2))
    check('shortlist/budget/B03',short['budget_summary']['efficientad_b03_gpu_hours'],round(manifests['ad']['gpu_run_elapsed_hours'],2))
    assert not loaded['ad_calibration'].keys() & loaded['ad_fusion'].keys()

    # Secondary operating point derived independently from locked calibration grid.
    eligible=[p for p in grid if p['frr']<=.01]
    secondary=min(eligible,key=lambda p:(-p['defect_recall'],p['tau_yolo'],p['ecpb']))
    h01=threshold['h01_yolo11n_cascade']; h02=threshold['h02_yolo11s_cascade']
    points={'B01':(h01['tau_yolo'],h01['tau_ad']),'B02':(h02['tau_yolo'],h02['tau_ad']),
            'B03':(h01['tau_yolo'],h01['tau_ad']), 'H00':(h01['tau_yolo'],h01['tau_ad']),
            'H01':(h01['tau_yolo'],h01['tau_ad']),'H02':(h02['tau_yolo'],h02['tau_ad']),
            'H01_secondary':(secondary['tau_yolo'],secondary['tau_ad']),
            'H00_secondary':(secondary['tau_yolo'],secondary['tau_ad'])}
    expected={}
    for split in ('calibration','fusion'):
        yn,ys,ad=(loaded[f'{key}_{split}'] for key in ('yolo_n','yolo_s','ad'))
        assert yn.keys()==ys.keys()==ad.keys()
        samples=[]
        for sid in sorted(yn):
            assert yn[sid]['image_sha256']==ys[sid]['image_sha256']==ad[sid]['image_sha256']
            assert yn[sid]['source_group']==ys[sid]['source_group']==ad[sid]['source_group']
            samples.append((sid.endswith('_defect'),yn[sid]['yolo_image_score'],ys[sid]['yolo_image_score'],ad[sid]['anomaly_score']))
        expected[split]={}
        for method,(ty,ta) in points.items():
            model=method[:3]; result=oracle(samples,model,ty,ta)
            result.update(total=len(samples),n_defect=sum(s[0] for s in samples),n_good=sum(not s[0] for s in samples),
                          tau_yolo=None if model=='B03' else ty, tau_ad=None if model in {'B01','B02'} else ta,
                          route_rate=result['route_rate_total'],mean_latency_ms=None)
            if model.startswith('H'):
                result['yolo_fn_count']=sum(s[0] and s[2 if model=='H02' else 1]<ty for s in samples)
            expected[split][method]=result
        # The named H02-H00 comparison must reference the actual H00 (YOLO-n),
        # not an unreported OR variant using YOLO-s.
        deltas={}
        for short_key,long_key,metric in [('delta_recall','delta_recall_H01_minus_H00','defect_recall'),
            ('delta_frr','delta_FRR_H01_minus_H00','frr'),('delta_ecpb','delta_ECPB_H01_minus_H00','ecpb')]:
            value=expected[split]['H02'][metric]-expected[split]['H00'][metric]
            deltas[short_key]=deltas[long_key]=value
        deltas['delta_latency_ms_H01_minus_H00']=None
        expected[split]['H02']['deltas_H02_minus_H00']=deltas
        for method,row in bench['splits'][split].items():
            walk_cells(f'benchmark/json/{split}/{method}',row,expected[split][method])
            for key in METRICS+ROUTING:
                if key not in row:
                    check(f'benchmark/json/{split}/{method}/{key}',MISSING,expected[split][method][key])

    with (ROOT/'artifacts/benchmark/development_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        csv_rows=list(csv.DictReader(f))
    check('benchmark/csv/row_count',len(csv_rows),len(MODELS)*2)
    for row in csv_rows:
        split,model=row['split'],row['method']
        for key,value in row.items():
            if key in ('split','method','dataset_id','protocol_id','seed','device','artifact_hash','config_hash'): continue
            expected_cell=expected[split][model].get(key,None)
            # PLAN 7.2 explicitly permits N/A for YOLO-only AD routing. Its
            # unclassified-AD fraction is also inapplicable, not a numeric error.
            if model in {'B01','B02'} and value=='' and key in {
                'route_rate','route_rate_good','route_rate_defect','unclassified_positive_rate'}:
                expected_cell=None
            check(f'benchmark/csv/{split}/{model}/{key}',parse_csv(value),expected_cell)
        for key in ('dataset_id','protocol_id','seed','device','artifact_hash','config_hash'):
            if key not in row: check(f'benchmark/csv/{split}/{model}/{key}',MISSING,'required provenance')

    fusion=expected['fusion']; calibration=expected['calibration']
    ab_expected={}
    for key,actual in abl['ablations'].items():
        if key.startswith('A1'): base,treatment=fusion['B01'],fusion['B02']
        elif key.startswith('A2'): base,treatment=fusion['H00'],fusion['H01']
        else: base,treatment=fusion['H01'],fusion['H01_secondary']
        captions={
            'A1':'B02 (YOLO11s) minus B01 (YOLO11n), single-model, fusion split',
            'A2':'H01 (cascade) minus H00 (OR), same operating point, fusion split',
            'A3':'H01 secondary (FRR<=1%) minus H01 primary (FRR<=5%), fusion split'}
        exp={'comparison':captions[key[:2]]}
        for delta,metric in [('delta_recall','defect_recall'),('delta_frr','frr'),('delta_ecpb','ecpb'),
                             ('delta_balanced_accuracy','balanced_accuracy'),('delta_mcc','mcc')]:
            exp[delta]=treatment[metric]-base[metric]
        if key.startswith('A1'):
            exp.update(tau_yolo_B01=base['tau_yolo'],tau_yolo_B02=treatment['tau_yolo'],B01=base,B02=treatment)
        elif key.startswith('A2'):
            exp.update(tau_yolo=treatment['tau_yolo'],tau_ad=treatment['tau_ad'],delta_latency_ms=None,
                       route_rate_H01=treatment['route_rate'],route_rate_H00=base['route_rate'],
                       saving_rate_early_exit=1-treatment['route_rate'])
            for name in ('route_rate_good','route_rate_defect','rescued_defects','rescue_rate','added_false_rejects'):
                exp[name+'_H01']=treatment[name]
        else:
            exp.update(primary={name:base[name] for name in ('tau_yolo','tau_ad')},
                       secondary={name:treatment[name] for name in ('tau_yolo','tau_ad')},
                       H01_primary_fusion=base,H01_secondary_fusion=treatment,
                       H01_primary_calibration=calibration['H01'],H01_secondary_calibration=calibration['H01_secondary'])
        ab_expected[key]=exp
        walk_cells(f'ablation/json/{key}',actual,exp)
    with (ROOT/'artifacts/benchmark/ablation_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        ab_csv=list(csv.DictReader(f))
    check('ablation/csv/row_count',len(ab_csv),3)
    for row in ab_csv:
        key=row['ablation']; actual_json=abl['ablations'][key]
        for column,value in row.items():
            if column=='ablation': continue
            expected_cell=ab_expected[key].get(column) if column in actual_json else None
            check(f'ablation/csv/{key}/{column}',parse_csv(value),expected_cell)
    for row in short['shortlist']:
        model=row['model_id'][:3]
        for key,value in row['selection_metrics'].items():
            expected_value=1-fusion[model]['route_rate'] if key=='saving_rate' else fusion[model][key]
            check(f'shortlist/{model}/selection_metrics/{key}',value,expected_value)
    replay={}
    for name in ('development_summary.json','development_summary.csv','ablation_summary.json','ablation_summary.csv','shortlist.json'):
        replay[name]=sha(ROOT/'artifacts/benchmark'/name)==sha(ROOT/'.cache/step6-replay/artifacts'/name)
    suite={}
    for key,filename in [('targeted','step6-targeted.xml'),('full','step6-full-suite.xml'),
                         ('legacy_baseline','step6-legacy-baseline.xml')]:
        path=ROOT/'.cache'/filename
        if path.is_file():
            groups=list(ET.parse(path).getroot().iter('testsuite'))
            suite[key]={name:sum(int(g.get(name,0)) for g in groups) for name in ('tests','failures','errors','skipped')}
            suite[key]['failed_tests']=[c.get('classname')+'::'+c.get('name')
                for c in ET.parse(path).getroot().iter('testcase') if c.find('failure') is not None]
            suite[key]['skip_reasons']=dict(Counter(c.find('skipped').get('message')
                for c in ET.parse(path).getroot().iter('testcase') if c.find('skipped') is not None))
    totals=dict(Counter(c['status'] for c in CHECKS))
    output=dict(oracle='stdlib only; independent Phase A formulas',label_source='Validated upstream sample_id suffix; no dataset manifest read',
                sources=sources,training_manifests=manifests,
                unique_gpu_run_elapsed_hours=sum(v['gpu_run_elapsed_hours'] for v in manifests.values()),
                expected_tables=expected,check_totals=totals,checks=CHECKS,replay_identical=replay,pytest=suite)
    path=ROOT/args.output; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(check_totals=totals,pytest=suite,replay_identical=replay),indent=2))
    for item in CHECKS:
        if item['status']=='mismatch': print(json.dumps(item,ensure_ascii=False))
    print('Evidence:',path)
    return 1 if any(c['status']!='pass' for c in CHECKS) else 0


if __name__=='__main__':
    raise SystemExit(main())
