"""Phase B bindings; no model, metric or ablation computation is mocked.

The runner hard-codes 460/306 rows. Synthetic fixtures use those sizes, with
unique fake IDs, to reach the actual behavior under test. Every expected metric
is still computed by the independent Phase A stdlib oracle.
"""
import hashlib
import json
import re
from pathlib import Path
from helpers.step6_spec import MODELS, oracle, synthetic_samples

DIRS = {"B01": "yolo/B01_yolo11n", "B02": "yolo/B02_yolo11s", "B03": "efficientad/B03_efficientad_s"}
CHECKSUM_KEYS = {"B01": "yolo_n", "B02": "yolo_s", "B03": "ad"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class SyntheticCase:
    def __init__(self, root, scenario):
        self.root = Path(root)
        pattern = synthetic_samples(scenario)
        defects = [s for s in pattern if s[0]]
        goods = [s for s in pattern if not s[0]]
        self.samples = [pool[i % len(pool)] for i in range(153) for pool in (defects, goods)]
        self.calibration = [(True,.9,.9,.9) for _ in range(230)] + [
            (False,.6 if i < 10 else .1,.6 if i < 10 else .1,.1) for i in range(230)]
        self.config = self.root / "configs/models/fusion_common.yaml"
        self.thresholds = self.root / "artifacts/fusion/thresholds.json"
        self.policy = self.root / "artifacts/fusion/routing_policy.json"
        write_json(self.config, {"recipe_id": "fusion_recipe_v1",
            "constraints": {"primary_frr_max": .05, "secondary_frr_max": .01},
            "cost_matrix": {"c_fn":100., "c_fp":1., "defect_prevalence":.01,"n_simulation":10000}})
        threshold_data = {"schema_version":1,"calibration_split":"calibration","target_frr":.05}
        for key, model in (("h01_yolo11n_cascade","H01"), ("h02_yolo11s_cascade","H02"), ("h00_yolo11n_or","H00")):
            fusion = {"h01_cascade": oracle(self.samples,model if model != "H00" else "H01",.5,.5),
                      "h00_or": oracle(self.samples,"H00",.5,.5)}
            threshold_data[key] = {"tau_yolo":.5,"tau_ad":.5,"fusion_eval":fusion,
                                   "calibration_metrics":oracle(self.calibration,model,.5,.5)}
        write_json(self.thresholds,threshold_data)
        grid = [dict(oracle(self.calibration,"H01",ty,.5),tau_yolo=ty,tau_ad=.5) for ty in (.5,.8)]
        write_json(self.thresholds.with_name("grid_calibration.json"),grid)
        checksums={}
        for model,directory in DIRS.items():
            checkpoint = hashlib.sha256(model.encode()).hexdigest()
            for split,samples in (("calibration",self.calibration),("fusion",self.samples)):
                records=[]
                for i,(label,yn,ys,ad) in enumerate(samples):
                    row={"sample_id":f"synthetic_{split}_{i:04}_{'defect' if label else 'good'}",
                         "split":split,"is_defect":label,"error_reason":None,"timing_ms":None,
                         "model_id":directory.split('/')[-1],"checkpoint_sha256":checkpoint,
                         "config_hash":checkpoint,"image_sha256":hashlib.sha256(f'{split}:{i}'.encode()).hexdigest(),
                         "source_group":"synthetic-group","preprocessing_version":"prep_v1"}
                    if model == "B03": row.update(anomaly_score=ad,boxes=[])
                    else: row.update(yolo_image_score=yn if model=="B01" else ys,detections=[])
                    records.append(row)
                path=self.prediction_path(model,split)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(json.dumps(r)+'\n' for r in records),'utf-8')
                checksums[f'{CHECKSUM_KEYS[model]}_{split}']=hashlib.sha256(path.read_bytes()).hexdigest()
            run_rel=Path('runs')/directory/'seed42/run_manifest.json'
            write_json(self.root/run_rel,{"model_id":directory.split('/')[-1],"seed":42,"device":"cuda:0",
                       "train_time_s":3600.,"epochs_run":1,"pretrained":{"source":"synthetic"}})
            write_json(self.root/'artifacts'/directory/'seed42/artifact.json',{
                "model_id":directory.split('/')[-1],"seed":42,"config_hash":checkpoint,
                "checkpoint_sha256":checkpoint,"checkpoint_best_sha256":checkpoint,
                "run_manifest":{"path":run_rel.as_posix(),"sha256":hashlib.sha256((self.root/run_rel).read_bytes()).hexdigest()}})
        write_json(self.policy,{"test_access":False,"tau_yolo":.5,"tau_ad":.5,"source_checksums":checksums})

    def prediction_path(self,model,split):
        return self.root/'artifacts'/DIRS[model]/'seed42'/f'preds_{split}.jsonl'

    def mutate_prediction(self,model,split,changes):
        path=self.prediction_path(model,split)
        records=[json.loads(line) for line in path.read_text('utf-8').splitlines()]
        records[0].update(changes)
        path.write_text(''.join(json.dumps(r)+'\n' for r in records),'utf-8')
        # Keep the checksum internally consistent to isolate row/model validation.
        policy=json.loads(self.policy.read_text('utf-8'))
        policy['source_checksums'][f'{CHECKSUM_KEYS[model]}_{split}']=hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(self.policy,policy)

    def mutate_budget(self,value):
        path=self.root/'runs/yolo/B01_yolo11n/seed42/run_manifest.json'
        data=json.loads(path.read_text('utf-8')); data['train_time_s']=value*3600
        write_json(path,data)
        meta_path=self.root/'artifacts/yolo/B01_yolo11n/seed42/artifact.json'
        meta=json.loads(meta_path.read_text('utf-8'))
        meta['run_manifest']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(meta_path,meta)

    def run(self):
        from pcb_lab.evaluation.benchmark import run_development_benchmark
        return run_development_benchmark(self.config,self.thresholds,self.policy,repo_root=self.root)


def build_case(tmp_path, scenario="mixed"):
    """Provide run(), mutate_prediction(model, split, changes), mutate_budget().

    Fixture supplies all three upstream models, allowed splits, locked 0.5
    primary thresholds, and a distinct secondary threshold. It must exercise
    the public run_development_benchmark without mocking metric computation.
    """
    return SyntheticCase(tmp_path,scenario)


def benchmark_rows(result):
    """Return flat rows with model_id, split, metrics, and original metadata."""
    if hasattr(result,"to_dict"): result=result.to_dict()
    if isinstance(result,list):
        return [dict(row,model_id=row.get("model_id",row.get("method"))) for row in result]
    return [dict(row,model_id=model,split=split) for split,methods in result["splits"].items()
            for model,row in methods.items() if model in MODELS]


def ablation_rows(result):
    """Return rows: kind, split, delta_recall/frr/ecpb, saving_rate, if applicable.

    Canonical signs: detector B02-B01, logic H01-H00, threshold secondary-primary.
    Any representation mapping/sign convention must be documented in Phase B.
    """
    if hasattr(result,"to_dict"): result=result.to_dict()
    pairs = [(r["ablation"],r) for r in result] if isinstance(result,list) else result["ablations"].items()
    kinds={"A1":"detector_capacity","A2":"fusion_logic","A3":"operating_threshold"}
    rows=[]
    for key,value in pairs:
        row=dict(value,kind=kinds[key[:2]],split="fusion")
        if 'saving_rate_early_exit' in row: row['saving_rate']=row['saving_rate_early_exit']
        if 'primary' in row:
            for stage in ('primary','secondary'):
                for threshold in ('tau_yolo','tau_ad'):
                    row[f'{stage}_{threshold}']=row[stage][threshold]
            # Extract declared targets from the comparison label, not new data.
            row['primary_target_frr']=float(re.search(r'primary \(FRR<=(\d+)%',row['comparison'])[1])/100
            row['secondary_target_frr']=float(re.search(r'secondary \(FRR<=(\d+)%',row['comparison'])[1])/100
        rows.append(row)
    return rows
