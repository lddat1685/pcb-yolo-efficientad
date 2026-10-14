"""Execute isolated, synthetic probes against production guards; never alter originals."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import contextlib
import copy
import importlib.util
import io
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
import pcb_lab.evaluation.test_eval as ev
import pcb_lab.runtime.profiler as profiler

spec=importlib.util.spec_from_file_location("audit_export",ROOT/"scripts/export_runtime.py")
export=importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

base=json.loads((ROOT/"artifacts/test/frozen_evaluation.json").read_text())
data=ROOT.parent/"DatasetVer4_Public"
results=[]

# Guard must reject malformed freeze BEFORE calling any model loader.
class ReachedLoader(Exception):
    pass

for key,value in (("status","DRAFT"),("test_split_sha256","0"*64),("sample_count",999),
                  ("checkpoint_sha256","0"*64),("test_access_authorized",False)):
    cfg=copy.deepcopy(base)
    if key=="checkpoint_sha256":
        cfg["checkpoints"]["B01_yolo11n"]["sha256"]=value
    else:
        cfg[key]=value
    frozen=OUT/f"probe_frozen_{key}.json"
    frozen.write_text(json.dumps(cfg),encoding="utf-8")
    reached=False
    error=None
    with patch.object(ev.YoloDetector,"from_artifact",side_effect=ReachedLoader("model loading reached")):
        try:
            ev.run_test_evaluation(frozen,dataset_root=data,repo_root=ROOT,warmup_runs=0)
        except ReachedLoader:
            reached=True
        except Exception as exc:
            error=type(exc).__name__+": "+str(exc)
    results.append({"probe":"frozen_guard_"+key,"model_loading_reached":reached,"error":error,
                    "guard_passed":not reached})

# Execute synthetic 440-image evaluation. Frozen threshold 0.99 must reject
# score 0.5, while the hardcoded global 0.14 accepts it. All writes stay here.
fake_root=OUT/"synthetic_evaluation"
(fake_root/"artifacts/test").mkdir(parents=True,exist_ok=True)
cfg=copy.deepcopy(base)
cfg["operating_points"]["B01"]["tau_yolo"]=0.99
frozen=fake_root/"frozen.json"
frozen.write_text(json.dumps(cfg),encoding="utf-8")
det=NS(class_id=0,class_name="open_circuit",confidence=0.5,xyxy_original=(1,1,10,10))
model=NS(predict=lambda image:[det])
fake_image=NS(convert=lambda mode:object())
with patch.object(ev.YoloDetector,"from_artifact",return_value=model), \
     patch.object(ev.EfficientAdDetector,"from_artifact",return_value=object()), \
     patch.object(ev,"_run_efficientad_fast",return_value=(0.0,[])) as ad, \
     patch.object(ev.Image,"open",return_value=fake_image), contextlib.redirect_stdout(io.StringIO()):
    ev.run_test_evaluation(frozen,dataset_root=data,repo_root=fake_root,warmup_runs=0)
    ad_calls=ad.call_count
rows=[json.loads(x) for x in (fake_root/"artifacts/test/predictions_test.jsonl").read_text().splitlines()]
results.append({"probe":"frozen_threshold_used","frozen_B01_threshold":.99,"synthetic_yolo_score":.5,
    "observed_B01_defect_count":sum(r["decision"]=="DEFECT" for r in rows if r["model_id"]=="B01"),
    "expected_B01_defect_count":0,"ad_calls":ad_calls,
    "logged_H01_route_true":sum(r["route_ad"] for r in rows if r["model_id"]=="H01")})

# Corrupt export load must never produce successful optimized parity.
class Run:
    decision="GOOD"
    anomaly_score=.2
    sample_id="synthetic_calibration_good"
    yolo_image_score=0.0
    efficientad_executed=True
    early_exit=False
    def to_dict(self):
        return dict(sample_id=self.sample_id,decision=self.decision)

registry=NS(get=lambda model:NS(ad=object()))
with patch.object(export,"InferenceService",return_value=object()), \
     patch.object(export,"GalleryImageSource",return_value=object()), \
     patch.object(profiler,"profile_inspect",return_value=Run()), \
     patch("torch.jit.load",side_effect=RuntimeError("injected invalid TorchScript archive")):
    result=export.verify_parity([object()],"B03",registry,None,None,n_samples=1)
results.append({"probe":"export_load_failure","reported_parity":result["parity"]})

with patch.object(export,"InferenceService",return_value=object()), \
     patch.object(export,"GalleryImageSource",return_value=object()), \
     patch.object(profiler,"profile_inspect",return_value=Run()), \
     patch("torch.jit.load",return_value=NS(eval=lambda:None)):
    # object source lacks get_frame, so the verification try-block raises.
    result=export.verify_parity([object()],"B03",registry,None,None,n_samples=1)
results.append({"probe":"export_inference_exception","reported_parity":result["parity"]})

(OUT/"adversarial_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
