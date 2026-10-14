"""Opt-in real-checkpoint CLI/AppTest parity on calibration only; no artifact writes."""
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from streamlit.testing.v1 import AppTest
from pcb_lab.inference.registry import ModelRegistry

FIELDS = ["sample_id", "model_id", "mode", "decision", "reason", "routing_trace",
          "yolo_executed", "efficientad_executed", "early_exit", "yolo_image_score",
          "anomaly_score", "detections", "anomaly_regions", "heatmap", "error_reason"]


def main():
    settings = REPO / ".cache/step8-ultralytics"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    registry = ModelRegistry()
    samples = registry.gallery("calibration")
    defect = next(s.sample_id for s in samples if s.is_defect)
    good = next(s.sample_id for s in samples if not s.is_defect)
    cases = [(model, defect) for model in registry.specs] + [("H01", good)]
    # Select an AD-positive calibration example from existing metadata if available.
    cache = REPO / "artifacts/efficientad/B03_efficientad_s/seed42/preds_calibration.jsonl"
    if cache.exists():
        ids = {s.sample_id for s in samples}
        for line in cache.read_text(encoding="utf-8-sig").splitlines():
            row = json.loads(line)
            if row["sample_id"] in ids and row["anomaly_score"] >= registry.get("B03").tau_ad:
                cases.append(("B03", row["sample_id"]))
                break
        yolo_cache = REPO / "artifacts/yolo/B01_yolo11n/seed42/preds_calibration.jsonl"
        if yolo_cache.exists():
            scores = {r["sample_id"]: r["anomaly_score"] for r in map(json.loads, cache.read_text(encoding="utf-8-sig").splitlines())}
            good_ids = {s.sample_id for s in samples if not s.is_defect}
            for line in yolo_cache.read_text(encoding="utf-8-sig").splitlines():
                row = json.loads(line)
                if (row["sample_id"] in good_ids and row["yolo_image_score"] < registry.get("H01").tau_yolo
                        and scores[row["sample_id"]] < registry.get("H01").tau_ad):
                    cases.append(("H01", row["sample_id"]))
                    break
    app = AppTest.from_file(str(REPO / "src/pcb_lab/ui/app.py"), default_timeout=180).run()
    assert not app.exception
    app.radio(key="single_source").set_value("Gallery").run()
    evidence = []
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    for model, sample_id in cases:
        app.sidebar.selectbox[0].set_value(model).run()
        app.selectbox(key="single_sample").set_value(sample_id).run()
        app.button[0].click().run()
        assert not app.exception, app.exception
        ui = json.loads(json.dumps(app.session_state.single_result[1].to_dict()))
        command = [sys.executable, str(REPO / "scripts/inspect_image.py"), "--model", model,
                   "--sample-id", sample_id, "--split", "calibration"]
        process = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert process.returncode == 0, process.stdout + process.stderr
        cli = json.loads(process.stdout)
        assert ui["decision"] != "ERROR", ui
        for field in FIELDS:
            assert ui[field] == cli[field], f"{model}/{sample_id}: {field} differs: UI={ui[field]}, CLI={cli[field]}"
        row = {"model": model, "sample_id": sample_id, "decision": ui["decision"],
               "early_exit": ui["early_exit"], "ad_executed": ui["efficientad_executed"],
               "yolo_score": ui["yolo_image_score"], "ad_score": ui["anomaly_score"],
               "boxes": len(ui["detections"]), "exact_fields": len(FIELDS)}
        evidence.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"status": "PASS", "cases": len(evidence), "comparison": "exact equality; timings/run IDs excluded"}), flush=True)


if __name__ == "__main__":
    main()
