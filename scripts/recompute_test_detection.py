"""Recompute AP50/AP50:95 from archived detection evidence without inference.

Accepts audit replay boxes as new evidence; never relabels them as the original
Step 7 run. Verifies frozen checkpoint/image hashes and exact canonical coverage.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pcb_lab.data.manifest import resolve_dataset_root, sha256_file
from pcb_lab.evaluation.detection import evaluate_detections
from pcb_lab.evaluation.test_eval import CLASSES, _validate_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen_path = REPO / "artifacts/test/frozen_evaluation.json"
    frozen = json.loads(frozen_path.read_text())
    ds = resolve_dataset_root(None)
    ids = set(_validate_frozen(frozen, REPO, ds))
    manifest_path = ds / "benchmarks/deeppcb/manifests/samples.jsonl"
    truth = {r["sample_id"]: r for r in map(json.loads, manifest_path.read_text().splitlines()) if r["sample_id"] in ids}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_class, provenance = [], {}
    for mid, key in (("B01", "B01_yolo11n"), ("B02", "B02_yolo11s")):
        path = args.replay_dir / f"replay_{mid}_test_boxes.jsonl"
        rows = list(map(json.loads, path.read_text().splitlines()))
        if len(rows) != 440 or {r["sample_id"] for r in rows} != ids:
            raise PermissionError("Replay coverage differs from frozen test")
        for row in rows:
            gt = truth[row["sample_id"]]
            if row["checkpoint_sha256"] != frozen["checkpoints"][key]["sha256"] or row["image_sha256"] != gt["sha256"]:
                raise PermissionError("Replay provenance mismatch")
            row["ground_truth_boxes"] = gt["boxes"]
        per_class.extend(evaluate_detections(rows, CLASSES, frozen["operating_points"][mid]["tau_yolo"], model_id=mid))
        # Self-contained raw boxes + GT for recalculation, explicitly labelled replay.
        (args.output_dir / f"replay_{mid}_with_ground_truth.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        provenance[path.name] = sha256_file(path)
    report = dict(evidence="independent audit replay, not original Step 7 detections",
                  source_sha256=provenance, frozen_sha256=sha256_file(frozen_path),
                  manifest_sha256=sha256_file(manifest_path), per_class=per_class)
    (args.output_dir / "detection_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({mid: {key: sum(r[key] for r in per_class if r["model_id"] == mid)/6
                            for key in ("ap_50", "ap_50_95")} for mid in ("B01", "B02")}, indent=2))


if __name__ == "__main__":
    main()
