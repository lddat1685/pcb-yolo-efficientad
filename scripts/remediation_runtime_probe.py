"""Small real-checkpoint diagnostic; never claims a full canonical E2E benchmark."""
import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pcb_lab.inference.registry import ModelRegistry
from pcb_lab.inference.service import InferenceService
from pcb_lab.inference.sources import GalleryImageSource
from pcb_lab.evaluation.test_eval import _measure_inspection
from export_runtime import verify_parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    registry = ModelRegistry(REPO)
    service = InferenceService(registry)
    scores = [json.loads(line) for line in (REPO / "artifacts/yolo/B01_yolo11n/seed42/preds_calibration.jsonl").read_text().splitlines()]
    scores.sort(key=lambda r: r["yolo_image_score"])
    gallery = {s.sample_id: s for s in registry.gallery("calibration")}
    samples = [gallery[scores[i]["sample_id"]] for i in (0, -1)]
    cold = {}
    for mid in ("B01", "B02", "B03"):
        _, cold[mid] = _measure_inspection(service, GalleryImageSource(samples[0]), mid)
    with (args.output_dir / "pipeline_runs.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            for mid in registry.specs:
                result, elapsed = _measure_inspection(service, GalleryImageSource(sample), mid)
                record = result.to_dict()
                record["latency_e2e_ms"] = elapsed
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(mid, sample.sample_id, result.decision, round(elapsed, 2), flush=True)
    spec = registry.get("B03")
    path = REPO / "artifacts/runtime/B03/efficientad/efficientad.pt"
    if not path.is_file():
        path = REPO / "artifacts/runtime/H01/efficientad/efficientad.pt"
    parity = verify_parity(samples, "B03", registry, None, spec.ad.checkpoint.parent,
                           n_samples=len(samples), exported_ad_path=path)
    (args.output_dir / "torchscript_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    (args.output_dir / "protocol.json").write_text(json.dumps(dict(
        scope="diagnostic_only", split="calibration", samples=2, timing_rounds=1,
        threads=args.threads, warmup="one call each B01/B02/B03", cold_ms=cold,
        limitations="Not the canonical 440-image / 3-round latency experiment",
        parity_passed=parity["parity"]["passed"]), indent=2), encoding="utf-8")
    if not parity["parity"]["passed"]:
        raise SystemExit("TorchScript parity failed; see evidence")


if __name__ == "__main__":
    main()
