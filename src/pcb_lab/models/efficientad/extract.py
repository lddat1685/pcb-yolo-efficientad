"""Deterministic, one-row-per-manifest-image calibration/fusion predictions."""
import json
from pathlib import Path

from pcb_lab.data.manifest import artifact_path, output_path, sha256_file
from pcb_lab.data.samples import EvalSet
from .adapter import EfficientAdDetector, SmokeArtifactError


def extract_efficientad_predictions(artifact_dir, dataset_root, partitions=("calibration", "fusion"),
                                    out_dir=None, *, allow_smoke=False, limit=None, device="cpu") -> dict:
    partitions = tuple(partitions)
    if not partitions or any(p not in ("calibration", "fusion") for p in partitions):
        raise PermissionError("Only calibration/fusion prediction extraction is permitted")
    if len(set(partitions)) != len(partitions):
        raise ValueError("Duplicate partitions are not permitted")
    detector = EfficientAdDetector.from_artifact(artifact_dir, allow_smoke=allow_smoke, device=device)
    meta = detector.meta
    if limit is not None and (not meta["smoke"] or type(limit) is not int or limit <= 0):
        raise ValueError("A positive extraction limit is permitted only for smoke artifacts")
    destination = output_path(Path(dataset_root), Path(out_dir or artifact_dir))
    if meta["smoke"] and "artifacts" in {p.lower() for p in destination.parts}:
        raise SmokeArtifactError("Smoke predictions must not be written into official artifacts/")
    datasets = {p: EvalSet(dataset_root, p) for p in partitions}
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for partition, dataset in datasets.items():
        samples = sorted(dataset, key=lambda sample: sample.sample_id)
        if limit is not None:
            samples = samples[:limit]
        target = artifact_path(destination, f"preds_{partition}.jsonl")
        temporary = artifact_path(destination, f"preds_{partition}.jsonl.tmp")
        errors = 0
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for sample in samples:
                row = {"run_id": meta["run_id"], "sample_id": sample.sample_id,
                    "image_sha256": sample.sha256, "split": partition, "source_group": sample.source_group,
                    "model_id": meta["model_id"], "checkpoint_sha256": meta["checkpoint_sha256"],
                    "config_hash": meta["config_hash"], "preprocessing_version": meta["preprocessing_version"],
                    "tile_params": meta["tile_params"], "anomaly_score": None, "boxes": [],
                    "error_reason": None, "timing_ms": None}
                try:
                    if sample.partition != partition:
                        raise PermissionError("Sample does not belong to requested partition")
                    if sha256_file(sample.image_path) != sample.sha256:
                        raise ValueError("Image SHA256 differs from manifest")
                    prediction = detector.predict(sample.load_image())
                    row["anomaly_score"] = prediction.anomaly_score
                    row["boxes"] = [{"xyxy": list(box.xyxy_original), "confidence": box.confidence,
                                     "class_name": box.class_name} for box in prediction.boxes]
                except Exception as exc:
                    errors += 1
                    row["error_reason"] = f"{type(exc).__name__}: {exc}"
                    row["anomaly_score"], row["boxes"] = None, []
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(target)
        result[partition] = {"path": str(target), "n_rows": len(samples), "n_errors": errors,
                             "sha256": sha256_file(target)}
    return result
