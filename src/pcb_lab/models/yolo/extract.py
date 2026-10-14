"""Extract raw YOLO predictions for calibration/fusion (low fixed confidence floor).

Reads samples via the Step 2 ManifestDataset, runs the adapter, and writes one
deterministic JSONL row per image. Test (and anything outside calibration/fusion) is
blocked with PermissionError. Inference errors are recorded per-image, never dropped
or silently turned into GOOD/0.
"""
from __future__ import annotations

import json
from pathlib import Path

from pcb_lab.data.manifest import artifact_path, output_path, resolve_dataset_root, load_dataset, sha256_file
from pcb_lab.data.samples import ManifestDataset
from pcb_lab.models.yolo.adapter import YoloDetector


def _resolve_artifact_refs(artifact_dir):
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8-sig"))
    return meta


def extract_predictions(artifact_dir, dataset_root, partitions=("calibration", "fusion"),
                        out_dir=None) -> dict:
    artifact_dir = Path(artifact_dir)
    if "test" in partitions:
        raise PermissionError("extract_predictions refuses 'test' partition")
    for p in partitions:
        if p not in ("calibration", "fusion"):
            raise PermissionError(f"extract_predictions only allows calibration/fusion, got {p!r}")

    dataset_root = resolve_dataset_root(dataset_root if dataset_root is not None else None)
    out_dir = output_path(dataset_root, Path(out_dir) if out_dir is not None else artifact_dir)
    out_paths = {p: artifact_path(out_dir, f"preds_{p}.jsonl") for p in partitions}
    meta = _resolve_artifact_refs(artifact_dir)
    names = load_dataset(dataset_root).names

    detector = YoloDetector.from_artifact(artifact_dir, allow_smoke=meta.get("smoke", False))
    desc = detector.describe()
    infer_params = desc["infer_params"]

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"artifact_dir": str(artifact_dir), "partitions": {}, "errors": []}
    reference = meta["run_manifest"]
    recorded = Path(reference["path"])
    candidates = {recorded.resolve()} if recorded.is_absolute() else {
        (parent / recorded).resolve() for parent in (artifact_dir, *artifact_dir.resolve().parents)
        if (parent / recorded).is_file()
    }
    if len(candidates) != 1:
        raise ValueError("Run manifest path is missing or ambiguous")
    manifest_path = candidates.pop()
    if sha256_file(manifest_path) != reference["sha256"]:
        raise ValueError("Run manifest SHA256 mismatch")
    run_id = json.loads(manifest_path.read_text(encoding="utf-8-sig"))["run_id"]
    checkpoint_sha = desc.get("checkpoint_sha256")

    for partition in partitions:
        samples = ManifestDataset(dataset_root, partition)
        out_path = out_paths[partition]
        rows = []
        n_det = 0
        for sample in samples:
            row = {
                "run_id": run_id,
                "sample_id": sample.sample_id,
                "image_sha256": sample.sha256,
                "split": partition,
                "source_group": sample.source_group,
                "model_id": meta["model_id"],
                "checkpoint_sha256": checkpoint_sha,
                "config_hash": meta.get("config_hash"),
                "preprocessing_version": meta.get("preprocessing_version", "prep_v1"),
                "preprocessing_hash": meta.get("preprocessing_hash"),
                "inference_params": infer_params,
                "detections": [],
                "yolo_image_score": 0.0,
                "error_reason": None,
                "timing_ms": None,
            }
            try:
                dets = detector.predict(sample.load_image())
                row["detections"] = [
                    {"class_id": d.class_id, "class_name": d.class_name,
                     "confidence": round(d.confidence, 6),
                     "xyxy_original": [round(v, 2) for v in d.xyxy_original]}
                    for d in dets
                ]
                row["yolo_image_score"] = round(detector.image_score(dets), 6)
                n_det += len(dets)
            except Exception as exc:  # record, never drop
                row["error_reason"] = f"{type(exc).__name__}: {exc}"
                summary["errors"].append({"sample_id": sample.sample_id, "reason": row["error_reason"]})
            rows.append(row)
        # Deterministic: sort by sample_id, same output bytes for same inputs.
        rows.sort(key=lambda r: r["sample_id"])
        with out_path.open("w", encoding="utf-8") as stream:
            for r in rows:
                stream.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        summary["partitions"][partition] = {
            "n_images": len(rows), "n_detections": n_det, "out": str(out_path),
            "sha256": sha256_file(out_path),
        }
    return summary
