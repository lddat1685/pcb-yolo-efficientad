"""Versioned development-only DET, iso-FRR and fixed-weight resize ablation.

Post-test exploratory analysis: never changes the registered test thresholds.
Resize uses the same B03 checkpoint and normalization, PIL bilinear 640->256,
one tile, max anomaly score. It does not claim a retrained resize baseline.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pcb_lab.data.manifest import sha256_file, resolve_dataset_root
from pcb_lab.data.samples import ManifestDataset
from pcb_lab.fusion.sweep import grid_search_2d


def score_metrics(rows, tau, score_key):
    labels = np.asarray([r.get("is_defect", r["sample_id"].endswith("_defect")) for r in rows], dtype=bool)
    positive = np.asarray([r[score_key] for r in rows]) >= tau
    if not labels.any() or labels.all():
        raise ValueError("Both labels are required")
    return dict(recall=float(positive[labels].mean()), frr=float(positive[~labels].mean()),
                miss_rate=float((~positive[labels]).mean()), sample_count=len(rows))


def det_curve(rows, score_key):
    if any(r["split"] not in {"calibration", "fusion"} or r.get("error_reason") for r in rows):
        raise PermissionError("Only valid development predictions are permitted")
    scores = np.asarray([r[score_key] for r in rows], dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and nonempty")
    thresholds = [*sorted(set(scores)), float(np.nextafter(scores.max(), np.inf))]
    return [dict(threshold=float(t), **score_metrics(rows, t, score_key)) for t in thresholds]


def iso_frr(calibration, fusion, score_key):
    if {r["split"] for r in calibration} != {"calibration"} or {r["split"] for r in fusion} != {"fusion"}:
        raise PermissionError("Choose thresholds on calibration; evaluate on fusion only")
    curve = det_curve(calibration, score_key)
    result = []
    for cap in (.01, .03, .05):
        best = max((r for r in curve if r["frr"] <= cap),
                   key=lambda r: (r["recall"], -r["frr"], r["threshold"]))
        result.append(dict(frr_cap=cap, threshold=best["threshold"], calibration=best,
                           fusion=score_metrics(fusion, best["threshold"], score_key)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize-per-class", type=int, default=0,
                        help="0 runs all development images; positive value is a labelled pilot per class/split")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    torch.set_num_threads(args.threads)
    roots = dict(B01=REPO / "artifacts/yolo/B01_yolo11n/seed42",
                 B02=REPO / "artifacts/yolo/B02_yolo11s/seed42",
                 B03=REPO / "artifacts/efficientad/B03_efficientad_s/seed42")
    source_hashes, data, curves, iso = {}, {}, [], {}
    for mid, root in roots.items():
        data[mid] = {}
        key = "anomaly_score" if mid == "B03" else "yolo_image_score"
        for split in ("calibration", "fusion"):
            path = root / f"preds_{split}.jsonl"
            source_hashes[str(path.relative_to(REPO))] = sha256_file(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            data[mid][split] = rows
            curves.extend(dict(model_id=mid, split=split, **row) for row in det_curve(rows, key))
        iso[mid] = iso_frr(data[mid]["calibration"], data[mid]["fusion"], key)
    config_path = REPO / "configs/models/fusion_common.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_hashes[str(config_path.relative_to(REPO))] = sha256_file(config_path)
    h02 = grid_search_2d(data["B02"]["calibration"], data["B03"]["calibration"], config)
    (out / "grid_h02_calibration.json").write_text(json.dumps([p.to_dict() for p in h02.grid_points], indent=2), encoding="utf-8")
    with (out / "det_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (out / "iso_frr.json").write_text(json.dumps(iso, indent=2), encoding="utf-8")
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    detector = EfficientAdDetector.from_artifact(roots["B03"])
    checkpoint_hash = sha256_file(roots["B03"] / "model.pt")
    source_hashes["B03_checkpoint"] = checkpoint_hash
    resize_data, paired_tile = {}, {}
    ds = resolve_dataset_root(None)
    with (out / "resize_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for split in ("calibration", "fusion"):
            samples = sorted(ManifestDataset(ds, split), key=lambda s: s.sample_id)
            if args.resize_per_class:
                samples = [s for label in (False, True) for s in [s for s in samples if s.is_defect == label][:args.resize_per_class]]
            tile_rows = {r["sample_id"]: r for r in data["B03"][split]}
            resize_data[split], paired_tile[split] = [], []
            for i, sample in enumerate(samples):
                tile = tile_rows[sample.sample_id]
                if tile["image_sha256"] != sha256_file(sample.image_path) or tile["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError("Paired tile/resize provenance mismatch")
                image = Image.fromarray(np.asarray(sample.load_image())).resize((256, 256), Image.Resampling.BILINEAR)
                score = float(detector.engine.infer_tiles(np.asarray(image)[None]).max())
                row = dict(sample_id=sample.sample_id, split=split, is_defect=sample.is_defect,
                           source_group=sample.source_group, anomaly_score=score,
                           image_sha256=sample.sha256, checkpoint_sha256=checkpoint_hash)
                resize_data[split].append(row)
                paired_tile[split].append(tile)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"resize {split}: {i+1}/{len(samples)}", flush=True)
    ablation = dict(resize=iso_frr(resize_data["calibration"], resize_data["fusion"], "anomaly_score"),
                    tile=iso_frr(paired_tile["calibration"], paired_tile["fusion"], "anomaly_score"))
    (out / "tile_vs_resize.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    protocol = dict(status="COMPLETE", test_access=False, exploratory_after_test=True,
                    resize_scope="pilot" if args.resize_per_class else "all_development",
                    resize_per_class=args.resize_per_class, torch_threads=args.threads,
                    fixed_checkpoint=True, fixed_normalization=True, interpolation="PIL bilinear",
                    calibration_counts={k: len(v) for k, v in resize_data.items()},
                    source_sha256=source_hashes, limitations=["Single seed; no retrained resize model",
                    "Pilot cannot establish full-development ablation" ] if args.resize_per_class else ["Single seed; no retrained resize model"])
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
