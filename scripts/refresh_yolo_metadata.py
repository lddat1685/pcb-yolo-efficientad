#!/usr/bin/env python
"""Revalidate existing YOLO checkpoints and refresh metadata without training.

Example (run from the repository root):
    python scripts/refresh_yolo_metadata.py --source-commit 7623c11d88e4502c879d6f0efa71aab4ae495839 --bundle exports/colab_bundle.zip

The original bundle must match the source commit before git_dirty=False can be
recovered. Validation uses calibration only; checkpoint and prediction hashes
are checked before and after. Original metadata is backed up under scratch/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def verify_bundle(root, bundle, commit):
    """Recover source provenance from the actual archive, not the edited worktree."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    resolved = git("rev-parse", f"{commit}^{{commit}}").decode().strip()
    tracked = set(git("ls-tree", "-r", "--name-only", resolved).decode().splitlines())
    with zipfile.ZipFile(bundle) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        required = {"src/pcb_lab/models/yolo/train.py", "configs/models/yolo11n.yaml",
                    "configs/models/yolo11s.yaml", "scripts/train_yolo.py"}
        if not required <= set(names):
            raise ValueError("Bundle lacks required training source/configuration")
        for name in names:
            if name not in tracked:
                raise ValueError(f"Cannot establish clean bundle provenance: untracked {name}")
            original = git("show", f"{resolved}:{name}")
            if archive.read(name).replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
                raise ValueError(f"Bundle differs from source commit: {name}")
    return resolved, len(names)


def dataset_inventory(root):
    """Stat entries only, including test paths; never open a test image."""
    return {p.relative_to(root).as_posix(): (p.is_dir(), p.stat().st_size if p.is_file() else 0,
                                           p.stat().st_mtime_ns)
            for p in root.rglob("*")}


def refresh(args):
    root = ROOT.resolve()
    work = root / "scratch" / "metadata_refresh"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "Ultralytics"))
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "true")
    os.environ.setdefault("ULTRALYTICS_NO_ANALYTICS", "1")
    import torch
    import ultralytics
    from pcb_lab.data.manifest import load_dataset, output_path, resolve_dataset_root, sha256_file
    from pcb_lab.models.yolo.train import (
        _config_hash, _merge_configs, _read_training_summary, _release_files_sha,
        _validate, _write_model_card,
    )
    from pcb_lab.models.yolo.view import build_yolo_view

    dataset_root = resolve_dataset_root(args.dataset_root)
    output_path(dataset_root, root)
    source_commit, bundle_files = verify_bundle(root, args.bundle, args.source_commit)
    dataset_before = dataset_inventory(dataset_root)
    protected = {}
    for model_id in args.models:
        for folder in (root / "artifacts/yolo" / model_id / f"seed{args.seed}",
                       root / "runs/yolo" / model_id / f"seed{args.seed}"):
            for path in sorted(set(folder.rglob("*.pt")) | set(folder.rglob("preds_*.jsonl"))):
                protected[path] = sha256_file(path)
    work.mkdir(parents=True, exist_ok=True)
    view = build_yolo_view(dataset_root, work / "view", partitions=("train", "calibration"))
    release_hash = _release_files_sha(load_dataset(dataset_root).release)
    pending = {}
    summaries = {}
    for model_id in args.models:
        art_dir = root / "artifacts/yolo" / model_id / f"seed{args.seed}"
        run_dir = root / "runs/yolo" / model_id / f"seed{args.seed}"
        artifact = read_json(art_dir / "artifact.json")
        run = read_json(run_dir / "run_manifest.json")
        calibration = read_json(art_dir / "calibration_per_class.json")
        config = _merge_configs([root / "configs/models/yolo_common.yaml",
                                 root / "configs/models" / (run["architecture"] + ".yaml")])
        if not (artifact["config_hash"] == run["config_hash"] == _config_hash(config)
                and artifact["view_signature"] == run["view_signature"] == view.view_signature
                and run["dataset_release_sha256"] == release_hash):
            raise ValueError(f"{model_id}: configuration/dataset identity mismatch")
        if artifact["inference_params"] != config["infer"] or run["smoke"] or artifact["smoke"]:
            raise ValueError(f"{model_id}: incompatible inference configuration or smoke artifact")
        if run["ultralytics_version"] != ultralytics.__version__:
            raise ValueError("Use the original Ultralytics version for metadata refresh")
        if run["git_commit"] not in (None, source_commit):
            raise ValueError("Recorded source commit differs from the supplied bundle")
        for kind in ("best", "last"):
            expected = artifact[f"checkpoint_{kind}_sha256"]
            if not (sha256_file(art_dir / f"{kind}.pt") == expected
                    == sha256_file(run_dir / "weights" / f"{kind}.pt")
                    == run["checkpoints"][f"{kind}_sha256"]):
                raise ValueError(f"{model_id}: checkpoint hash mismatch")

        training = _read_training_summary(run_dir, run["args_used"]["epochs"])
        print(f"Refreshing {model_id}: calibration only, rect=False, imgsz={config['infer']['imgsz']}", flush=True)
        overall, per_class = _validate(model_id, art_dir / "best.pt", view,
                                       config["infer"], args.device,
                                       save_dir=work / "validation" / model_id, batch=args.batch)
        if overall["pr_conf"] is None or set(per_class) != set(artifact["class_order"]):
            raise ValueError("Incomplete calibration metrics")
        if {k: v["n_gt_boxes"] for k, v in per_class.items()} != view.box_counts["calibration"]:
            raise ValueError("Calibration ground-truth counts disagree with the manifest")
        validation_context = {
            "split": "calibration", "rect": False, "imgsz": config["infer"]["imgsz"],
            "batch": args.batch, "device": args.device, "half": config["infer"].get("half", False),
            "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
            "python": platform.python_version(), "checkpoint_sha256": artifact["checkpoint_best_sha256"],
        }
        run.update(training, git_commit=source_commit, git_dirty=False)
        run["validation"] = {"overall": overall, "per_class": per_class, "context": validation_context}
        run["git_provenance"] = {"source": "original Colab bundle matched against source commit",
                                  "bundle_sha256": sha256_file(args.bundle), "verified_files": bundle_files}
        calibration.update(overall=overall, per_class=per_class,
                           source="ultralytics validate on calibration (rect=False; metadata refresh without training)",
                           validation_context=validation_context)
        manifest_bytes = json_bytes(run)
        artifact["created_by_git_commit"] = source_commit
        artifact["run_manifest"]["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        if (root / artifact["run_manifest"]["path"]).resolve() != (run_dir / "run_manifest.json").resolve():
            raise ValueError("Unexpected run_manifest path")
        card = work / f"{model_id}_model_card.md"
        _write_model_card(card, model_id, run["architecture"], config, view, overall, per_class,
                          ultralytics_version=run["ultralytics_version"], torch_version=run["torch_version"],
                          config_hash=run["config_hash"])
        card_text = card.read_text(encoding="utf-8") + (
            f"\n\nMetadata refresh: calibration only, rect=False, imgsz={config['infer']['imgsz']}; "
            f"ultralytics {ultralytics.__version__}, torch {torch.__version__}, device={args.device}.\n"
            f"pr_conf={overall['pr_conf']}; epochs_run={training['epochs_run']}, "
            f"best_epoch={training['best_epoch']}, train_time_s={training['train_time_s']}.\n"
            f"Training source commit: {source_commit}. Checkpoints and predictions preserved.\n")
        pending[run_dir / "run_manifest.json"] = manifest_bytes
        pending[art_dir / "artifact.json"] = json_bytes(artifact)
        pending[art_dir / "calibration_per_class.json"] = json_bytes(calibration)
        pending[art_dir / "model_card.md"] = card_text.encode("utf-8")
        summaries[model_id] = {**training, "overall": overall, "per_class": per_class,
                               "validation_context": validation_context,
                               "run_manifest_sha256": artifact["run_manifest"]["sha256"],
                               "results_csv_sha256": sha256_file(run_dir / "results.csv")}

    def verify_unchanged():
        if any(sha256_file(p) != digest for p, digest in protected.items()):
            raise ValueError("Protected checkpoint/prediction bytes changed")
        if dataset_inventory(dataset_root) != dataset_before:
            raise ValueError("Dataset inventory changed")

    verify_unchanged()
    for path, content in pending.items():
        backup = work / "before" / path.relative_to(root)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(path.read_bytes())
        temp = path.with_name(path.name + ".refresh.tmp")
        temp.write_bytes(content)
        temp.replace(path)
    verify_unchanged()
    report = {"schema_version": 1, "source_commit": source_commit,
              "bundle_sha256": sha256_file(args.bundle), "verified_bundle_files": bundle_files,
              "models": summaries, "dataset_inventory_unchanged": True,
              "protected_files_unchanged": True,
              "protected_sha256": {p.relative_to(root).as_posix(): digest for p, digest in protected.items()}}
    report_path = root / "reports/phase_c_metadata_refresh.json"
    report_path.write_bytes(json_bytes(report))
    print(f"Refreshed {len(pending)} metadata files; preserved {len(protected)} checkpoint/prediction files.")
    print(f"Evidence: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=("B01_yolo11n", "B02_yolo11s"),
                        default=["B01_yolo11n", "B02_yolo11s"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    refresh(parser.parse_args())


if __name__ == "__main__":
    main()
