"""YOLO training runner with locked config, deterministic online augmentation, artifacts.

Honors the Step 3 contract: refuses to run while lr0/batch/workers/augment are null;
writes nothing to DATASET_ROOT; never creates official artifacts in smoke mode; refuses
to overwrite an existing run dir (unless resume=True); records every real ultralytics
argument used plus peak memory.
"""
from __future__ import annotations

import hashlib
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from pcb_lab.data.manifest import (
    artifact_path,
    dataset_path,
    load_dataset,
    output_path,
    resolve_dataset_root,
    sha256_file,
)
from pcb_lab.models.yolo.view import build_yolo_view

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_yolo_dataset
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import ultralytics as _ultra_pkg


class ConfigError(ValueError):
    """Raised on invalid or incomplete training configuration."""


ALLOWED_CONFIG_KEYS = {
    "model_id", "recipe_id", "architecture", "pretrained",
    "train", "selection", "infer", "classes_ref",
}


def _merge_configs(paths) -> dict:
    merged = {}
    for p in paths:
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8-sig"))
        if data is None:
            data = {}  # comment-only / empty file
        if not isinstance(data, dict):
            raise ConfigError(f"Config {p} must be a mapping")
        for k in data:
            if k not in ALLOWED_CONFIG_KEYS:
                raise ConfigError(f"Unknown config key {k!r} in {p}")
        # Reject unknown keys inside known mapping sections as well (typo guard).
        _ALLOWED_SUBKEYS = {
            "train": {"imgsz", "epochs", "patience", "batch", "workers", "optimizer", "lr0", "deterministic", "cache", "plots", "cos_lr", "warmup_epochs", "augment"},
            "selection": {"metric", "frr_cap", "split"},
            "infer": {"conf_floor", "iou", "max_det", "agnostic_nms", "half", "imgsz"},
        }
        for section, allowed in _ALLOWED_SUBKEYS.items():
            if section in data:
                if not isinstance(data[section], dict):
                    raise ConfigError(f"Config section {section} must be a mapping")
                for sub in data[section]:
                    if sub not in allowed:
                        raise ConfigError(f"Unknown config key {section}.{sub!r} in {p}")
        merged.update(data)
    return merged


def _config_hash(cfg: dict) -> str:
    canonical = {k: (dict(v) if isinstance(v, dict) else v) for k, v in sorted(cfg.items()) if k != "seed"}
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _check_nulls(cfg: dict) -> None:
    t = cfg.get("train", {})
    for key in ("lr0", "batch", "workers", "augment"):
        if t.get(key) is None:
            raise ConfigError(f"train.{key} is null; the trainer refuses to run until it is set")


def _resolve_amp(amp, device):
    # AMP is only meaningful and safe on a CUDA device; on CPU it adds overhead with
    # no benefit, so it is forced off there regardless of the explicit flag.
    if amp is not None:
        return bool(amp)
    dev = (device or "cpu")
    if str(dev).lower() in ("cpu", "none", ""):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _geometric_augment(img_bgr, bboxes_norm_xywh, k, flip):
    """Apply dihedral (k*90ccw + optional fliplr) to BGR image + normalized xywh boxes.

    Operates directly in normalized xywh space (the boxes are already normalized to
    the image's own W/H, so rotation is a rigid permutation of normalized coords:
    90 deg CCW sends (cx, cy, bw, bh) -> (cy, 1 - cx, bh, bw)). Mirrors
    pcb_lab.data.augment.apply_augmentation exactly (Step 2 approved recipe).
    Returns (img_bgr, bboxes_norm_xywh) with the same column layout.
    """
    for _ in range(k):
        img_bgr = np.ascontiguousarray(np.rot90(img_bgr, k=1, axes=(0, 1)))
    if flip:
        img_bgr = np.ascontiguousarray(np.fliplr(img_bgr))
    boxes = []
    for (cx, cy, bw, bh) in (bboxes_norm_xywh if len(bboxes_norm_xywh) else np.empty((0, 4))):
        for _ in range(k):
            cx, cy, bw, bh = cy, 1.0 - cx, bh, bw  # 90 deg CCW in normalized space
        if flip:
            cx = 1.0 - cx
        boxes.append([cx, cy, bw, bh])
    out = np.array(boxes, dtype=np.float64) if boxes else np.empty((0, 4), dtype=np.float64)
    return img_bgr, out


class _AugmentedYOLODataset(YOLODataset):
    """YOLODataset that applies the deterministic geometric yolo_aug_v1 online.

    The base view is left unchanged (1.792 train images). Augmentation is per-sample,
    seeded by (epoch, index) so it is reproducible and never expands the image count.
    All photometric/mosaic hyperparams are forced to 0.0 by the runner.
    """

    def __init__(self, *args, aug_seed=0, **kwargs):
        self._aug_seed = int(aug_seed)
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        if not self.augment:
            return label
        # Seed deterministically from epoch + index; epoch advances each epoch.
        epoch = getattr(self, "_epoch", 0)
        rng = np.random.default_rng((self._aug_seed * 1_000_003 + epoch) * 1_000_003 + index)
        k = int(rng.integers(0, 4))
        flip = bool(rng.random() < 0.5)
        img = label["img"]  # BGR uint8 HWC
        instances = label["instances"]  # Instances: normalized xywh boxes
        bboxes = instances._bboxes.bboxes  # (n,4) normalized xywh
        if bboxes.size:
            img, bboxes = _geometric_augment(img, np.asarray(bboxes, dtype=np.float64), k, flip)
            instances.update(bboxes.astype(np.float32))
        else:
            if k:
                img = np.ascontiguousarray(np.rot90(img, k=k, axes=(0, 1)))
            if flip:
                img = img[:, ::-1, :].copy()
        label["img"] = img
        return label

    def _set_epoch(self, epoch):
        self._epoch = int(epoch)


class _PCBDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that injects the augmented dataset and tracks peak memory."""

    def __init__(self, cfg=..., overrides=None, _callbacks=None, aug_seed=0, peak=None):
        from ultralytics.cfg import DEFAULT_CFG
        if cfg is ...:
            cfg = DEFAULT_CFG
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._aug_seed = int(aug_seed)
        self._peak = peak  # dict with 'ram_mb': {value, method}

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(self.unwrap_stride()), 32)
        # Build the dataset directly with our augmented class when training.
        from ultralytics.data.dataset import YOLODataset as _Y
        if mode == "train":
            dataset = _AugmentedYOLODataset
        else:
            dataset = _Y
        rect = mode == "val"
        stride = gs
        return self._make_yolo_dataset(dataset, img_path, batch, mode, rect, stride)

    def unwrap_stride(self):
        from ultralytics.utils.torch_utils import unwrap_model
        return unwrap_model(self.model).stride.max()

    def _make_yolo_dataset(self, dataset_cls, img_path, batch, mode, rect, stride):
        pad = 0.0 if mode == "train" else 0.5
        rect = bool(self.args.rect or rect)
        fraction = 1.0
        augment = mode == "train"
        kwargs = dict(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=augment, hyp=self.args, rect=rect, cache=self.args.cache or None,
            single_cls=bool(self.args.single_cls or False), stride=stride, pad=pad,
            prefix=("", "val: ")[mode == "val"], task=self.args.task,
            classes=self.args.classes, data=self.data, fraction=fraction,
        )
        if dataset_cls is _AugmentedYOLODataset:
            kwargs["aug_seed"] = self._aug_seed
        return dataset_cls(**kwargs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        if mode == "train" and isinstance(dataset, _AugmentedYOLODataset):
            dataset._set_epoch(self.start_epoch if hasattr(self, "start_epoch") else 0)
        self._last_dataset = dataset
        shuffle = mode == "train"
        return self._build_dataloader(dataset, batch_size, mode, rank, shuffle)

    def _build_dataloader(self, dataset, batch_size, mode, rank, shuffle):
        from ultralytics.data.build import build_dataloader
        return build_dataloader(
            dataset, batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle, rank=rank,
            drop_last=bool(self.args.compile) and mode == "train",
            device=self.device,
        )

    def _step_epoch_hook(self):
        ds = getattr(self, "_last_dataset", None)
        if isinstance(ds, _AugmentedYOLODataset):
            ds._set_epoch(self.epoch)

    def train(self):
        # Wrap epoch advance to refresh per-epoch augmentation seed.
        super().train()

    def final_eval(self):
        # Upstream strips optimizer/epoch from last.pt during final evaluation.
        # Preserve the actual training checkpoint, including EMA and optimizer.
        backup = self.last.with_name("last_resume.pt")
        if self.last.is_file():
            shutil.copyfile(self.last, backup)
        try:
            super().final_eval()
        finally:
            if backup.is_file():
                backup.replace(self.last)


def _f1_optimal_conf(box):
    """Confidence at the maximum of the mean F1 curve (Ultralytics F1-optimal).

    ``metrics.box.f1`` is a per-class array at the optimum, not a scalar, so the
    scalar ``pr_conf`` must come from ``px`` at argmax(smooth(mean(f1_curve))).
    Mirrors ``ap_per_class`` (metrics.py): i = smooth(f1_curve.mean(0), 0.1).argmax().
    Returns None when curves are unavailable.
    """
    try:
        f1_curve = getattr(box, "f1_curve", None)
        px = getattr(box, "px", None)
        if f1_curve is None or px is None:
            return None
        f1_curve = np.asarray(f1_curve, dtype=np.float64)
        px = np.asarray(px, dtype=np.float64)
        if (f1_curve.ndim not in (1, 2) or px.ndim != 1
                or f1_curve.size == 0 or px.size == 0
                or not np.isfinite(f1_curve).all() or not np.isfinite(px).all()):
            return None
        mean_f1 = f1_curve.mean(axis=0) if f1_curve.ndim > 1 else f1_curve
        if mean_f1.size != px.size:
            return None
        mean_f1 = _smooth(mean_f1, 0.1)
        idx = int(np.argmax(mean_f1))
        value = float(px[idx])
        if not 0 <= value <= 1:
            return None
        return value
    except (TypeError, ValueError, IndexError):
        return None


def _smooth(y, f=0.05):
    """Box filter of fraction f (mirrors ultralytics.utils.metrics.smooth)."""
    from ultralytics.utils.metrics import smooth
    return smooth(np.asarray(y, dtype=np.float64), f)


def _calibration_gt_counts(view):
    """Ground-truth box counts per class from the calibration view (manifest)."""
    try:
        counts = (view.box_counts or {}).get("calibration", {})
        return {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return {}


def _read_training_summary(runs_base, planned_epochs):
    """Derive epochs_run/best_epoch/train_time_s/early_stopped from results.csv.

    Best epoch = argmax of metrics/mAP50-95(B) (Ultralytics fitness in this
    version is pure mAP50-95: Metric.fitness weights [0,0,0,1]). Missing or
    malformed evidence raises instead of reporting planned epochs as observed.
    """
    try:
        path = Path(runs_base) / "results.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = [{k.strip(): v.strip() for k, v in row.items()}
                    for row in csv.DictReader(stream)]
        if not rows:
            raise ValueError("empty results.csv")
        epochs = [int(row["epoch"]) for row in rows]
        fitness = [float(row["metrics/mAP50-95(B)"]) for row in rows]
        elapsed = float(rows[-1]["time"])
        if epochs != list(range(1, len(rows) + 1)):
            raise ValueError("epoch history must be complete and contiguous from 1")
        if not np.isfinite(fitness).all() or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("nonfinite or invalid training measurements")
        # Ultralytics saves best.pt again on an equal best fitness.
        best_index = max(range(len(rows)), key=lambda i: (fitness[i], epochs[i]))
        return {
            "epochs_run": len(rows),
            "best_epoch": epochs[best_index],
            "train_time_s": round(elapsed, 1),
            "early_stopped": len(rows) < int(planned_epochs),
        }
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Cannot derive training summary from {path}: {exc}") from exc


def _unwrap_model_stride(model):
    from ultralytics.utils.torch_utils import unwrap_model
    return int(unwrap_model(model).stride.max())


def _measure_peak_ram_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes
            class _PMEM(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            mem = _PMEM()
            mem.cb = ctypes.sizeof(_PMEM)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(mem), ctypes.sizeof(mem))
            return mem.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    return float("nan")


@dataclass
class RunResult:
    run_dir: Path
    artifact_dir: Path | None
    run_manifest: dict
    artifact: dict | None
    smoke: bool


def _git_state():
    root = Path(__file__).resolve().parents[4]
    try:
        git = shutil.which("git")
        if git is None and os.name == "nt":
            candidate = Path(os.environ.get("ProgramFiles", "")) / "Git/cmd/git.exe"
            git = str(candidate) if candidate.is_file() else "git"
        command = [git or "git", "-c", f"safe.directory={root.as_posix()}"]
        commit = subprocess.check_output(command + ["rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(command + ["status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL).decode().strip() != ""
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        provenance = json.loads((root / "bundle_provenance.json").read_text(encoding="utf-8"))
        commit, dirty = provenance["git_commit"], provenance["git_dirty"]
        if (isinstance(commit, str) and len(commit) == 40
                and all(c in "0123456789abcdef" for c in commit) and type(dirty) is bool):
            return commit, dirty
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None, None



def _release_files_sha(release):
    files = release.get("files", {}) if isinstance(release, dict) else {}
    blob = json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _class_order(dataset_root):
    return load_dataset(dataset_root).names


def _preprocessing_hash():
    from pcb_lab.inference.preprocessing import preprocessing_hash
    return preprocessing_hash()


def _download_pretrained(rel):
    p = Path(rel)
    if not p.exists():
        raise ConfigError("Pretrained not found locally and download not authorized here")
    return p


def train_yolo(config_path, seed=42, out_root=".", device=None, smoke=False,
               resume=False, allow_download=False, amp=None):
    dataset_root = resolve_dataset_root()
    out_root = output_path(dataset_root, Path(out_root))
    os.environ.setdefault("ULTRALYTICS_NO_ANALYTICS", "1")
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError("Config not found: " + str(config_path))
    common = config_path.parent / "yolo_common.yaml"
    paths = ([common] if common.exists() else []) + [config_path]
    cfg = _merge_configs(paths)
    _check_nulls(cfg)

    model_id = cfg["model_id"]
    arch = cfg["architecture"]
    recipe_id = cfg.get("recipe_id")
    train_cfg = dict(cfg["train"])
    infer_cfg = dict(cfg["infer"])
    classes_ref = cfg.get("classes_ref", "benchmarks/deeppcb/configs/classes.json")

    release = load_dataset(dataset_root).release
    dataset_release_sha = _release_files_sha(release)

    pretrained_rel = cfg.get("pretrained")
    if smoke:
        pretrained_path = None
        pretrained_sha = None
        pretrained_source = None
    else:
        if not pretrained_rel:
            raise ConfigError("pretrained must be set for a real run")
        pretrained_path = Path(pretrained_rel).resolve()
        if not pretrained_path.exists():
            if allow_download:
                pretrained_path = _download_pretrained(pretrained_path).resolve()
            else:
                raise ConfigError("Pretrained missing; pass allow_download=True to fetch")
        pretrained_sha = sha256_file(pretrained_path)
        pretrained_source = "local" if not allow_download else "ultralytics-assets(github)"

    config_hash = _config_hash(cfg)

    # Resolve every derived output directory before the first mkdir. This also
    # rejects model_id escapes and existing junctions outside the output root.
    runs_base = artifact_path(out_root, f"runs/{'smoke' if smoke else 'yolo'}/{model_id}/seed{seed}")
    view_out = artifact_path(out_root, f"data_refs/{'smoke_view' if smoke else 'yolo_view'}/{model_id}")
    artifact_dir = None if smoke else artifact_path(out_root, f"artifacts/yolo/{model_id}/seed{seed}")
    if runs_base.exists() and not resume:
        raise ConfigError("Run dir already exists; refuse overwrite (use resume=True)")
    runs_base.mkdir(parents=True, exist_ok=True)

    limit = {"good": 8, "defect": 8} if smoke else None
    view = build_yolo_view(dataset_root, view_out, partitions=("train", "calibration"),
                           link="hardlink", limit=limit)

    # Smoke = random init from architecture config (no pretrained, no download);
    # real run = pretrained checkpoint. A bare arch name would route to .pt and
    # trigger a download, so smoke uses the "<arch>.yaml" config path.
    if smoke:
        model_arg = f"{arch}.yaml"
    else:
        model_arg = str(pretrained_path) if pretrained_path else arch
    overrides = {
        "model": model_arg,
        "data": str(view.data_yaml),
        "imgsz": train_cfg["imgsz"],
        "epochs": (1 if smoke else train_cfg["epochs"]),
        "patience": train_cfg["patience"],
        "batch": train_cfg["batch"],
        "workers": train_cfg["workers"],
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "deterministic": train_cfg.get("deterministic", True),
        "seed": seed,
        "device": (device or "cpu"),
        "project": str(runs_base.parent),
        "name": runs_base.name,
        "exist_ok": True,
        "cache": train_cfg.get("cache", False),
        "plots": train_cfg.get("plots", True),
        "amp": _resolve_amp(amp, device),
        "conf": infer_cfg["conf_floor"],
        "iou": infer_cfg["iou"],
        "max_det": infer_cfg["max_det"],
        "agnostic_nms": infer_cfg["agnostic_nms"],
        "half": infer_cfg.get("half", False),
        "verbose": False,
    }
    if resume:
        resume_path = runs_base / "weights/last.pt"
        if not resume_path.is_file():
            raise ConfigError(f"Resume checkpoint missing: {resume_path}")
        overrides["resume"] = str(resume_path)
    if smoke:
        # A 16-image smoke run must actually update weights: upstream nbs=64
        # can accumulate beyond the entire tiny epoch without one optimizer step.
        overrides["nbs"] = train_cfg["batch"]
    overrides.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                    scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
                    mosaic=0.0, mixup=0.0, copy_paste=0.0, bgr=0.0,
                    close_mosaic=0, cos_lr=train_cfg.get("cos_lr", False),
                    warmup_epochs=train_cfg.get("warmup_epochs", 3.0))

    peak = {"ram_mb": {"value": None, "method": "process peak RSS (psutil; win32 psapi fallback)"},
            "vram_mb": {"value": None, "method": "torch.cuda.max_memory_allocated; null on CPU"}}

    prev_cwd = Path.cwd()
    try:
        os.chdir(runs_base)
        model = YOLO(model_arg)
        trainer = _PCBDetectionTrainer(overrides=overrides, aug_seed=seed, peak=peak)
        trainer.train()
    finally:
        os.chdir(prev_cwd)

    best_pt = runs_base / "weights" / "best.pt"
    last_pt = runs_base / "weights" / "last.pt"
    best_sha = sha256_file(best_pt) if best_pt.exists() else None
    last_sha = sha256_file(last_pt) if last_pt.exists() else None

    val_metrics, per_class = _validate(model_id, best_pt, view, infer_cfg, device)

    peak["ram_mb"]["value"] = round(_measure_peak_ram_mb(), 1)
    if device and str(device).lower() not in ("cpu", "none", ""):
        try:
            if torch.cuda.is_available():
                peak["vram_mb"]["value"] = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
        except Exception:
            peak["vram_mb"]["value"] = None

    commit, dirty = _git_state()
    training_summary = _read_training_summary(runs_base, overrides["epochs"])
    run_manifest = {
        "schema_version": 1,
        "run_id": model_id + "-seed" + str(seed),
        "model_id": model_id,
        "architecture": arch,
        "seed": seed,
        "smoke": bool(smoke),
        "config_hash": config_hash,
        "recipe_id": recipe_id,
        "git_commit": commit,
        "git_dirty": dirty,
        "ultralytics_version": _ultra_pkg.__version__,
        "torch_version": torch.__version__,
        "cuda_version": (torch.version.cuda if torch.cuda.is_available() else None),
        "device": (device or "cpu"),
        "python": platform.python_version(),
        "dataset_release_sha256": dataset_release_sha,
        "view_signature": view.view_signature,
        "link_mode": view.link_mode,
        "partitions_used": {"train": view.counts.get("train", {}).get("images"), "val": "calibration"},
        "pretrained": {"path": (str(pretrained_path) if pretrained_path else None),
                       "sha256": pretrained_sha, "source": pretrained_source},
        "args_used": overrides,
        "epochs_run": training_summary["epochs_run"],
        "best_epoch": training_summary["best_epoch"],
        "early_stopped": training_summary["early_stopped"],
        "train_time_s": training_summary["train_time_s"],
        "peak_vram_mb": peak["vram_mb"],
        "peak_ram_mb": peak["ram_mb"],
        "checkpoints": {"best_sha256": best_sha, "last_sha256": last_sha},
        "resumed_from": str(resume_path) if resume else None,
        "validation": {"overall": val_metrics, "per_class": per_class},
    }
    (runs_base / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (runs_base / "env.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": _ultra_pkg.__version__,
        "cuda_available": torch.cuda.is_available(), "device": (device or "cpu"),
    }, indent=2), encoding="utf-8")

    artifact = None
    if not smoke:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for src in (best_pt, last_pt):
            if src.exists():
                shutil.copyfile(src, artifact_dir / src.name)
        (artifact_dir / "calibration_per_class.json").write_text(json.dumps({
            "split": "calibration", "n_images": view.counts["calibration"]["images"],
            "infer_params": infer_cfg, "overall": val_metrics, "per_class": per_class,
            "source": "ultralytics validate on calibration",
            "ultralytics_version": _ultra_pkg.__version__,
        }, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        artifact = {
            "schema_version": 1, "model_id": model_id, "seed": seed, "smoke": False,
            "checkpoint_best_sha256": best_sha, "checkpoint_last_sha256": last_sha,
            "class_order": _class_order(dataset_root),
            "config_hash": config_hash,
            "preprocessing_version": "prep_v1",
            "preprocessing_hash": _preprocessing_hash(),
            "inference_params": infer_cfg,
            "view_signature": view.view_signature,
            "run_manifest": {"path": str((runs_base / "run_manifest.json").relative_to(out_root)),
                             "sha256": sha256_file(runs_base / "run_manifest.json")},
            "created_by_git_commit": commit,
        }
        (artifact_dir / "artifact.json").write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        _write_model_card(artifact_dir / "model_card.md", model_id, arch, cfg, view, val_metrics, per_class,
                          ultralytics_version=_ultra_pkg.__version__, torch_version=torch.__version__,
                          config_hash=config_hash)
        _link_preds(out_root, artifact_dir, model_id, seed)
    return RunResult(runs_base, artifact_dir, run_manifest, artifact, smoke)


def _validate(model_id, best_pt, view, infer_cfg, device, *, save_dir=None, batch=16):
    from ultralytics import YOLO
    if not best_pt or not best_pt.exists():
        return {"note": "no best.pt"}, {}
    model = YOLO(str(best_pt))
    save_dir = Path(save_dir) if save_dir is not None else view.data_yaml.parent / "validation" / model_id
    metrics = model.val(
        data=str(view.data_yaml), split="val",
        imgsz=infer_cfg["imgsz"], conf=infer_cfg["conf_floor"], iou=infer_cfg["iou"],
        max_det=infer_cfg["max_det"], agnostic_nms=infer_cfg["agnostic_nms"],
        device=(device or "cpu"), verbose=False, plots=False, rect=False,
        half=infer_cfg.get("half", False), workers=0, batch=batch,
        project=str(save_dir.parent), name=save_dir.name, exist_ok=True,
    )
    names = model.names
    overall = {
        "map50": _safe(metrics.box.map50),
        "map50_95": _safe(metrics.box.map),
        "precision": _safe(metrics.box.mp),
        "recall": _safe(metrics.box.mr),
        "pr_definition": "Ultralytics F1-optimal confidence (not operating threshold)",
        "pr_conf": _f1_optimal_conf(metrics.box),
    }
    gt_counts = _calibration_gt_counts(view)
    per_class = {}
    ap_class = getattr(metrics.box, "ap_class_index", None)
    if ap_class is not None:
        ap50 = list(metrics.box.ap50)
        ap = list(metrics.box.ap)
        p = list(metrics.box.p)
        r = list(metrics.box.r)
        for i, cid in enumerate(ap_class):
            name = names[cid] if cid < len(names) else str(cid)
            n_gt = gt_counts.get(name)
            if n_gt is None:
                nt_per_class = getattr(metrics, "nt_per_class", None)
                if nt_per_class is not None and cid < len(nt_per_class):
                    n_gt = int(nt_per_class[cid])
            per_class[name] = {
                "n_gt_boxes": n_gt,
                "precision": _safe(p[i]) if i < len(p) else None,
                "recall": _safe(r[i]) if i < len(r) else None,
                "ap50": _safe(ap50[i]) if i < len(ap50) else None,
                "ap50_95": _safe(ap[i]) if i < len(ap) else None,
            }
    return overall, per_class


def _safe(v):
    try:
        import numbers
        if isinstance(v, numbers.Number) and not isinstance(v, bool):
            return float(v)
    except Exception:
        pass
    return None


def _write_model_card(path, model_id, arch, cfg, view, overall, per_class,
                      ultralytics_version=None, torch_version=None, config_hash=None):
    lines = []
    lines.append("Model card — " + model_id)
    lines.append("")
    lines.append("Muc dich")
    lines.append("Baseline phat hien 6 loai loi PCB (YOLO detect) dung kien truc " + arch + ".")
    lines.append("Day la baseline Buoc 3 cua do an; chi neu ho model da dung, khong so sanh voi model khac.")
    lines.append("")
    lines.append("Du lieu dung")
    lines.append("Train: " + str(view.counts["train"]["images"]) + " anh (good " +
                 str(view.counts["train"]["good"]) + " / defect " + str(view.counts["train"]["defect"]) + ").")
    lines.append("Chon checkpoint bang validation tren calibration: " +
                 str(view.counts["calibration"]["images"]) + " anh.")
    lines.append("Khong dung tap test de chon epoch hay nguong.")
    lines.append("")
    lines.append("Cau hinh")
    lines.append("optimizer=" + str(cfg["train"]["optimizer"]) + ", lr0=" + str(cfg["train"]["lr0"]) +
                 ", batch=" + str(cfg["train"]["batch"]) + ", imgsz=" + str(cfg["train"]["imgsz"]) +
                 ", epochs=" + str(cfg["train"]["epochs"]) + ", patience=" + str(cfg["train"]["patience"]) + ".")
    lines.append("Augmentation: yolo_aug_v1 (hinh hoc thuan: xoay 90 x k + lat ngang; tat moi photometric/mosaic).")
    lines.append("Infer: conf_floor=" + str(cfg["infer"]["conf_floor"]) + ", iou=" + str(cfg["infer"]["iou"]) +
                 ", max_det=" + str(cfg["infer"]["max_det"]) + ".")
    lines.append("")
    lines.append("Phien ban")
    ultra_v = ultralytics_version if ultralytics_version is not None else cfg.get("ultralytics_version", "n/a")
    torch_v = torch_version if torch_version is not None else cfg.get("torch_version", "n/a")
    cfg_hash = config_hash if config_hash is not None else cfg.get("config_hash", "n/a")
    lines.append("ultralytics " + str(ultra_v) + ", torch " +
                 str(torch_v) + ".")
    lines.append("config_hash=" + str(cfg_hash) + ", view_signature=" + view.view_signature + ".")
    lines.append("")
    lines.append("So lieu calibration (F1-optimal confidence, KHONG phai nguong van hanh)")
    lines.append("mAP50=" + str(overall.get("map50")) + ", mAP50-95=" + str(overall.get("map50_95")) +
                 ", precision=" + str(overall.get("precision")) + ", recall=" + str(overall.get("recall")) + ".")
    lines.append("")
    lines.append("Theo lop")
    for name in sorted(per_class.keys()):
        m = per_class[name]
        lines.append(name + ": ap50=" + str(m.get("ap50")) + ", ap50_95=" + str(m.get("ap50_95")) +
                     ", precision=" + str(m.get("precision")) + ", recall=" + str(m.get("recall")) + ".")
    lines.append("")
    lines.append("Gioi han")
    lines.append("Dataset chi co N=2 nhom nguon o moi partition (calibration/fusion/test);")
    lines.append("so anh khong dong nghia so quan sat vat ly doc lap.")
    lines.append("Khong co pixel mask chuan nen khong bao pixel AUROC/Dice/IoU segmentation.")
    lines.append("Chua chay tren test; khong dung test chon cau hinh.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _link_preds(out_root, artifact_dir, model_id, seed):
    preds_src = out_root / "artifacts" / "yolo" / model_id / ("seed" + str(seed))
    for part in ("calibration", "fusion"):
        src = preds_src / ("preds_" + part + ".jsonl")
        if src.exists():
            shutil.copyfile(src, artifact_dir / ("preds_" + part + ".jsonl"))
