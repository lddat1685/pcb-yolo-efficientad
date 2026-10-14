"""Auditable normal-only EfficientAD-S training and artifact export."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import platform
import random
import shutil
import subprocess
import threading
import time

import numpy as np
import psutil
import torch

from pcb_lab.data.manifest import METADATA_PATHS, artifact_path, output_path, resolve_dataset_root, sha256_file
from pcb_lab.inference.preprocessing import load_preprocessing_config
from .adapter import EfficientAdDetector, EfficientAdEngine
from .architecture import EfficientAdModel, load_teacher
from .config import ConfigError, config_hash, load_config
from .dataset import EfficientAdDataset, require_good

REPO_ROOT = Path(__file__).resolve().parents[4]
TILE_PARAMS = dict(tile_size=256, stride=224, pad_mode="reflect", stitch_method="linear", nms_iou=0.45)


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    artifact_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    config_hash: str
    checkpoint_sha256: str
    smoke: bool


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _git():
    git = shutil.which("git")
    if not git and os.environ.get("ProgramFiles"):
        candidate = Path(os.environ["ProgramFiles"]) / "Git/cmd/git.exe"
        if candidate.is_file():
            git = str(candidate)
    def read(*args):
        return subprocess.check_output([git, *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return read("rev-parse", "HEAD"), bool(read("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError, TypeError):
        return None, None


class _Memory:
    def __enter__(self):
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        def sample():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


@torch.no_grad()
def teacher_statistics(model, dataset, device):
    for sample in dataset.samples:
        require_good(sample, "train")
    sums = squares = None
    count = 0
    for sample, tile in dataset.iter_tiles():
        require_good(sample, "train")
        values = model.teacher_features(tile.to(device)).double()
        batch_sum = values.sum(dim=(0, 2, 3), keepdim=True)
        batch_square = values.square().sum(dim=(0, 2, 3), keepdim=True)
        sums = batch_sum if sums is None else sums + batch_sum
        squares = batch_square if squares is None else squares + batch_square
        count += values.shape[0] * values.shape[2] * values.shape[3]
    if not count:
        raise ValueError("No train good pixels for teacher statistics")
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    model.teacher_mean.copy_(mean.float())
    model.teacher_std.copy_(std.float())
    return {"source_split": "train", "n_good_images": len(dataset), "n_tiles": dataset.num_tiles,
            "n_feature_pixels_per_channel": count, "std_floor": 1e-6,
            "sample_ids": [s.sample_id for s in dataset.samples]}


@torch.no_grad()
def map_statistics(model, dataset, device, probabilities):
    for sample in dataset.samples:
        require_good(sample, "calibration")
    model.eval()
    values = []
    for sample, tile in dataset.iter_tiles():
        require_good(sample, "calibration")
        values.append(model.raw_maps(tile.to(device))[0].cpu().numpy().reshape(2, -1))
    if not values:
        raise ValueError("No calibration good images for map normalization")
    # 230 * 9 * 56 * 56 pixels per branch; exact NumPy quantiles, no random subsampling.
    branches = np.concatenate(values, axis=1)
    quantiles = np.quantile(branches, probabilities, axis=1).T
    if not np.isfinite(branches).all() or not np.isfinite(quantiles).all():
        raise ValueError("Nonfinite calibration maps")
    model.map_quantiles.copy_(torch.from_numpy(quantiles).to(device=device, dtype=torch.float32))
    combined = branches.mean(axis=0, dtype=np.float64)
    combined_q = np.quantile(combined, probabilities)
    return {"source_split": "calibration", "n_good_images": len(dataset),
            "quantile_min": float(combined_q[0]), "quantile_max": float(combined_q[1]),
            "mean": float(combined.mean()), "std": float(combined.std()),
            "quantile_probabilities": list(probabilities), "branch_quantiles": quantiles.tolist(),
            "n_tiles": dataset.num_tiles, "n_map_pixels_per_branch": branches.shape[1],
            "statistic_domain": "raw_56x56_tile_maps_before_padding",
            "normalization_formula": "mean(0.1*(branch-q_min)/max(q_max-q_min,1e-12))",
            "sample_ids": [s.sample_id for s in dataset.samples]}


def _distribution(values):
    if not values:
        return {key: None for key in ("min", "max", "mean", "std", "p50", "p90", "p95", "p99")}
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std()), **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)}}


def calibration_report(detector, dataset, val_limit=None):
    scores = {False: [], True: []}
    samples = dataset.samples
    if val_limit is not None and val_limit > 0:
        good = [s for s in samples if not s.is_defect][:val_limit]
        defect = [s for s in samples if s.is_defect][:val_limit]
        samples = good + defect
    for sample in samples:
        score = detector.predict(sample.load_image(), skip_boxes=True).anomaly_score
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite calibration score: {sample.sample_id}")
        scores[sample.is_defect].append(score)
    return {"split": "calibration", "n_good": len(scores[False]), "n_defect": len(scores[True]),
            "good_scores": _distribution(scores[False]), "defect_scores": _distribution(scores[True]),
            "source": "frozen_checkpoint_max_stitched_map; descriptive only; no auto-threshold"}


def train_efficientad(config_path, seed=42, out_root=".", device=None, smoke=False,
                      resume=False, allow_download=False, max_steps=70000, val_limit=None) -> RunResult:
    for flag, value in (("smoke", smoke), ("resume", resume), ("allow_download", allow_download)):
        if type(value) is not bool:
            raise ConfigError(f"{flag} must be boolean")
    cfg = load_config(config_path, seed=seed, device=device, smoke=smoke)
    digest = config_hash(cfg)
    root = resolve_dataset_root()
    base = output_path(root, Path(out_root))
    run_dir = output_path(root, base / "runs" / ("smoke" if smoke else "efficientad") / cfg["model_id"] / f"seed{seed}")
    export_dir = run_dir / "artifact" if smoke else output_path(root, base / "artifacts" / "efficientad" / cfg["model_id"] / f"seed{seed}")
    last_path = run_dir / "checkpoints" / "last.pt"
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run exists: {run_dir}; use resume=True")
    if not smoke and export_dir.exists() and not resume:
        raise FileExistsError(f"Artifact exists: {export_dir}")
    if resume and not last_path.is_file():
        raise FileNotFoundError("resume=True requires checkpoints/last.pt")
    train_set = EfficientAdDataset(root, "train", {"good": 4} if smoke else None)
    calibration_good = EfficientAdDataset(root, "calibration", {"good": 4} if smoke else None)
    if not smoke:
        calibration_good.samples = tuple(s for s in calibration_good.samples if not s.is_defect)
        calibration_good.num_tiles = sum(len(calibration_good.tile_manager.plan(s.height, s.width)) for s in calibration_good.samples)
    calibration = EfficientAdDataset(root, "calibration", {"good": 4, "defect": 4} if smoke else None)
    for sample in calibration_good.samples:
        require_good(sample, "calibration")
    if not len(train_set) or not len(calibration_good):
        raise ValueError("Train/calibration good partitions must not be empty")
    release_hash = sha256_file(root / METADATA_PATHS["release"])
    selected_device = torch.device(cfg["train"]["device"])
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if selected_device.type == "cuda":
        torch.cuda.set_device(selected_device)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.cuda.reset_peak_memory_stats(selected_device)
        except Exception:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = EfficientAdModel().to(selected_device)
    optimizer = torch.optim.AdamW(list(model.student.parameters()) + list(model.autoencoder.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    teacher_path = Path(cfg["pretrained_teacher"])
    if not teacher_path.is_absolute():
        teacher_path = REPO_ROOT / teacher_path
    if not teacher_path.exists():
        output_path(root, teacher_path.parent)
    start_epoch = steps = 0
    training_seconds = 0.0
    teacher_audit = None
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        for key, expected in (("config_hash", digest), ("dataset_release_sha256", release_hash),
                              ("seed", seed), ("smoke", smoke)):
            if saved.get(key) != expected:
                raise ConfigError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if selected_device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start_epoch, steps = saved["epochs_run"], saved["optimizer_steps"]
        training_seconds = saved["train_time_s"]
        teacher_audit, teacher_source = saved["teacher_statistics"], saved["pretrained_teacher"]
    else:
        teacher_source = load_teacher(model, teacher_path, allow_download=allow_download)
    # All guards above precede run-directory writes.
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    commit, dirty = _git()
    run_id = f"{cfg['model_id']}_seed{seed}_{digest[:12]}" + ("_smoke" if smoke else "")
    log_path = artifact_path(run_dir, "train.log")
    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    log(f"normal_only_verified=true train_good={len(train_set)} train_defect=0 tiles={train_set.num_tiles}")
    log(f"map_norm_source=calibration good={len(calibration_good)} defect=0; test_access=false")
    _json(run_dir / "effective_config.json", cfg)
    _json(run_dir / "normal_only_audit.json", {"train_sample_ids": [s.sample_id for s in train_set.samples],
          "calibration_good_sample_ids": [s.sample_id for s in calibration_good.samples],
          "train_defect_count": 0, "test_access": False})
    with _Memory() as memory:
        if teacher_audit is None:
            log("Computing teacher statistics from train good")
            teacher_audit = teacher_statistics(model, train_set, selected_device)
        actual_epochs_run = cfg["train"]["epochs"]
        for epoch in range(start_epoch, cfg["train"]["epochs"]):
            began = time.perf_counter()
            model.train()
            loss_sum = 0.0
            epoch_steps = 0
            order = np.random.default_rng(seed + epoch).permutation(len(train_set))
            for index in order:
                sample = train_set.samples[index]
                require_good(sample, "train")
                for tile in train_set[index]:
                    require_good(sample, "train")
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(tile.unsqueeze(0).to(selected_device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at {sample.sample_id}, epoch {epoch}")
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    steps += 1
                    epoch_steps += 1
                    if max_steps is not None and steps >= max_steps:
                        break
                if max_steps is not None and steps >= max_steps:
                    break
            training_seconds += time.perf_counter() - began
            _save(last_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if selected_device.type == "cuda" else [],
                "epochs_run": epoch + 1, "optimizer_steps": steps, "train_time_s": training_seconds,
                "config_hash": digest, "dataset_release_sha256": release_hash, "seed": seed, "smoke": smoke,
                "teacher_statistics": teacher_audit, "pretrained_teacher": teacher_source})
            log(f"epoch={epoch+1} steps={epoch_steps} mean_loss={loss_sum / max(1, epoch_steps):.8f}")
            if max_steps is not None and steps >= max_steps:
                log(f"Reached max_steps={max_steps} (total optimizer_steps={steps}) at epoch={epoch+1}. Stopping training.")
                actual_epochs_run = epoch + 1
                break
        log("Computing map normalization from calibration good only")
        norm = map_statistics(model, calibration_good, selected_device,
                              (cfg["normalization"]["quantile_min"], cfg["normalization"]["quantile_max"]))
        meta = {"schema_version": 1, "run_id": run_id, "model_id": cfg["model_id"], "seed": seed, "smoke": smoke,
                "config_hash": digest, "preprocessing_version": load_preprocessing_config()["preprocessing_version"],
                "tile_params": TILE_PARAMS, "normalization_stats": norm, "inference_params": cfg["infer"],
                "created_by_git_commit": commit}
        checkpoint = artifact_path(run_dir, "checkpoints/model.pt")
        payload = dict(meta, model_state=model.state_dict(), teacher_statistics=teacher_audit,
                       pretrained_teacher=teacher_source, effective_config=cfg, optimizer_steps=steps)
        _save(checkpoint, payload)
        meta["checkpoint_sha256"] = sha256_file(checkpoint)
        detector = EfficientAdDetector(EfficientAdEngine(model, selected_device), meta)
        log("Reporting calibration good/defect scores with frozen normalization")
        report = calibration_report(detector, calibration, val_limit=val_limit)
        # Reload uses the serialized weights, not the pretrained input checkpoint.
        reloaded = EfficientAdModel().to(selected_device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"])
        second = EfficientAdDetector(EfficientAdEngine(reloaded, selected_device), meta)
        probe = calibration_good.samples[0].load_image()
        before, after = detector.predict(probe), second.predict(probe)
        map_delta = float(np.max(np.abs(before.anomaly_map - after.anomaly_map)))
        score_delta = abs(before.anomaly_score - after.anomaly_score)
        if map_delta > 1e-5 or score_delta > 1e-5:
            raise ValueError("Checkpoint reload consistency failed")
        log(f"reload_map_max_abs={map_delta} reload_score_abs={score_delta}")
    peak_vram_mb = 0.0
    if selected_device.type == "cuda":
        try:
            peak_vram_mb = float(torch.cuda.max_memory_allocated(selected_device) / 2**20)
        except Exception:
            try:
                peak_vram_mb = float(torch.cuda.max_memory_allocated() / 2**20)
            except Exception:
                peak_vram_mb = 0.0
    manifest = {"schema_version": 1, "run_id": run_id, "model_id": cfg["model_id"],
        "architecture": cfg["architecture"], "seed": seed, "smoke": smoke, "config_hash": digest,
        "recipe_id": cfg["recipe_id"], "git_commit": commit, "git_dirty": dirty, "anomalib_version": None,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "device": str(selected_device), "python": platform.python_version(), "dataset_release_sha256": release_hash,
        "partitions_used": {"train": len(train_set), "val": "calibration", "n_good": report["n_good"], "n_defect": report["n_defect"]},
        "normal_only_verified": True, "args_used": cfg, "epochs_run": actual_epochs_run,
        "optimizer_steps": steps, "train_time_s": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": memory.peak / 2**20, "peak_ram_method": "RSS sampled every 50ms",
        "checkpoint_sha256": meta["checkpoint_sha256"], "teacher_statistics": teacher_audit,
        "pretrained_teacher": teacher_source, "imagenette_penalty": False, "augmentation": "none",
        "auto_threshold": False, "reload_map_max_abs": map_delta, "reload_score_abs": score_delta,
        "smoke_limit": {"good": 4} if smoke else None, "test_access": False}
    manifest_path = artifact_path(run_dir, "run_manifest.json")
    _json(manifest_path, manifest)
    _json(run_dir / "env.json", {"python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": str(selected_device), "torch_num_threads": torch.get_num_threads()})
    meta["run_manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, artifact_path(export_dir, "model.pt"))
    _json(artifact_path(export_dir, "artifact.json"), meta)
    _json(artifact_path(export_dir, "calibration_stats.json"), report)
    card = (f"# B03 EfficientAD-S{' — SMOKE, không dùng benchmark' if smoke else ''}\n\n"
        f"Mục đích: baseline phát hiện bất thường PCB. Teacher PDN-S pretrained frozen, student 768 kênh và autoencoder.\n\n"
        f"Dữ liệu: {len(train_set)} good train, {train_set.num_tiles} tile, 0 defect; {steps} optimizer step. "
        "Tile 256x256, stride 224, reflect và linear blending; score=max map đã stitch.\n\n"
        f"Teacher stats từ train good; map norm từ {len(calibration_good)} calibration good. "
        "Calibration defect chỉ dùng báo cáo sau khi khóa normalization. Không truy cập pixel test.\n\n"
        f"Phân phối score calibration:\n\n```json\n{json.dumps(report, ensure_ascii=False, indent=2)}\n```\n\n"
        "Recipe standalone: AdamW, không ImageNette penalty, không augmentation, không auto-threshold; "
        "không khẳng định tái lập điểm số bài báo. Tau_pixel chỉ tạo candidate box, chưa có tau_ad quyết định ảnh.\n\n"
        "Giới hạn: không phân loại sáu lớp lỗi (anomaly_unclassified); N=2 nhóm nguồn test, dedicated line; "
        "chưa chứng minh domain shift. Smoke chỉ kiểm pipeline, không đo chất lượng B03.\n")
    artifact_path(export_dir, "model_card.md").write_text(card, encoding="utf-8")
    validation_path = (run_dir / "validation.md" if smoke else
        output_path(root, base / "reports") / f"efficientad_{cfg['model_id']}_seed{seed}_validation.md")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(card + f"\nReload map max abs={map_delta}, score abs={score_delta}.\n", encoding="utf-8")
    log(f"Completed: {export_dir}")
    return RunResult(run_dir, export_dir, export_dir / "model.pt", manifest_path, digest, meta["checkpoint_sha256"], smoke)
