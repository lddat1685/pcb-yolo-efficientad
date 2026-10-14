"""Read the registered, frozen Step 7 operating points without modifying them."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pcb_lab.data.manifest import dataset_path, resolve_dataset_root
from pcb_lab.data.samples import ManifestDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MATRIX = {
    "B01": ("YOLO", "B01_yolo11n", None),
    "B02": ("YOLO", "B02_yolo11s", None),
    "B03": ("AD", None, "B03_efficientad_s"),
    "H00": ("OR", "B01_yolo11n", "B03_efficientad_s"),
    "H01": ("CASCADE", "B01_yolo11n", "B03_efficientad_s"),
    "H02": ("CASCADE", "B02_yolo11s", "B03_efficientad_s"),
    "H01_sec": ("CASCADE", "B01_yolo11n", "B03_efficientad_s"),
}


def file_stamp(path) -> tuple:
    path = Path(path)
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    checkpoint: Path
    sha256: str

    def fingerprint(self):
        return (self.model_id, self.sha256, file_stamp(self.checkpoint),
                file_stamp(self.checkpoint.parent / "artifact.json"))


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    mode: str
    yolo: ModelAsset | None
    ad: ModelAsset | None
    tau_yolo: float | None
    tau_ad: float | None
    config_hash: str


class ModelRegistry:
    def __init__(self, repo_root=REPO_ROOT, *, frozen_config_path=None):
        self.repo_root = Path(repo_root).resolve()
        self.freeze_path = Path(frozen_config_path) if frozen_config_path else self.repo_root / "artifacts/test/frozen_evaluation.json"
        raw = self.freeze_path.read_bytes()
        self.config_hash = hashlib.sha256(raw).hexdigest()
        self.frozen = json.loads(raw.decode("utf-8-sig"))
        if self.frozen.get("status") != "FROZEN" or self.frozen.get("split_name") != "test":
            raise ValueError("Cấu hình Bước 7 chưa được khóa")
        self.specs = {}
        assets = {}
        for key, entry in self.frozen["checkpoints"].items():
            digest = entry["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Checkpoint SHA256 không hợp lệ")
            assets[key] = ModelAsset(key, dataset_path(self.repo_root, entry["path"]), digest)
        for model_id, (mode, yolo, ad) in MODEL_MATRIX.items():
            if model_id == "H01_sec" and model_id not in self.frozen["operating_points"]:
                continue
            point = self.frozen["operating_points"][model_id]
            ty, ta = point["tau_yolo"], point["tau_ad"]
            import math
            if yolo and (not isinstance(ty, (float, int)) or not math.isfinite(ty) or not 0 < ty <= 1):
                raise ValueError("Ngưỡng YOLO không hợp lệ")
            if ad and (not isinstance(ta, (float, int)) or not math.isfinite(ta)):
                raise ValueError("Ngưỡng AD không hợp lệ")
            self.specs[model_id] = ModelSpec(model_id, mode, assets.get(yolo), assets.get(ad), ty, ta, self.config_hash)

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self.specs:
            raise ValueError(f"Model chưa đăng ký: {model_id}")
        return self.specs[model_id]

    def gallery(self, partition="calibration", *, dataset_root=None, report_mode=False):
        if partition not in {"calibration", "fusion", "test"}:
            raise ValueError("Gallery chỉ hỗ trợ calibration/fusion/test")
        allow_test = partition == "test"
        if allow_test:
            if not report_mode or self.frozen.get("test_access_authorized") is not True:
                raise PermissionError("Gallery test chỉ mở trong chế độ báo cáo đã khóa")
            from .reports import load_benchmark
            load_benchmark(self.repo_root)
        samples = tuple(ManifestDataset(resolve_dataset_root(dataset_root), partition, allow_test=allow_test))
        if allow_test:
            digest = hashlib.sha256(json.dumps(sorted(s.sample_id for s in samples)).encode()).hexdigest()
            if len(samples) != self.frozen["sample_count"] or digest != self.frozen["test_split_sha256"]:
                raise ValueError("Gallery test không khớp danh sách ảnh đã khóa")
        return samples
