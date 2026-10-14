# tests/conftest.py
#
# Sở hữu: VERIFIER (step1/tests + step2/tests + step3/tests). IMPLEMENTER
# KHÔNG sửa file này.
#
# Mục đích:
#   - Thêm ../src vào sys.path để import được pcb_lab.* khi chạy pytest từ repo root.
#   - Đăng ký marker `dataset` cho các test cần dataset thật (tự skip nếu thiếu DATASET_ROOT).
#   - Import module cần test BÊN TRONG fixture/hàm để pytest vẫn collect được
#     ngay cả khi implementer chưa merge code (test đỏ ở Giai đoạn A là bình thường).
#
# Bảo vệ: không đọc src/ của impl; chỉ import tên symbol đã được contract định nghĩa
#   (pcb_lab.data.audit.run_audit, pcb_lab.data.eda.render_overlays,
#    pcb_lab.data.samples.*, pcb_lab.data.augment.*, pcb_lab.inference.tiling.*,
#    pcb_lab.inference.preprocessing.*, scripts/check_loaders.py, scripts/prep_preview.py).

import os
import sys
from pathlib import Path

import pytest

# Step 4 fixtures are verifier-owned and keep EfficientAD imports lazy.
pytest_plugins = ["helpers.step4_fixtures"]

# --- Thêm src vào sys.path -------------------------------------------------
# Repo root = thư mục chứa tests/ ; src nằm tại <root>/src.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# --- Marker `dataset` ------------------------------------------------------
# Test gắn @pytest.mark.dataset dùng dataset THẬT (DATASET_ROOT). Tự skip khi
# biến môi trường DATASET_ROOT không trỏ tới một dataset hợp lệ.
def _real_dataset_root() -> Path | None:
    from pcb_lab.data.manifest import resolve_dataset_root
    p = resolve_dataset_root()
    manifest = p / "benchmarks" / "deeppcb" / "manifests" / "samples.jsonl"
    if p.is_dir() and manifest.is_file():
        return p
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: needs torch+ultralytics; slow, CPU-capable training/reload"
    )
    config.addinivalue_line(
        "markers", "artifacts: needs STEP3_ARTIFACT_DIR pointing to a real artifact"
    )
    config.addinivalue_line(
        "markers",
        "dataset: test cần dataset thật; tự skip nếu DATASET_ROOT thiếu hoặc không hợp lệ",
    )


def pytest_collection_modifyitems(config, items):
    # Step 3 contract: absent optional inputs skip; a supplied broken artifact fails.
    for item in items:
        if item.get_closest_marker("artifacts") and not os.environ.get("STEP3_ARTIFACT_DIR"):
            item.add_marker(pytest.mark.skip(reason="STEP3_ARTIFACT_DIR is not set"))


@pytest.fixture
def step3_artifact_dir():
    value = os.environ.get("STEP3_ARTIFACT_DIR")
    if not value:
        pytest.skip("STEP3_ARTIFACT_DIR is not set")
    path = Path(value).resolve()
    assert path.is_dir(), f"STEP3_ARTIFACT_DIR does not exist: {path}"
    assert (path / "artifact.json").is_file(), f"Missing artifact.json: {path}"
    return path


@pytest.fixture
def step3_cpu_dependencies(tmp_path, monkeypatch):
    # Step 3 smoke: no GPU requirement; broken installed packages must fail visibly.
    from importlib.util import find_spec
    missing = [name for name in ("torch", "ultralytics") if find_spec(name) is None]
    if missing:
        pytest.skip("smoke dependencies missing: " + ", ".join(missing))
    # Keep library import-time settings in the test's own directory.
    settings = tmp_path / "ultralytics-settings"
    settings.mkdir()
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(settings))
    import torch
    import ultralytics
    return torch, ultralytics


@pytest.fixture(autouse=True)
def step3_no_network(request, monkeypatch):
    # Step 3 contract: no implicit downloads. Limit the guard to verifier-owned YOLO tests.
    if not request.node.path.name.startswith("test_yolo_"):
        return
    import socket
    def deny_network(*args, **kwargs):
        pytest.fail("Step 3 test attempted network access without allow_download=True")
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)


@pytest.fixture
def real_dataset_root():
    """Trả DATASET_ROOT thật, hoặc skip test nếu không có.

    Bảo vệ plan §1.1 (dataset 2.998 ảnh) và contract (dataset chỉ đọc).
    """
    root = _real_dataset_root()
    if root is None:
        pytest.skip("DATASET_ROOT không được đặt hoặc không trỏ tới dataset hợp lệ")
    return root


@pytest.fixture
def synthetic_dataset_root(tmp_path):
    """Dataset giả SẠCH trên tmp_path, dựng từ Step 1 DatasetFixture.

    Dùng cho test loader/tiling/preprocessing trên dataset giả: 4 partition,
    mỗi partition có good + defect, 6 lớp, 6 cặp KHÔNG rò rỉ. Trả về Path root.
    Bảo vệ plan §1.2 (phân chia 4 partition) và contract (schema manifest).

    KHÔNG decode/ghi ảnh test; fixture chỉ cung cấp metadata + ảnh giả hợp lệ.
    """
    from helpers.dataset_builder import build_clean_fixture
    fx = build_clean_fixture(tmp_path / "ds")
    return fx.root


# --- Import helpers: import BÊN TRONG hàm để collect được khi chưa có impl ---
def import_run_audit():
    """Import run_audit từ pcb_lab.data.audit (theo contract)."""
    from pcb_lab.data.audit import run_audit  # noqa: F401
    return run_audit


def import_render_overlays():
    """Import render_overlays từ pcb_lab.data.eda (theo contract)."""
    from pcb_lab.data.eda import render_overlays  # noqa: F401
    return render_overlays


def import_dataset_input_error():
    """Import exception bắt buộc (nếu impl định nghĩa). Trả None nếu chưa có."""
    try:
        from pcb_lab.data.manifest import DatasetInputError  # noqa: F401
        return DatasetInputError
    except Exception:
        return None


# --- Step 2 import helpers (tên symbol từ docs/handoff/step2-contract.md) ------
def import_tile_manager():
    """Import TileManager từ pcb_lab.inference.tiling (theo contract)."""
    from pcb_lab.inference.tiling import TileManager  # noqa: F401
    return TileManager


def import_preprocessing():
    """Import tiền xử lý từ pcb_lab.inference.preprocessing (theo contract).

    Trả tuple (to_canonical, letterbox, unletterbox_boxes, normalize,
    resize_for_ad, upsample_map, prepare_input). Dùng BÊN TRONG test.
    """
    from pcb_lab.inference.preprocessing import (
        to_canonical, letterbox, unletterbox_boxes, normalize,
        resize_for_ad, upsample_map, prepare_input,
    )
    return {
        "to_canonical": to_canonical,
        "letterbox": letterbox,
        "unletterbox_boxes": unletterbox_boxes,
        "normalize": normalize,
        "resize_for_ad": resize_for_ad,
        "upsample_map": upsample_map,
        "prepare_input": prepare_input,
    }


def import_samples():
    """Import loader từ pcb_lab.data.samples (theo contract).

    Trả tuple (Sample, ManifestDataset, YoloTrainSet, AdTrainSet, EvalSet,
    Box). Dùng BÊN TRONG test.
    """
    from pcb_lab.data.samples import (
        Sample, ManifestDataset, YoloTrainSet, AdTrainSet, EvalSet, Box,
    )
    return {
        "Sample": Sample,
        "ManifestDataset": ManifestDataset,
        "YoloTrainSet": YoloTrainSet,
        "AdTrainSet": AdTrainSet,
        "EvalSet": EvalSet,
        "Box": Box,
    }


def import_augment():
    """Import hàm augment từ pcb_lab.data.augment (theo contract).

    Contract chỉ định recipe aug_v1 áp cho split 'train'. Tên symbol cụ thể chưa
    pin, nên thử nhiều tên khả dĩ và trả về callable. Giai đoạn A: test đỏ là bình thường.
    """
    import importlib
    mod = importlib.import_module("pcb_lab.data.augment")
    for name in ("apply_augment", "augment", "apply_recipe", "Augmenter", "augment_image"):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError("pcb_lab.data.augment không có entry point dự kiến")

