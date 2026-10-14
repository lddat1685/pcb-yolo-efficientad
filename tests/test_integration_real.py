# tests/test_integration_real.py
#
# VERIFIER - Giai đoạn A.6: test tích hợp trên dataset THẬT (marker dataset).
#
# Số kỳ vọng là ORACLE từ plan §1.2 (KHÔNG lấy từ code impl):
#   - YoloTrainSet train = 1.792 (895 good + 897 defect).
#   - AdTrainSet 895 ảnh đều good; num_tiles = 895 * 9 = 8.055.
#   - calibration 460; fusion 306; development_selection == fusion.
#   - test 440 (220 good + 220 defect), sample_id duy nhất; load_image -> PermissionError.
#   - check_loaders.py chạy 2 lần -> reports/loader_check.json GIỐNG HỆT.
#   - 3 ảnh train đọc bằng Pillow độc lập, so với to_canonical.
#
# KHÔNG decode/xem ảnh test (metadata-only cho test partition). Tự skip nếu thiếu DATASET_ROOT.

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import import_samples, import_preprocessing, import_tile_manager

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Oracle plan §1.2 (không lấy từ code impl) ---
ORACLE = {
    "yolo_train": 1792,
    "yolo_train_good": 895,
    "yolo_train_defect": 897,
    "ad_train_good": 895,
    "calibration": 460,
    "fusion": 306,
    "test_total": 440,
    "test_good": 220,
    "test_defect": 220,
    "tiles_per_image": 9,            # TileManager.plan(640,640) -> 9 (contract)
    "ad_train_num_tiles": 895 * 9,   # = 8055
}


@pytest.mark.dataset
def test_yolo_train_set_counts(real_dataset_root):
    """Bảo vệ plan §1.2: YoloTrainSet train = 1.792 (895 good + 897 defect)."""
    S = import_samples()
    ds = S["YoloTrainSet"](real_dataset_root)
    assert len(ds) == ORACLE["yolo_train"], f"train phải 1792, got {len(ds)}"
    good = sum(1 for s in ds if not s.is_defect)
    defect = sum(1 for s in ds if s.is_defect)
    assert good == ORACLE["yolo_train_good"], f"train good phải 895, got {good}"
    assert defect == ORACLE["yolo_train_defect"], f"train defect phải 897, got {defect}"


@pytest.mark.dataset
def test_ad_train_set_good_only_and_num_tiles(real_dataset_root):
    """Bảo vệ plan §5 Bước 2: AdTrainSet chỉ good train (895); num_tiles = 895*9 = 8055."""
    S = import_samples()
    ds = S["AdTrainSet"](real_dataset_root)
    assert len(ds) == ORACLE["ad_train_good"], f"AdTrainSet good phải 895, got {len(ds)}"
    assert all(not s.is_defect for s in ds), "AdTrainSet chứa defect?"
    tm = import_tile_manager()()
    assert ds.tiles_per_image == len(tm.plan(640, 640)), "tiles_per_image lấy từ TileManager"
    assert ds.num_tiles == ORACLE["ad_train_num_tiles"], f"num_tiles phải 8055, got {ds.num_tiles}"


@pytest.mark.dataset
def test_calibration_count(real_dataset_root):
    """Bảo vệ plan §1.2: calibration 460."""
    S = import_samples()
    ds = S["EvalSet"](real_dataset_root, "calibration")
    assert len(ds) == ORACLE["calibration"], f"calibration phải 460, got {len(ds)}"


@pytest.mark.dataset
def test_fusion_equals_development_selection(real_dataset_root):
    """Bảo vệ plan §1.2 + contract: fusion 306; development_selection == fusion."""
    S = import_samples()
    fus = S["EvalSet"](real_dataset_root, "fusion")
    dev = S["EvalSet"](real_dataset_root, "development_selection")
    assert len(fus) == ORACLE["fusion"], f"fusion phải 306, got {len(fus)}"
    assert [s.sample_id for s in fus] == [s.sample_id for s in dev], "dev_selection phải == fusion"


@pytest.mark.dataset
def test_test_partition_unique_and_forbidden(real_dataset_root):
    """Bảo vệ plan §1.2 + contract: test 440 (220 good + 220 defect), sample_id duy nhất,
    load_image() -> PermissionError."""
    S = import_samples()
    ds = S["ManifestDataset"](real_dataset_root, "test", allow_test=False)
    assert len(ds) == ORACLE["test_total"], f"test phải 440, got {len(ds)}"
    ids = [s.sample_id for s in ds]
    assert len(ids) == len(set(ids)), "sample_id test phải duy nhất (không trùng)"
    good = sum(1 for s in ds if not s.is_defect)
    defect = sum(1 for s in ds if s.is_defect)
    assert good == ORACLE["test_good"] and defect == ORACLE["test_defect"], \
        f"test good/defect phải 220/220, got {good}/{defect}"
    with pytest.raises(PermissionError):
        ds[0].load_image()


@pytest.mark.dataset
def test_check_loaders_deterministic(real_dataset_root, tmp_path):
    """Bảo vệ plan §5 Bước 2: check_loaders.py 2 lần -> reports/loader_check.json GIỐNG HỆT."""
    script = REPO_ROOT / "scripts" / "check_loaders.py"
    if not script.exists():
        pytest.skip("impl chưa merge (check_loaders.py chưa có)")
    out1 = tmp_path / "r1" / "loader_check.json"
    out2 = tmp_path / "r2" / "loader_check.json"
    import os
    env = dict(os.environ)  # B evidence: same invalid sys.environ call as preprocessing CLI test.
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for out in (out1, out2):
        proc = subprocess.run(
            [sys.executable, str(script), "--dataset-root", str(real_dataset_root),
             "--out", str(out)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
        )
        assert proc.returncode == 0, f"check_loaders thất bại: {proc.stderr}"
    a = json.loads(out1.read_text(encoding="utf-8"))
    b = json.loads(out2.read_text(encoding="utf-8"))
    assert a == b, "loader_check.json phải tất định (2 lần giống hệt)"


@pytest.mark.dataset
def test_to_canonical_matches_pillow_independent(real_dataset_root):
    """Bảo vệ contract: to_canonical trùng với đọc Pillow độc lập trên 3 ảnh train.

    Đọc 3 sample_id đầu của train qua Sample.image_path, decode bằng PIL trực tiếp,
    so sánh với to_canonical(Sample.load_image()) (RGB uint8 HWC). Không dùng ảnh test.
    """
    S = import_samples()
    P = import_preprocessing()
    ds = S["YoloTrainSet"](real_dataset_root)
    sample_ids = [s.sample_id for s in ds][:3]
    samples = {s.sample_id: s for s in ds}
    for sid in sample_ids:
        s = samples[sid]
        with Image.open(s.image_path) as im:
            im = im.convert("RGB")
            pil_arr = np.asarray(im)
        impl_arr = P["to_canonical"](s.load_image())
        assert impl_arr.shape == pil_arr.shape, f"{sid}: shape lệch"
        assert np.array_equal(impl_arr, pil_arr), \
            f"{sid}: to_canonical phải trùng đọc PIL độc lập"
