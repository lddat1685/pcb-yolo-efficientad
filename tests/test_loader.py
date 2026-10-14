# tests/test_loader.py
#
# VERIFIER - Giai đoạn A.4: test loader trên dataset giả (tái dùng Step 1 DatasetFixture).
#
# Bảo vệ contract (docs/handoff/step2-contract.md) và plan §1.2, §5 Bước 2:
#   - Số lượng: YoloTrainSet=train(good+defect); AdTrainSet=chỉ good train; EvalSet mỗi ảnh 1 lần.
#   - Gating allow_test: len() + metadata OK, load_image() -> PermissionError (test partition).
#   - development_selection == fusion (cùng mẫu, cùng thứ tự).
#   - Thứ tự tất định (sort sample_id); hai lần duyệt cho cùng thứ tự.
#   - AdTrainSet.iter_tiles(): mọi tile thuộc partition train, khớp partition ảnh nguồn;
#     num_tiles = số ảnh * tiles_per_image.
#
# KHÔNG decode/việc ảnh test; loader test chỉ dùng metadata. Import BÊN TRONG test.

import numpy as np
import pytest

from conftest import import_samples


def _sample_ids(loader):
    ids = []
    for s in loader:
        ids.append(s.sample_id)
    return ids


def _partitions(loader):
    return [s.partition for s in loader]


# ===========================================================================
# Số lượng
# ===========================================================================
def test_yolo_train_set_counts(synthetic_dataset_root):
    """Bảo vệ plan §5 Bước 2: YoloTrainSet = toàn bộ train (good + defect)."""
    S = import_samples()
    ds = S["YoloTrainSet"](synthetic_dataset_root)
    # synthetic fixture: mỗi partition 1 good + 2 defect => train có 3 sample
    md = S["ManifestDataset"](synthetic_dataset_root, "train")
    assert len(ds) == len(md), f"YoloTrainSet phải = train (good+defect): {len(ds)} vs {len(md)}"
    assert len(ds) == 3, "fixture: train phải 3 sample (1 good + 2 defect)"


def test_ad_train_set_only_good_train(synthetic_dataset_root):
    """Bảo vệ plan §5 Bước 2: AdTrainSet chỉ chứa ảnh good của train."""
    S = import_samples()
    ds = S["AdTrainSet"](synthetic_dataset_root)
    md = S["ManifestDataset"](synthetic_dataset_root, "train")
    good_train = [s for s in md if not s.is_defect]
    assert len(ds) == len(good_train), "AdTrainSet phải = good train"
    assert all(not s.is_defect for s in ds), "AdTrainSet chứa defect?"
    assert len(ds) == 1, "fixture: train good = 1"


def test_eval_set_each_image_once(synthetic_dataset_root):
    """Bảo vệ contract: EvalSet mỗi ảnh đúng một lần."""
    S = import_samples()
    ds = S["EvalSet"](synthetic_dataset_root, "calibration")
    ids = _sample_ids(ds)
    assert len(ids) == len(set(ids)), "EvalSet phải mỗi ảnh 1 lần (không lặp)"
    assert len(ds) == 3, "fixture: calibration = 3 sample"


# ===========================================================================
# Gating allow_test (test partition)
# ===========================================================================
def test_test_partition_len_and_metadata_ok(synthetic_dataset_root):
    """Bảo vệ contract: test + allow_test=False -> len() và metadata được phép."""
    S = import_samples()
    ds = S["ManifestDataset"](synthetic_dataset_root, "test", allow_test=False)
    assert len(ds) == 3, "len() test phải OK (không decode)"
    s = ds[0]
    # metadata truy cập được mà không mở ảnh
    assert isinstance(s.sample_id, str) and s.sample_id
    assert s.partition == "test"
    assert isinstance(s.is_defect, bool)


def test_test_partition_load_image_forbidden(synthetic_dataset_root):
    """Bảo vệ contract + plan §5 Bước 2: test + allow_test=False -> load_image() PermissionError."""
    S = import_samples()
    ds = S["ManifestDataset"](synthetic_dataset_root, "test", allow_test=False)
    s = ds[0]
    with pytest.raises(PermissionError):
        s.load_image()
    # kể cả truy cập ảnh trực tiếp cũng phải bị cấm
    with pytest.raises(PermissionError):
        s.load_image()


def test_test_partition_allow_test_loads(synthetic_dataset_root):
    """Bảo vệ contract: cho phép rõ ràng (allow_test=True) thì load_image() hoạt động."""
    S = import_samples()
    ds = S["ManifestDataset"](synthetic_dataset_root, "test", allow_test=True)
    s = ds[0]
    arr = s.load_image()
    assert isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3


# ===========================================================================
# development_selection == fusion
# ===========================================================================
def test_development_selection_equals_fusion(synthetic_dataset_root):
    """Bảo vệ contract: 'development_selection' là alias của 'fusion' (cùng mẫu, cùng thứ tự)."""
    S = import_samples()
    fus = S["EvalSet"](synthetic_dataset_root, "fusion")
    dev = S["EvalSet"](synthetic_dataset_root, "development_selection")
    assert _sample_ids(fus) == _sample_ids(dev), "development_selection phải == fusion (cùng thứ tự)"
    assert _partitions(fus) == _partitions(dev)


# ===========================================================================
# Thứ tự tất định
# ===========================================================================
def test_order_deterministic_two_iterations(synthetic_dataset_root):
    """Bảo vệ contract: thứ tự tất định; hai lần duyệt cho cùng thứ tự."""
    S = import_samples()
    ds = S["YoloTrainSet"](synthetic_dataset_root)
    a = _sample_ids(ds)
    b = _sample_ids(ds)
    assert a == b, "hai lần duyệt phải cùng thứ tự"


def test_order_sorted_by_sample_id(synthetic_dataset_root):
    """Bảo vệ contract: thứ tự sort theo sample_id (tất định)."""
    S = import_samples()
    ds = S["YoloTrainSet"](synthetic_dataset_root)
    ids = _sample_ids(ds)
    assert ids == sorted(ids), "phải sort theo sample_id"


# ===========================================================================
# AdTrainSet iter_tiles
# ===========================================================================
def test_ad_train_iter_tiles_partition_and_count(synthetic_dataset_root):
    """Bảo vệ contract: mọi tile thuộc partition train, khớp ảnh nguồn;
    num_tiles = số ảnh * tiles_per_image."""
    S = import_samples()
    ds = S["AdTrainSet"](synthetic_dataset_root)
    assert ds.num_tiles == len(ds) * ds.tiles_per_image, "num_tiles = số ảnh * tiles_per_image"
    seen_samples = set()
    n = 0
    for sample, spec, tile in ds.iter_tiles():
        assert sample.partition == "train", "tile phải thuộc partition train"
        assert isinstance(tile, np.ndarray) and tile.shape == (256, 256, 3), "tile phải [256,256,3]"
        seen_samples.add(sample.sample_id)
        n += 1
    assert n == ds.num_tiles, "số tile duyệt phải = num_tiles"
    assert seen_samples == {s.sample_id for s in ds}, "tile phải khớp mọi ảnh nguồn"


def test_ad_train_iter_tiles_deterministic(synthetic_dataset_root):
    """Bảo vệ contract: iter_tiles tất định (cùng thứ tự hai lần)."""
    S = import_samples()
    ds = S["AdTrainSet"](synthetic_dataset_root)
    ids_a = [s.sample_id for s, _, _ in ds.iter_tiles()]
    ids_b = [s.sample_id for s, _, _ in ds.iter_tiles()]
    assert ids_a == ids_b, "iter_tiles phải tất định"
