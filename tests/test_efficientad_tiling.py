"""Independent spatial patterns detect resizing, seam jumps and incorrect max pooling."""

import numpy as np
import pytest

from helpers.step4_spec import FakeEngine, detector_meta, field, tile_maps_from_global


def test_adapter_engine_receives_native_rgb_uint8_without_double_normalization():
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    image = np.random.default_rng(42).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    original = image.copy()
    engine = FakeEngine(np.zeros((9, 256, 256), np.float32))
    EfficientAdDetector(engine, detector_meta()).predict(image)
    assert len(engine.inputs) == 1
    for tile, (y, x) in zip(engine.inputs[0], ((y, x) for y in (0, 224, 448) for x in (0, 224, 448))):
        h, w = min(256, 640 - y), min(256, 640 - x)
        np.testing.assert_array_equal(tile[:h, :w], original[y:y + h, x:x + w])
    np.testing.assert_array_equal(image, original)


def test_training_dataset_emits_nine_native_tensor_tiles_per_good_image(step4_dataset):
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    import torch
    tiles_seen = 0
    for item in EfficientAdDataset(step4_dataset.root, split="train"):
        assert isinstance(item, torch.Tensor)
        assert tuple(item.shape) in ((9, 3, 256, 256), (3, 256, 256))
        assert torch.isfinite(item).all()
        tiles_seen += 9 if item.ndim == 4 else 1
    good_count = sum(r["split"] == "train" and not r["is_defect"] for r in step4_dataset.records)
    assert tiles_seen == good_count * 9


def test_native_nine_tiles_keep_original_pixels_and_locked_stride():
    from pcb_lab.inference.tiling import TileManager
    image = np.random.default_rng(42).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    tiles, specs = TileManager(tile_size=256, stride=224).split(image)
    assert tiles.shape == (9, 256, 256, 3) and tiles.dtype == np.uint8
    assert [(s.x0, s.y0) for s in specs] == [(x, y) for y in (0, 224, 448) for x in (0, 224, 448)]
    for tile, spec in zip(tiles, specs):
        h, w = min(256, 640 - spec.y0), min(256, 640 - spec.x0)
        np.testing.assert_array_equal(tile[:h, :w], image[spec.y0:spec.y0 + h, spec.x0:spec.x0 + w])


@pytest.mark.parametrize("value", [0.0, 0.375, 2.0])
def test_linear_blending_preserves_constant_and_has_no_edge_holes(value):
    from pcb_lab.inference.tiling import TileManager
    manager = TileManager()
    result = manager.stitch(np.full((9, 256, 256), value, np.float32), manager.plan(640, 640), (640, 640))
    assert result.shape == (640, 640) and result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, value, atol=1e-6, rtol=0)


def test_linear_blending_reconstructs_independent_affine_field():
    from pcb_lab.inference.tiling import TileManager
    y, x = np.mgrid[:640, :640]
    expected = ((x + 2 * y) / 1917).astype(np.float32)
    manager = TileManager()
    result = manager.stitch(tile_maps_from_global(expected), manager.plan(640, 640), (640, 640))
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("axis", [0, 1])
def test_overlap_is_linear_monotone_and_continuous(axis):
    from pcb_lab.inference.tiling import TileManager
    manager = TileManager()
    specs = manager.plan(640, 640)
    maps = np.stack([np.full((256, 256), s.row if axis == 0 else s.col, np.float32) for s in specs])
    result = manager.stitch(maps, specs, (640, 640))
    line = result[:, 100] if axis == 0 else result[100, :]
    assert np.all(np.diff(line) >= -1e-6)
    for index, start in enumerate((224, 448)):
        np.testing.assert_allclose(line[start - 2:start], index, atol=1e-6)
        np.testing.assert_allclose(line[start + 32:start + 34], index + 1, atol=1e-6)
        increments = np.diff(line[start:start + 32])
        assert np.all(increments > 0), "average/hard overwrite is not feathering"
        np.testing.assert_allclose(increments, increments.mean(), atol=1e-6, rtol=0)
        assert np.max(np.abs(np.diff(line[start - 1:start + 33]))) <= 2 / 32


def test_padding_is_cropped_before_global_score():
    from pcb_lab.inference.tiling import TileManager
    manager = TileManager()
    specs = manager.plan(640, 640)
    maps = np.full((9, 256, 256), 0.25, np.float32)
    for tile, spec in zip(maps, specs):
        tile[min(256, 640 - spec.y0):, :] = 10000
        tile[:, min(256, 640 - spec.x0):] = 10000
    stitched = manager.stitch(maps, specs, (640, 640))
    assert float(np.max(stitched)) == pytest.approx(0.25)


def test_adapter_score_is_max_stitched_map_not_mean(step4_predict_map):
    heatmap = np.full((640, 640), 0.125, np.float32)
    heatmap[230:240, 230:240] = 0.875
    result = step4_predict_map(heatmap)
    actual = field(result, "anomaly_map")
    assert isinstance(actual, np.ndarray) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, heatmap, atol=1e-6, rtol=0)
    assert type(field(result, "anomaly_score")) is float
    assert field(result, "anomaly_score") == float(np.max(actual))


def test_adapter_score_excludes_extreme_padding(step4_predict_tiles):
    maps = np.full((9, 256, 256), 0.25, np.float32)
    for index, (y, x) in enumerate((y, x) for y in (0, 224, 448) for x in (0, 224, 448)):
        maps[index, min(256, 640 - y):, :] = 10000
        maps[index, :, min(256, 640 - x):] = 10000
    result = step4_predict_tiles(maps)
    assert field(result, "anomaly_score") == pytest.approx(0.25)
    assert field(result, "anomaly_score") == float(np.max(field(result, "anomaly_map")))
