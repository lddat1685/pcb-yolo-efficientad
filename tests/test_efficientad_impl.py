"""Implementer checks for Step 4 isolation, R8, provenance and extraction.

These checks complement, and do not replace, independent VERIFIER tests.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def recipe(tmp_path, **changes):
    cfg = yaml.safe_load((REPO / "configs/models/efficientad_s.yaml").read_text(encoding="utf-8-sig"))
    for key, value in changes.items():
        section, field = key.split("__")
        cfg[section][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.parametrize("field,value", [("lr", None), ("weight_decay", None), ("lr", float("nan")),
    ("batch_size", 2), ("tile_size", 320), ("stride", 256), ("unknown", 1), ("deterministic", False)])
def test_config_rejects_incomplete_and_unknown(tmp_path, field, value):
    from pcb_lab.models.efficientad.config import ConfigError, load_config
    with pytest.raises(ConfigError):
        load_config(recipe(tmp_path, **{"train__" + field: value}), device="cpu", smoke=True)


def test_hash_excludes_nested_seed_and_canonicalizes_device(tmp_path):
    from pcb_lab.models.efficientad.config import config_hash, load_config
    path = recipe(tmp_path)
    first = load_config(path, seed=42, device="0")
    second = load_config(path, seed=43, device="cuda:0")
    assert config_hash(first) == config_hash(second)
    second["train"]["lr"] *= 2
    assert config_hash(first) != config_hash(second)


def test_test_partition_denied_before_any_io(monkeypatch):
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    from pcb_lab.models.efficientad.extract import extract_efficientad_predictions
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("Unexpected file access"))
    with pytest.raises(PermissionError):
        EfficientAdDataset("missing", "test")
    with pytest.raises(PermissionError):
        extract_efficientad_predictions("missing", "missing", partitions=("calibration", "test"))


def test_injected_train_defect_fails_before_decode(monkeypatch):
    import pcb_lab.models.efficientad.dataset as module
    bad = SimpleNamespace(sample_id="leaked", partition="train", is_defect=True, boxes=(object(),))
    monkeypatch.setattr(module, "AdTrainSet", lambda _root: [bad])
    with pytest.raises(module.DataLeakError):
        module.EfficientAdDataset("unused")


def test_calibration_stats_reject_defect_before_decode():
    from pcb_lab.models.efficientad.dataset import DataLeakError
    from pcb_lab.models.efficientad.train import map_statistics
    bad = SimpleNamespace(sample_id="leaked", partition="calibration", is_defect=True, boxes=())
    dataset = SimpleNamespace(samples=[bad], iter_tiles=lambda: pytest.fail("Decoded a defect"))
    with pytest.raises(DataLeakError):
        map_statistics(None, dataset, "cpu", (.9, .995))


def test_native_tiles_and_metadata_counts(real_dataset_root):
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    train = EfficientAdDataset(real_dataset_root)
    assert len(train) == 895 and train.num_tiles == 8055
    assert all(s.partition == "train" and not s.is_defect and not s.boxes for s in train.samples)
    limited = EfficientAdDataset(real_dataset_root, limit={"good": 4})
    assert len(limited) == 4 and limited.num_tiles == 36
    assert limited[0].shape == (9, 3, 256, 256)
    assert len(EfficientAdDataset(real_dataset_root, "calibration")) == 460
    assert len(EfficientAdDataset(real_dataset_root, "fusion")) == 306


class PixelEngine:
    def infer_tiles(self, tiles):
        assert tiles.shape[1:] == (256, 256, 3) and tiles.dtype == np.uint8
        return tiles[..., 0].astype(np.float32) / 255


def test_blending_and_r8_component_area_score():
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    pixels = np.zeros((640, 640, 3), np.uint8)
    pixels[220:231, 220:232] = 200  # crosses horizontal and vertical tile seams
    pixels[100:103, 100:107] = 180  # area 21: retained
    pixels[150:154, 150:155] = 180  # area 20: strictly rejected
    pixels[20, 20] = 255  # max image score survives; opening removes its box
    detector = EfficientAdDetector(PixelEngine(), {})
    result = detector.predict(pixels)
    np.testing.assert_allclose(result.anomaly_map, pixels[..., 0] / 255, rtol=0, atol=1e-7)
    assert result.anomaly_map.dtype == np.float32 and result.anomaly_map.shape == (640, 640)
    assert result.anomaly_score == 1.0
    assert [box.xyxy_original for box in result.boxes] == [(220., 220., 232., 231.), (100., 100., 107., 103.)]
    assert result.boxes[0].confidence == pytest.approx(200 / 255)
    assert all(box.class_id is None and box.class_name == "anomaly_unclassified" for box in result.boxes)


def test_nms_and_component_confidence_not_bounding_rect_max():
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    image = np.zeros((640, 640, 3), np.uint8)
    image[80:150, 80:150] = 180
    image[85:145, 85:145] = 0  # hollow component
    image[100:105, 100:105] = 250  # separate component inside bounding rectangle
    result = EfficientAdDetector(PixelEngine(), {}).predict(image)
    assert len(result.boxes) == 2
    assert sorted(b.confidence for b in result.boxes) == pytest.approx([180 / 255, 250 / 255])


def test_global_nms_suppresses_overlapping_component_rectangles():
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    image = np.zeros((640, 640, 3), np.uint8)
    image[100:105, 100:160] = 180
    image[100:160, 100:105] = 180
    image[110:170, 110:170] = 200
    # Disjoint masks, rectangles have IoU=2500/4700 > .45.
    result = EfficientAdDetector(PixelEngine(), {}).predict(image)
    assert len(result.boxes) == 1
    assert result.boxes[0].xyxy_original == (110., 110., 170., 170.)


def test_run_overwrite_and_resume_without_checkpoint_rejected(tmp_path, monkeypatch):
    import pcb_lab.models.efficientad.train as module
    config = recipe(tmp_path)
    target = tmp_path / "out"
    run = target / "runs/smoke/B03_efficientad_s/seed42"
    run.mkdir(parents=True)
    monkeypatch.setattr(module, "EfficientAdDataset", lambda *_a, **_kw: pytest.fail("Unexpected dataset access"))
    with pytest.raises(FileExistsError):
        module.train_efficientad(config, out_root=target, smoke=True)
    with pytest.raises(FileNotFoundError):
        module.train_efficientad(config, out_root=target, smoke=True, resume=True)


def test_adapter_rejects_nan_map():
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    engine = SimpleNamespace(infer_tiles=lambda tiles: np.full(tiles.shape[:3], np.nan, np.float32))
    with pytest.raises(ValueError):
        EfficientAdDetector(engine, {}).predict(np.zeros((640, 640, 3), np.uint8))


def test_real_model_backward_freezes_teacher():
    import torch
    from pcb_lab.models.efficientad.architecture import EfficientAdModel
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = EfficientAdModel().train()
    assert not model.teacher.training
    loss = model.loss(torch.rand(1, 3, 256, 256))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.student.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.autoencoder.parameters())


@pytest.mark.parametrize("source,target", [(1, 3), (4, 8), (9, 15), (16, 32), (33, 63), (64, 127), (128, 56)])
def test_deterministic_resize_forward_and_gradient(source, target):
    import torch
    from pcb_lab.models.efficientad.architecture import deterministic_bilinear
    torch.manual_seed(42)
    original = torch.rand(1, 2, source, source, dtype=torch.float64, requires_grad=True)
    reference_input = original.detach().clone().requires_grad_()
    actual = deterministic_bilinear(original, target)
    expected = torch.nn.functional.interpolate(reference_input, size=(target, target), mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(original.grad, reference_input.grad, rtol=1e-10, atol=1e-10)


def test_artifact_smoke_and_sha_guards_before_torch_load(tmp_path):
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector, ArtifactMismatchError, SmokeArtifactError
    (tmp_path / "artifact.json").write_text(json.dumps({"smoke": True, "checkpoint_sha256": "wrong"}))
    (tmp_path / "model.pt").write_bytes(b"not a torch checkpoint")
    with pytest.raises(SmokeArtifactError):
        EfficientAdDetector.from_artifact(tmp_path)
    with pytest.raises(ArtifactMismatchError):
        EfficientAdDetector.from_artifact(tmp_path, allow_smoke=True)


def test_full_extraction_rows_determinism_and_errors(real_dataset_root, tmp_path, monkeypatch):
    """Exercise all manifest rows using an injected detector; not a model benchmark."""
    import pcb_lab.models.efficientad.extract as module
    from pcb_lab.models.efficientad.adapter import EfficientAdResult
    from pcb_lab.data.samples import EvalSet
    meta = dict(run_id="fake", model_id="B03_efficientad_s", smoke=False,
                checkpoint_sha256="a" * 64, config_hash="b" * 64, preprocessing_version="fake", tile_params={})
    class Fake:
        def predict(self, image):
            return EfficientAdResult(float(image.max()) / 255, image[..., 0].astype(np.float32), [])
    detector = Fake()
    detector.meta = meta
    monkeypatch.setattr(module.EfficientAdDetector, "from_artifact", lambda *_a, **_kw: detector)
    result = module.extract_efficientad_predictions("unused", real_dataset_root, out_dir=tmp_path)
    assert result["calibration"]["n_rows"] == 460 and result["fusion"]["n_rows"] == 306
    assert all(info["n_errors"] == 0 for info in result.values())
    repeated = module.extract_efficientad_predictions("unused", real_dataset_root, out_dir=tmp_path)
    assert result == repeated
    rows = [json.loads(line) for line in (tmp_path / "preds_calibration.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == sorted(s.sample_id for s in EvalSet(real_dataset_root, "calibration"))
    assert all(row["config_hash"] == meta["config_hash"] and row["checkpoint_sha256"] == meta["checkpoint_sha256"]
               and row["timing_ms"] is None for row in rows)
    broken = list(EvalSet(real_dataset_root, "calibration"))[:2]
    broken[0] = replace(broken[0], image_path=tmp_path / "missing.png")
    monkeypatch.setattr(module, "EvalSet", lambda *_a: broken)
    result = module.extract_efficientad_predictions("unused", real_dataset_root, partitions=("calibration",), out_dir=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "preds_calibration.jsonl").read_text().splitlines()]
    assert result["calibration"]["n_rows"] == 2 and result["calibration"]["n_errors"] == 1
    assert rows[0]["anomaly_score"] is None and rows[0]["error_reason"].startswith("FileNotFoundError")
