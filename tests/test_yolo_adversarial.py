"""Phase B attacks. Synthetic data only for every deliberately unsafe output/mutation."""

import importlib
import json
from pathlib import Path

import pytest

from helpers.step3_spec import (CLASSES, FakeEngine, approved_test_config, dataset_snapshot,
    assert_boxes_close_unordered, detection_record, detector_meta, fake_checkpoint_loader, make_fake_artifact, read_json,
    records, sha256, write_config)


@pytest.mark.parametrize("attack", ["class6", "outside", "degenerate"])
def test_invalid_manifest_box_rejected(tmp_path, attack):
    # Protect plan §1.3/§11 and view invariant: reject invalid manifest boxes before building YOLO labels.
    from helpers.dataset_builder import build_clean_fixture
    from pcb_lab.models.yolo.view import build_yolo_view
    fx = build_clean_fixture(tmp_path / "source")
    sample = next(row for row in fx.records if row["split"] == "train" and row["is_defect"])
    if attack == "class6":
        fx.set_record_class_id(sample["sample_id"], 6)
    else:
        fx.set_record_box_invalid(sample["sample_id"], [-1, 0, 20, 20] if attack == "outside" else [20, 20, 20, 40])
    fx.write_manifest()
    fx.write_release()
    before = dataset_snapshot(fx.root)
    with pytest.raises(ValueError):
        build_yolo_view(fx.root, tmp_path / "view")
    assert dataset_snapshot(fx.root) == before


def test_copy_mode_does_not_alias_source_inode(synthetic_dataset_root, tmp_path):
    # Protect contract explicit link='copy': view edits must not mutate the source through a hardlink.
    import os
    from pcb_lab.models.yolo.view import build_yolo_view
    out = tmp_path / "copy"
    build_yolo_view(synthetic_dataset_root, out, link="copy")
    source_by_hash = {row["sha256"]: synthetic_dataset_root / row["image"]
                      for row in records(synthetic_dataset_root) if row["split"] == "train"}
    for image in (out / "images/train").iterdir():
        assert not os.path.samefile(image, source_by_hash[sha256(image)]), "copy produced a hardlink"


def test_existing_empty_run_directory_rejected(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract train.py: existing run directory must be refused unless resume=True.
    train = importlib.import_module("pcb_lab.models.yolo.train")
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    path = write_config(tmp_path / "cfg", approved_test_config())
    (tmp_path / "out/runs/smoke/B01_yolo11n/seed42").mkdir(parents=True)
    monkeypatch.setattr(train, "YOLO", lambda *a, **k: pytest.fail("Existing directory was accepted"))
    with pytest.raises(train.ConfigError):
        train.train_yolo(path, out_root=tmp_path / "out", device="cpu", smoke=True)


def test_existing_nonempty_run_directory_preserved(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract no overwrite: fail before replacing any existing checkpoint/run file.
    train = importlib.import_module("pcb_lab.models.yolo.train")
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    path = write_config(tmp_path / "cfg", approved_test_config())
    run = tmp_path / "out/runs/smoke/B01_yolo11n/seed42"
    run.mkdir(parents=True)
    sentinel = run / "existing.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    with pytest.raises(train.ConfigError):
        train.train_yolo(path, out_root=tmp_path / "out", smoke=True)
    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_trainer_rejects_output_inside_dataset_before_writing(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract read-only dataset: output guard must precede mkdir, even on rejected calls.
    train = importlib.import_module("pcb_lab.models.yolo.train")
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    path = write_config(tmp_path / "cfg", approved_test_config())
    before = dataset_snapshot(synthetic_dataset_root)
    with pytest.raises((ValueError, PermissionError)):
        train.train_yolo(path, out_root=synthetic_dataset_root / "ILLEGAL", smoke=True)
    after = dataset_snapshot(synthetic_dataset_root)
    assert after == before, sorted(set(after) - set(before))


def test_extraction_rejects_output_inside_dataset(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract no dataset writes: explicitly supplied prediction directory must be guarded.
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *a, **k: FakeEngine())
    before = dataset_snapshot(synthetic_dataset_root)
    try:
        with pytest.raises((ValueError, PermissionError)):
            extract_predictions(artifact, synthetic_dataset_root, partitions=("calibration",),
                                out_dir=synthetic_dataset_root / "ILLEGAL")
    finally:
        after = dataset_snapshot(synthetic_dataset_root)
        assert after == before, sorted(set(after) - set(before))


def test_all_four_coordinates_break_adapter_sort_ties():
    # Protect contract deterministic tie order: equal confidence/class/x1/y1 still sort x2 then y2.
    import numpy as np
    from pcb_lab.models.yolo.adapter import YoloDetector
    engine = FakeEngine([[10, 10, 80, 90, 0.5, 0], [10, 10, 70, 90, 0.5, 0],
                         [10, 10, 70, 80, 0.5, 0]])
    detector = YoloDetector(engine, detector_meta())
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    first = [detection_record(d) for d in detector.predict(image)]
    engine.detections = engine.detections[::-1].copy()
    second = [detection_record(d) for d in detector.predict(image)]
    assert first == second


def test_extra_checkpoint_class_is_rejected(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect plan six-class schema and contract ClassOrderError; a matching prefix is insufficient.
    from pcb_lab.models.yolo import adapter
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    fake_checkpoint_loader(monkeypatch, CLASSES + ["unexpected_seventh_class"])
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *a, **k: FakeEngine())
    with pytest.raises(adapter.ClassOrderError):
        adapter.YoloDetector.from_artifact(artifact)


def test_online_augmentation_flips_pixels_with_the_box():
    # Protect plan §5 Bước 2/3 and contract reuse: an asymmetric image/box must undergo the same flip.
    import numpy as np
    train = importlib.import_module("pcb_lab.models.yolo.train")
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[2:5, 1:3] = 255
    boxes = np.array([[0.2, 0.35, 0.2, 0.3]])
    output, mapped = train._geometric_augment(image, boxes, k=0, flip=True)
    np.testing.assert_array_equal(output, image[:, ::-1, :])
    np.testing.assert_allclose(mapped, [[0.8, 0.35, 0.2, 0.3]])


def test_extract_detection_schema_independent_of_run_id(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect plan §8 Detection schema even if a separate provenance/run_id assertion already fails.
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *a, **k: FakeEngine([[10, 10, 30, 30, 0.5, 0]]))
    out = tmp_path / "preds"
    extract_predictions(artifact, synthetic_dataset_root, partitions=("calibration",), out_dir=out)
    rows = [json.loads(line) for line in (out / "preds_calibration.jsonl").read_text().splitlines()]
    assert rows and all(row["error_reason"] is None for row in rows)
    assert all("xyxy_original" in det for row in rows for det in row["detections"])


def test_extract_preserves_engine_failure_without_dropping_row(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect plan §6.7/§11 failure handling independently from coordinate/provenance schema tests.
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    fake_checkpoint_loader(monkeypatch)
    def fail(_pixels):
        raise RuntimeError("explicit-probe-error")
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *a, **k: FakeEngine(check=fail))
    out = tmp_path / "preds"
    extract_predictions(artifact, synthetic_dataset_root, partitions=("calibration",), out_dir=out)
    rows = [json.loads(line) for line in (out / "preds_calibration.jsonl").read_text().splitlines()]
    expected = sorted(row["sample_id"] for row in records(synthetic_dataset_root) if row["split"] == "calibration")
    assert [row["sample_id"] for row in rows] == expected
    assert all("explicit-probe-error" in row["error_reason"] for row in rows)


def test_relative_pretrained_still_points_to_existing_file(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract local pretrained/allow_download=False: chdir must not invalidate a verified path.
    train = importlib.import_module("pcb_lab.models.yolo.train")
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    monkeypatch.chdir(tmp_path)
    config = approved_test_config()
    pretrained = tmp_path / config["pretrained"]
    pretrained.parent.mkdir(parents=True)
    pretrained.write_bytes(b"synthetic-local-checkpoint-for-path-probe")
    path = write_config(tmp_path / "cfg", config)
    seen = {}
    def capture(argument):
        seen["path"] = Path(argument).resolve()
        seen["exists"] = Path(argument).is_file()
        raise StopIteration("stop before deserialization")
    monkeypatch.setattr(train, "YOLO", capture)
    with pytest.raises(StopIteration):
        train.train_yolo(path, out_root=tmp_path / "out", device="cpu", allow_download=False)
    assert seen == {"path": pretrained.resolve(), "exists": True}, seen


def test_resume_is_forwarded_to_actual_trainer(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract resume=True: bypassing overwrite guard alone does not resume training.
    train = importlib.import_module("pcb_lab.models.yolo.train")
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    path = write_config(tmp_path / "cfg", approved_test_config())
    run = tmp_path / "out/runs/smoke/B01_yolo11n/seed42"
    (run / "weights").mkdir(parents=True)
    (run / "weights/last.pt").write_bytes(b"existing-checkpoint-path-probe")
    seen = {}
    monkeypatch.setattr(train, "YOLO", lambda *a, **k: object())
    def capture(*args, **kwargs):
        seen.update(kwargs["overrides"])
        raise StopIteration("stop before training")
    monkeypatch.setattr(train, "_PCBDetectionTrainer", capture)
    with pytest.raises(StopIteration):
        train.train_yolo(path, out_root=tmp_path / "out", device="cpu", smoke=True, resume=True)
    assert seen.get("resume"), seen.get("resume")


def test_roundtrip_oracle_handles_equal_x1_without_relaxing_tolerance():
    # Protect contract <=0.01 px oracle itself: same x1/class must not cause wrong box pairing after rounding.
    actual = [(0, (260.99968, 546, 292.99968, 574)), (0, (261.00032, 143.00032, 290, 167.00032))]
    expected = [(0, (261, 143, 290, 167)), (0, (261, 546, 293, 574))]
    assert_boxes_close_unordered(actual, expected)
    with pytest.raises(AssertionError):
        assert_boxes_close_unordered(actual, [(0, (262, 143, 290, 167)), expected[1]])


def test_extraction_byte_determinism_independent_of_schema(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect contract deterministic one-line-per-image independently of separately failing run_id/box-key checks.
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *a, **k: FakeEngine([[10, 10, 30, 30, 0.5, 0]]))
    outputs = []
    for index in range(2):
        out = tmp_path / f"predictions-{index}"
        extract_predictions(artifact, synthetic_dataset_root, out_dir=out)
        payload = {}
        for partition in ("calibration", "fusion"):
            path = out / f"preds_{partition}.jsonl"
            payload[partition] = path.read_bytes()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            expected = sorted(r["sample_id"] for r in records(synthetic_dataset_root) if r["split"] == partition)
            assert [row["sample_id"] for row in rows] == expected
            assert all(row["timing_ms"] is None and row["error_reason"] is None for row in rows)
        outputs.append(payload)
    assert outputs[0] == outputs[1]
