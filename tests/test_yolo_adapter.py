"""RGB, geometry, deterministic ranking and artifact guards from the contract."""

import pytest

from helpers.step3_spec import (CLASSES, FakeEngine, assert_error_name, detection_record,
                               detector_meta, fake_checkpoint_loader, make_fake_artifact,
                               read_json, sha256, write_json)


def test_engine_receives_rgb_and_describe_retains_provenance():
    # Protect plan §3.1/§8/§11 and contract adapter.py: uint8 RGB, identity and infer parameters.
    import numpy as np
    from pcb_lab.models.yolo.adapter import YoloDetector
    pixels = np.empty((640, 640, 3), dtype=np.uint8)
    pixels[:] = [231, 47, 13]
    def check(actual):
        np.testing.assert_array_equal(actual, pixels)
    engine = FakeEngine(check=check)
    meta = detector_meta()
    detector = YoloDetector(engine, meta)
    detector.load()
    assert detector.predict(pixels) == []
    assert engine.calls == 1
    description = detector.describe()
    for key, value in meta.items():
        assert description[key] == value


@pytest.mark.parametrize("width,height,scale,left,top", [
    (640, 640, 1.0, 0, 0), (800, 600, 0.8, 0, 80), (300, 700, 640 / 700, 183, 0)])
def test_adapter_returns_original_coordinates(width, height, scale, left, top):
    # Protect plan §11 coordinates and contract adapter: independent hand-computed letterbox oracle.
    import numpy as np
    from pcb_lab.models.yolo.adapter import YoloDetector
    original = np.array([12.25, 20.5, width - 17.5, height - 10.25])
    letterboxed = original * scale + np.array([left, top, left, top])
    engine = FakeEngine([[*letterboxed, 0.75, 2]])
    result = YoloDetector(engine, detector_meta()).predict(np.zeros((height, width, 3), dtype=np.uint8))
    assert len(result) == 1
    det = detection_record(result[0])
    assert det["class_id"] == 2 and det["class_name"] == CLASSES[2]
    assert det["confidence"] == pytest.approx(0.75)
    np.testing.assert_allclose(det["xyxy_original"], original, atol=0.01, rtol=0)


def test_adapter_sort_ties_and_image_score_are_order_independent():
    # Protect contract adapter sort(conf desc, class, xyxy), plan §7.3 max-confidence score.
    import numpy as np
    from pcb_lab.models.yolo import adapter
    rows = [[30, 20, 70, 90, 0.8, 1], [20, 20, 70, 90, 0.8, 1],
            [40, 20, 70, 90, 0.8, 0], [10, 20, 70, 90, 0.95, 5],
            [10, 20, 70, 90, 0.4, 0]]
    engine = FakeEngine(rows)
    detector = adapter.YoloDetector(engine, detector_meta())
    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    first = detector.predict(pixels)
    engine.detections = engine.detections[::-1].copy()
    second = detector.predict(pixels)
    records = [detection_record(det) for det in first]
    assert records == [detection_record(det) for det in second]
    keys = [(-det["confidence"], det["class_id"], tuple(det["xyxy_original"])) for det in records]
    assert keys == sorted(keys) and len(records) == len(rows)
    # B correction: contract leaves placement open; implementation exposes this on the detector.
    assert detector.image_score(first) == pytest.approx(0.95)
    assert detector.image_score([]) == 0.0


@pytest.fixture
def fake_artifact(synthetic_dataset_root, tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    return make_fake_artifact(tmp_path / "fake-artifact", synthetic_dataset_root)


@pytest.mark.parametrize("filename", ["artifact.json", "best.pt"])
def test_from_artifact_missing_required_file(fake_artifact, filename):
    # Protect supervisor clarification: missing artifact.json/best.pt -> FileNotFoundError.
    from pcb_lab.models.yolo.adapter import YoloDetector
    (fake_artifact / filename).unlink()
    with pytest.raises(FileNotFoundError):
        YoloDetector.from_artifact(fake_artifact)


def test_from_artifact_rejects_one_byte_checkpoint_corruption(fake_artifact):
    # Protect plan §11 provenance and contract ArtifactMismatchError before model deserialization.
    from pcb_lab.models.yolo.adapter import YoloDetector
    path = fake_artifact / "best.pt"
    damaged = bytearray(path.read_bytes())
    damaged[0] ^= 1
    path.write_bytes(damaged)
    assert_error_name("ArtifactMismatchError", lambda: YoloDetector.from_artifact(fake_artifact))


def test_from_artifact_smoke_requires_explicit_opt_in(fake_artifact, monkeypatch):
    # Protect contract SmokeArtifactError plus allow_smoke positive control; no real torch file.
    from pcb_lab.models.yolo import adapter
    path = fake_artifact / "artifact.json"
    meta = read_json(path)
    meta["smoke"] = True
    run_path = fake_artifact / "run_manifest.json"
    run = read_json(run_path)
    run["smoke"] = True
    write_json(run_path, run)
    meta["run_manifest"]["sha256"] = sha256(run_path)
    write_json(path, meta)
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *args, **kwargs: FakeEngine())
    assert_error_name("SmokeArtifactError", lambda: adapter.YoloDetector.from_artifact(fake_artifact))
    assert adapter.YoloDetector.from_artifact(fake_artifact, allow_smoke=True) is not None


def test_from_artifact_rejects_model_class_order(fake_artifact, monkeypatch):
    # Protect contract ClassOrderError: artifact metadata stays correct; CHECKPOINT model.names changes.
    from pcb_lab.models.yolo import adapter
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *args, **kwargs: FakeEngine())
    assert adapter.YoloDetector.from_artifact(fake_artifact) is not None
    wrong_order = CLASSES.copy()
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    fake_checkpoint_loader(monkeypatch, wrong_order)
    assert_error_name("ClassOrderError", lambda: adapter.YoloDetector.from_artifact(fake_artifact))


def test_standalone_artifact_labels_do_not_need_dataset(fake_artifact, monkeypatch):
    from pcb_lab.models.yolo import adapter
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *args, **kwargs: FakeEngine())
    def no_dataset(*args):
        raise AssertionError("Standalone loading must not resolve a dataset")
    monkeypatch.setattr(adapter, "resolve_classes_root", no_dataset)
    assert adapter.YoloDetector.from_artifact(fake_artifact, class_names=CLASSES) is not None
    fake_checkpoint_loader(monkeypatch, list(reversed(CLASSES)))
    with pytest.raises(adapter.ClassOrderError):
        adapter.YoloDetector.from_artifact(fake_artifact, class_names=CLASSES)


@pytest.mark.parametrize("names", [[], ["a", "a"], [1], [""], "a", ["wrong"]])
def test_standalone_rejects_invalid_artifact_labels(fake_artifact, names):
    from pcb_lab.models.yolo import adapter
    with pytest.raises(adapter.ClassOrderError):
        adapter.YoloDetector.from_artifact(fake_artifact, class_names=names)
