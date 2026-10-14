"""Save/reload without relying on private checkpoint keys or engine construction."""

import shutil

import numpy as np
import pytest

from helpers.step4_spec import (assert_named_error, field, make_extraction_artifact,
                               FakeEngine, detector_meta, read_json, resolve_manifest, sha256, write_json)


def test_fake_engine_describe_preserves_normalization_and_inference_metadata():
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    meta = detector_meta()
    detector = EfficientAdDetector(FakeEngine(np.zeros((9, 256, 256), np.float32)), meta)
    description = detector.describe()
    assert description["input_color"] == "rgb"
    assert description["model_id"] == meta["model_id"]
    assert description["checkpoint_sha256"] == meta["checkpoint_sha256"]
    assert description["norm_stats"] == meta["normalization_stats"]
    assert description["infer_params"] == meta["inference_params"]


def test_checkpoint_hash_mismatch_is_explicit_before_deserialization(tmp_path):
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    root, meta, _ = make_extraction_artifact(tmp_path / "tampered")
    meta["checkpoint_sha256"] = "0" * 64
    write_json(root / "artifact.json", meta)
    with pytest.raises(Exception) as caught:
        EfficientAdDetector.from_artifact(root)
    assert_named_error(caught.value, "ArtifactMismatchError")


def test_smoke_artifact_is_rejected_by_default(tmp_path):
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    root, meta, _ = make_extraction_artifact(tmp_path / "smoke")
    meta["smoke"] = True
    write_json(root / "artifact.json", meta)
    with pytest.raises(Exception) as caught:
        EfficientAdDetector.from_artifact(root)
    assert_named_error(caught.value, "SmokeArtifactError")


def test_real_describe_matches_artifact(step4_artifact_dir):
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    meta = read_json(step4_artifact_dir / "artifact.json")
    description = EfficientAdDetector.from_artifact(step4_artifact_dir).describe()
    assert isinstance(description, dict)
    assert {"model_id", "checkpoint_sha256", "norm_stats", "infer_params", "input_color"} <= description.keys()
    assert description["model_id"] == meta["model_id"]
    assert description["checkpoint_sha256"] == meta["checkpoint_sha256"]
    assert description["norm_stats"] == meta["normalization_stats"]
    assert description["infer_params"] == meta["inference_params"]
    assert description["input_color"] == "rgb"


def test_checkpoint_save_reload_preserves_scores_and_maps(step4_artifact_dir, real_dataset_root, tmp_path):
    import torch
    from pcb_lab.data.samples import ManifestDataset
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector

    before = EfficientAdDetector.from_artifact(step4_artifact_dir)
    samples = sorted(ManifestDataset(real_dataset_root, "calibration"), key=lambda s: s.sample_id)
    chosen = [next(s for s in samples if s.is_defect), next(s for s in samples if not s.is_defect)]
    chosen += [s for s in samples if s.sample_id not in {x.sample_id for x in chosen}][:3]
    assert len(chosen) == 5 and {s.is_defect for s in chosen} == {True, False}
    images = [s.load_image() for s in chosen]
    expected = [before.predict(image) for image in images]

    # A supplied STEP4_ARTIFACT_DIR is a trusted local model. Preserve its entire
    # payload; do not invent teacher/student/autoencoder state_dict key names.
    payload = torch.load(step4_artifact_dir / "model.pt", map_location="cpu", weights_only=False)
    clone = tmp_path / "roundtrip"
    clone.mkdir()
    for filename in ("artifact.json", "model_card.md", "calibration_stats.json"):
        shutil.copy2(step4_artifact_dir / filename, clone / filename)
    torch.save(payload, clone / "model.pt")
    meta = read_json(clone / "artifact.json")
    run = read_json(resolve_manifest(step4_artifact_dir, meta))
    meta["checkpoint_sha256"] = run["checkpoint_sha256"] = sha256(clone / "model.pt")
    write_json(clone / "run_manifest.json", run)
    meta["run_manifest"] = dict(path=str(clone / "run_manifest.json"), sha256=sha256(clone / "run_manifest.json"))
    write_json(clone / "artifact.json", meta)
    after = EfficientAdDetector.from_artifact(clone)
    for image, reference in zip(images, expected):
        actual = after.predict(image)
        assert type(field(actual, "anomaly_score")) is float
        assert abs(field(actual, "anomaly_score") - field(reference, "anomaly_score")) <= 1e-5
        actual_map, expected_map = field(actual, "anomaly_map"), field(reference, "anomaly_map")
        assert actual_map.shape == expected_map.shape == (640, 640)
        assert actual_map.dtype == expected_map.dtype == np.float32
        assert np.isfinite(actual_map).all() and np.isfinite(expected_map).all()
        np.testing.assert_allclose(actual_map, expected_map, atol=1e-5, rtol=0)
        assert field(actual, "anomaly_score") == float(np.max(actual_map))
