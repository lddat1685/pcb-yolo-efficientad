"""Extraction drives the real adapter with a fake engine; failed images stay in JSONL."""

import hashlib
import json

import pytest

from helpers.step3_spec import (FakeEngine, assert_prediction_row, dataset_snapshot,
                               fake_checkpoint_loader, make_fake_artifact, read_json)


@pytest.mark.parametrize("partitions", [("test",), ("train",), ("calibration", "test"),
                                        ("fusion", "train"), ("unknown",)])
def test_extract_rejects_forbidden_partition_before_loading(synthetic_dataset_root, tmp_path, monkeypatch, partitions):
    # Protect plan §1.2/§6 and contract extract.py: every non calibration/fusion path is denied.
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    def forbidden_load(*args, **kwargs):
        pytest.fail("Artifact was loaded before rejecting forbidden partitions")
    monkeypatch.setattr(adapter.YoloDetector, "from_artifact", forbidden_load)
    out = tmp_path / "out"
    with pytest.raises(PermissionError):
        extract_predictions(tmp_path / "absent-artifact", synthetic_dataset_root,
                            partitions=partitions, out_dir=out)
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize("inject_error", [False, True])
def test_extract_complete_deterministic_and_preserves_errors(synthetic_dataset_root, tmp_path, monkeypatch, inject_error):
    # Protect plan §6.1/§6.7/§8, contract extraction: exactly one sorted row/image including errors.
    from pcb_lab.data.samples import ManifestDataset
    from pcb_lab.inference.preprocessing import prepare_input
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    samples = {part: list(ManifestDataset(synthetic_dataset_root, part)) for part in ("calibration", "fusion")}
    failing = samples["calibration"][1]
    target_hash = hashlib.sha256(prepare_input(failing.load_image(), "yolo").pixels.tobytes()).hexdigest()
    def check(pixels):
        if inject_error and hashlib.sha256(pixels.tobytes()).hexdigest() == target_hash:
            raise RuntimeError("VERIFIER_ENGINE_FAILURE")
    # Boundary confidence is exactly representable; no float32 ambiguity at conf_floor.
    engine = FakeEngine([[20, 30, 50, 60, 0.75, 2]], check=check)
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *args, **kwargs: engine)
    meta, run = read_json(artifact / "artifact.json"), read_json(artifact / "run_manifest.json")
    before = dataset_snapshot(synthetic_dataset_root)
    outputs = []
    try:
        for iteration in range(2):
            out = tmp_path / f"extract-{iteration}"
            report = extract_predictions(artifact, synthetic_dataset_root, out_dir=out)
            assert isinstance(report, dict)
            payload = {}
            errors = []
            for part, expected in samples.items():
                path = out / f"preds_{part}.jsonl"
                payload[part] = path.read_bytes()
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                assert len(rows) == len(expected)
                assert [row["sample_id"] for row in rows] == sorted(s.sample_id for s in expected)
                assert len({row["sample_id"] for row in rows}) == len(rows)
                for row, sample in zip(rows, sorted(expected, key=lambda s: s.sample_id)):
                    assert_prediction_row(row, sample, meta, run)
                    if row["error_reason"] is not None:
                        errors.append(row["sample_id"])
                        assert "VERIFIER_ENGINE_FAILURE" in row["error_reason"]
                    else:
                        assert len(row["detections"]) == 1
                        assert row["yolo_image_score"] == pytest.approx(0.75)
            assert errors == ([failing.sample_id] if inject_error else [])
            outputs.append(payload)
        assert outputs[0] == outputs[1], "Raw JSONL must be byte-identical; no wall-clock timing"
        assert engine.calls == 2 * sum(map(len, samples.values()))
    finally:
        assert dataset_snapshot(synthetic_dataset_root) == before


def test_extract_keeps_successful_empty_prediction(synthetic_dataset_root, tmp_path, monkeypatch):
    # Protect plan §7.3 and contract extraction: valid empty inference is explicitly score 0, no error.
    from pcb_lab.models.yolo import adapter
    from pcb_lab.models.yolo.extract import extract_predictions
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    artifact = make_fake_artifact(tmp_path / "artifact", synthetic_dataset_root)
    fake_checkpoint_loader(monkeypatch)
    monkeypatch.setattr(adapter, "UltralyticsEngine", lambda *args, **kwargs: FakeEngine())
    out = tmp_path / "out"
    extract_predictions(artifact, synthetic_dataset_root, partitions=("calibration",), out_dir=out)
    rows = [json.loads(line) for line in (out / "preds_calibration.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert row["detections"] == [] and row["error_reason"] is None
        assert row["yolo_image_score"] == 0.0 and row["timing_ms"] is None
