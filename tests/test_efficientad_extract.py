"""Extraction acceptance: full populations, deterministic rows, visible failures."""

import hashlib

import numpy as np
import pytest

from helpers.step4_spec import (assert_prediction, build_dataset, make_extraction_artifact,
                               read_json, read_rows, resolve_manifest)


@pytest.mark.parametrize("partitions", [("test",), ("train",), ("unknown",),
    ("calibration", "test"), ("fusion", "train"), ("calibration", "fusion", "test")])
def test_extract_forbidden_partition_fails_before_loading(tmp_path, monkeypatch, partitions):
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector
    from pcb_lab.models.efficientad.extract import extract_efficientad_predictions
    def forbidden(*args, **kwargs):
        pytest.fail("Forbidden partitions must be validated before artifact loading")
    monkeypatch.setattr(EfficientAdDetector, "from_artifact", forbidden)
    out = tmp_path / "out"
    with pytest.raises(PermissionError):
        extract_efficientad_predictions(tmp_path / "absent-artifact", tmp_path / "absent-dataset",
                                       partitions=partitions, out_dir=out)
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize("failure", [None, "decode", "inference"])
def test_extract_complete_sorted_deterministic_and_preserves_failures(tmp_path, monkeypatch, failure):
    from pcb_lab.data.samples import Sample
    from pcb_lab.models.efficientad.adapter import Detection, EfficientAdDetector, EfficientAdResult
    from pcb_lab.models.efficientad.extract import extract_efficientad_predictions

    fixture = build_dataset(tmp_path / "dataset", {"train": 2, "calibration": 460, "fusion": 306})
    root, meta, run = make_extraction_artifact(tmp_path / "artifact")
    monkeypatch.setenv("DATASET_ROOT", str(fixture.root))
    target_id = "calibration_0001"
    decoded, inferred = [], []
    original = Sample.load_image
    def checked_load(sample):
        assert sample.partition in ("calibration", "fusion"), "No train/test access during extraction"
        decoded.append(sample.sample_id)
        if failure == "decode" and sample.sample_id == target_id:
            raise OSError("VERIFIER_DECODE_FAILURE")
        return original(sample)
    monkeypatch.setattr(Sample, "load_image", checked_load)

    class FakeDetector:
        def __init__(self):
            # Phase B binding: the concrete detector exposes artifact metadata.
            # run_id comes from the fixture manifest; assertions stay unchanged.
            self.meta = dict(meta, run_id=run["run_id"])

        def predict(self, image):
            assert image.shape == (640, 640, 3)
            identity = tuple(int(c) for c in image[0, 0])
            inferred.append(identity)
            if failure == "inference" and identity == (1, 0, len("calibration")):
                raise RuntimeError("VERIFIER_INFERENCE_FAILURE")
            # Include successful zero-score images so error != legitimate GOOD.
            score = 0.0 if identity[0] % 2 == 0 else 0.75
            boxes = [] if score == 0 else [Detection(xyxy_original=(20., 30., 50., 60.),
                confidence=score, class_name="anomaly_unclassified", class_id=None)]
            return EfficientAdResult(anomaly_score=score,
                                     anomaly_map=np.full((640, 640), score, np.float32), boxes=boxes)
        def describe(self):
            return dict(model_id=meta["model_id"], checkpoint_sha256=meta["checkpoint_sha256"],
                        norm_stats=meta["normalization_stats"], infer_params=meta["inference_params"], input_color="rgb")

    monkeypatch.setattr(EfficientAdDetector, "from_artifact", lambda *args, **kwargs: FakeDetector())
    # All files here were generated locally; no test image exists, even in this fixture.
    def snapshot():
        return {p.relative_to(fixture.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in fixture.root.rglob("*") if p.is_file()}
    before = snapshot()
    payloads = []
    for repeat in range(2):
        decoded.clear()
        inferred.clear()
        out = tmp_path / f"predictions-{repeat}"
        report = extract_efficientad_predictions(root, fixture.root, out_dir=out)
        assert isinstance(report, dict)
        errors, payload = [], {}
        for split, count in (("calibration", 460), ("fusion", 306)):
            path = out / f"preds_{split}.jsonl"
            payload[split] = path.read_bytes()
            rows = read_rows(path)
            expected = sorted((r for r in fixture.records if r["split"] == split), key=lambda r: r["sample_id"])
            assert len(rows) == len(expected) == count
            assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in expected]
            assert len({r["sample_id"] for r in rows}) == count
            for row, record in zip(rows, expected):
                assert_prediction(row, record, meta, run)
                if row["error_reason"] is not None:
                    errors.append(row["sample_id"])
                    assert f"VERIFIER_{failure.upper()}_FAILURE" in row["error_reason"]
                else:
                    score = 0.0 if int(record["sample_id"].rsplit("_", 1)[1]) % 256 % 2 == 0 else 0.75
                    assert row["anomaly_score"] == score
                    assert len(row["boxes"]) == (0 if score == 0 else 1)
        assert errors == ([target_id] if failure else [])
        assert len(decoded) == 766 and len(set(decoded)) == 766
        assert len(inferred) == 766 - int(failure == "decode")
        payloads.append(payload)
    assert payloads[0] == payloads[1], "JSONL must be byte-identical; timing_ms is null"
    assert snapshot() == before, "Extraction must keep the dataset read-only"


def test_real_prediction_rows_cover_exact_development_populations(step4_artifact_dir, real_dataset_root):
    meta = read_json(step4_artifact_dir / "artifact.json")
    run = read_json(resolve_manifest(step4_artifact_dir, meta))
    manifest = read_rows(real_dataset_root / "benchmarks/deeppcb/manifests/samples.jsonl")
    held_out = [r for r in manifest if r["split"] == "test"]  # metadata only
    for split, count in (("calibration", 460), ("fusion", 306)):
        rows = read_rows(step4_artifact_dir / f"preds_{split}.jsonl")
        expected = sorted((r for r in manifest if r["split"] == split), key=lambda r: r["sample_id"])
        assert len(rows) == len(expected) == count
        assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in expected]
        assert len({r["sample_id"] for r in rows}) == count
        for row, record in zip(rows, expected):
            assert_prediction(row, record, meta, run)
            assert row["error_reason"] is None, (row["sample_id"], row["error_reason"])
        for key, source_key in (("sample_id", "sample_id"), ("image_sha256", "sha256"), ("source_group", "source_group")):
            assert not {r[key] for r in rows} & {r[source_key] for r in held_out}
