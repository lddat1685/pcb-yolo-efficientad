"""Real-artifact checks opt in with STEP3_ARTIFACT_DIR, independently of implementation imports."""

import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

import pytest

from helpers.step3_spec import (CLASSES, assert_artifact_schema, assert_number, assert_prediction_row,
                               assert_run_schema, read_json, resolve_recorded_path, sha256)
from helpers.step3_ap import ap50_101, read_rows

pytestmark = pytest.mark.artifacts


def test_real_artifact_manifest_schema_and_file_hashes(step3_artifact_dir):
    # Protect plan §5 Bước 3/§8/§11 provenance and contract artifact schema/invariant 6.
    root = step3_artifact_dir
    artifact = read_json(root / "artifact.json")
    assert_artifact_schema(artifact)
    manifest = resolve_recorded_path(artifact["run_manifest"]["path"], root)
    assert sha256(manifest) == artifact["run_manifest"]["sha256"]
    run = read_json(manifest)
    assert_run_schema(run)
    for kind in ("best", "last"):
        actual = sha256(root / f"{kind}.pt")
        assert actual == artifact[f"checkpoint_{kind}_sha256"] == run["checkpoints"][f"{kind}_sha256"]
    assert artifact["smoke"] is False and run["smoke"] is False
    assert run["git_dirty"] is False
    for key in ("model_id", "seed", "config_hash", "view_signature"):
        assert artifact[key] == run[key]
    assert artifact["created_by_git_commit"] == run["git_commit"]


def test_real_artifact_calibration_schema(step3_artifact_dir):
    # Protect plan §5 Bước 3/§7.3 and contract calibration report: AP separate from F1-optimal P/R.
    report = read_json(step3_artifact_dir / "calibration_per_class.json")
    assert {"split", "n_images", "overall", "per_class", "source", "ultralytics_version"} <= report.keys()
    assert report["split"] == "calibration" and report["n_images"] == 460
    assert type(report["n_images"]) is int
    assert isinstance(report["source"], str) and report["source"]
    assert isinstance(report["ultralytics_version"], str) and report["ultralytics_version"]
    # Contract calls this "infer params" without prescribing its spelling.
    params = [report[key] for key in ("infer_params", "inference_params", "infer") if key in report]
    assert params, "Calibration report must disclose its inference parameters"
    assert all(value == read_json(step3_artifact_dir / "artifact.json")["inference_params"] for value in params)
    overall = report["overall"]
    assert {"map50", "map50_95", "precision", "recall", "pr_definition", "pr_conf"} <= overall.keys()
    assert isinstance(overall["pr_definition"], str) and overall["pr_definition"]
    for key in ("map50", "map50_95", "precision", "recall", "pr_conf"):
        assert_number(overall[key])
        assert overall[key] <= 1
    assert set(report["per_class"]) == set(CLASSES)
    for metrics in report["per_class"].values():
        assert {"n_gt_boxes", "precision", "recall", "ap50", "ap50_95"} <= metrics.keys()
        assert type(metrics["n_gt_boxes"]) is int and metrics["n_gt_boxes"] >= 0
        for key in ("precision", "recall", "ap50", "ap50_95"):
            assert_number(metrics[key])
            assert metrics[key] <= 1


@pytest.mark.dataset
def test_real_artifact_prediction_rows(step3_artifact_dir, real_dataset_root):
    # Protect plan §1.2/§6/§8 and contract invariant 5: full 460/306 images, no silent errors.
    from pcb_lab.data.samples import ManifestDataset
    root = step3_artifact_dir
    meta = read_json(root / "artifact.json")
    run = read_json(resolve_recorded_path(meta["run_manifest"]["path"], root))
    for part, count in (("calibration", 460), ("fusion", 306)):
        samples = sorted(ManifestDataset(real_dataset_root, part), key=lambda s: s.sample_id)
        rows = [json.loads(line) for line in (root / f"preds_{part}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == len(samples) == count
        assert [row["sample_id"] for row in rows] == [sample.sample_id for sample in samples]
        assert len({row["sample_id"] for row in rows}) == count
        for row, sample in zip(rows, samples):
            assert_prediction_row(row, sample, meta, run)
            assert row["error_reason"] is None, (row["sample_id"], row["error_reason"])
            assert len(row["detections"]) < meta["inference_params"]["max_det"], "Investigate max_det saturation"


def test_real_checkpoint_hashes_independent_of_metadata_nulls(step3_artifact_dir):
    # Protect contract invariant 6: null provenance must not prevent independent checksum verification.
    root = step3_artifact_dir
    meta = read_json(root / "artifact.json")
    manifest = resolve_recorded_path(meta["run_manifest"]["path"], root)
    run = read_json(manifest)
    assert sha256(manifest) == meta["run_manifest"]["sha256"]
    for kind in ("best", "last"):
        assert sha256(root / f"{kind}.pt") == sha256(manifest.parent / "weights" / f"{kind}.pt") == \
            meta[f"checkpoint_{kind}_sha256"] == run["checkpoints"][f"{kind}_sha256"]


@pytest.mark.dataset
def test_real_calibration_gt_counts_independent(real_dataset_root, record_property):
    # Protect plan §1.3/§5 Bước 3: count actual GT, never infer GT count from a precision/recall curve.
    from pcb_lab.data.samples import ManifestDataset
    rows = [r for r in read_rows(real_dataset_root / "benchmarks/deeppcb/manifests/samples.jsonl")
            if r["split"] == "calibration"]
    manifest = Counter(b["class_name"] for row in rows for b in row["boxes"])
    samples = Counter(b.class_name for sample in ManifestDataset(real_dataset_root, "calibration") for b in sample.boxes)
    labels = Counter()
    for row in rows:
        for line in (real_dataset_root / row["label"]).read_text().splitlines():
            if line.strip():
                labels[CLASSES[int(line.split()[0])]] += 1
    assert manifest == samples == labels
    assert len(rows) == 460 and set(manifest) == set(CLASSES)
    record_property("calibration_gt_counts", json.dumps(dict(manifest), sort_keys=True))


@pytest.mark.dataset
def test_real_ap50_recomputed_with_101_points(step3_artifact_dir, real_dataset_root, record_property):
    # Compare the report's 101-point trapezoidal convention. The step-envelope
    # average is a different AP definition; retain both in the recorded evidence.
    rows = read_rows(step3_artifact_dir / "preds_calibration.jsonl")
    gt = [r for r in read_rows(real_dataset_root / "benchmarks/deeppcb/manifests/samples.jsonl")
          if r["split"] == "calibration"]
    computed = ap50_101(rows, gt, CLASSES)
    report = read_json(step3_artifact_dir / "calibration_per_class.json")
    record_property("independent_ap50", json.dumps(computed, sort_keys=True))
    deltas = {name: computed["per_class"][name]["ap50_linear101"] - report["per_class"][name]["ap50"] for name in CLASSES}
    deltas["overall"] = computed["overall_map50_linear101"] - report["overall"]["map50"]
    assert all(abs(delta) <= 0.02 for delta in deltas.values()), deltas


@pytest.mark.dataset
def test_real_prediction_partitions_do_not_include_test(step3_artifact_dir, real_dataset_root):
    # Protect plan §1.2/§11 and Phase C §1/4: test metadata only, never decode or predict a test image.
    manifest = read_rows(real_dataset_root / "benchmarks/deeppcb/manifests/samples.jsonl")
    held_out = [r for r in manifest if r["split"] == "test"]
    assert len(held_out) == 440
    for part in ("calibration", "fusion"):
        rows = read_rows(step3_artifact_dir / f"preds_{part}.jsonl")
        for prediction_key, source_key in (("sample_id", "sample_id"), ("image_sha256", "sha256"), ("source_group", "source_group")):
            assert not {r[prediction_key] for r in rows} & {r[source_key] for r in held_out}
    meta = read_json(step3_artifact_dir / "artifact.json")
    run = read_json(resolve_recorded_path(meta["run_manifest"]["path"], step3_artifact_dir))
    assert run["partitions_used"]["val"] == "calibration"
    assert not {"test", "fusion"} & run["partitions_used"].keys()
    assert "test" not in json.dumps(run["args_used"]).replace("\\", "/").split("/")


@pytest.fixture
def pure_reload(step3_artifact_dir, real_dataset_root, tmp_path):
    manifest = read_rows(real_dataset_root / "benchmarks/deeppcb/manifests/samples.jsonl")
    samples = sorted((r for r in manifest if r["split"] == "calibration"), key=lambda r: r["sample_id"])[:5]
    meta = read_json(step3_artifact_dir / "artifact.json")
    request = {"artifact_dir": str(step3_artifact_dir), "infer_params": meta["inference_params"],
               "samples": [{"sample_id": r["sample_id"], "path": str(real_dataset_root / r["image"])} for r in samples]}
    request_path, output = tmp_path / "request.json", tmp_path / "pure.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    script = Path(__file__).parent / "helpers/step3_pure_reload.py"
    proc = subprocess.run([sys.executable, str(script), str(request_path), str(output)],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return read_json(output), samples


@pytest.mark.dataset
def test_real_pytorch_checkpoints_load_without_pcb_lab(pure_reload):
    # Protect Phase C §2: isolated Ultralytics load of both .pt files and exact canonical class order.
    result, _ = pure_reload
    assert result["pcb_lab_imported"] is False
    for ckpt in result["checkpoints"].values():
        assert ckpt["class_order"] == CLASSES
        assert ckpt["model_type"] == "DetectionModel"


@pytest.mark.dataset
def test_real_adapter_matches_pure_path_on_five_calibration_images(step3_artifact_dir, real_dataset_root, pure_reload, record_property):
    # Protect Phase C §2 and plan §8 RGB/coordinates: <=0.5 px and <=1e-3 confidence on five real images.
    from pcb_lab.data.samples import ManifestDataset
    from pcb_lab.models.yolo.adapter import YoloDetector
    result, chosen = pure_reload
    wanted = {r["sample_id"] for r in chosen}
    detector = YoloDetector.from_artifact(step3_artifact_dir)
    coord_error = confidence_error = 0.0
    count = 0
    for sample in ManifestDataset(real_dataset_root, "calibration"):
        if sample.sample_id not in wanted:
            continue
        count += 1
        actual = detector.predict(sample.load_image())
        expected = result["predictions"][sample.sample_id]
        assert len(actual) == len(expected), sample.sample_id
        remaining = list(expected)
        for detection in actual:
            candidates = [(max(abs(a-b) for a,b in zip(detection.xyxy_original, row[:4])),
                           abs(detection.confidence-row[4]), index) for index,row in enumerate(remaining)
                          if int(row[5]) == detection.class_id]
            assert candidates, sample.sample_id
            box_delta, conf_delta, index = min(candidates)
            assert box_delta <= 0.5 and conf_delta <= 1e-3, (sample.sample_id, box_delta, conf_delta)
            coord_error = max(coord_error, box_delta)
            confidence_error = max(confidence_error, conf_delta)
            remaining.pop(index)
    assert count == 5
    record_property("adapter_max_xyxy_delta", coord_error)
    record_property("adapter_max_conf_delta", confidence_error)
