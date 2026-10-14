"""Independent Step 3 oracles. Never import YOLO implementation at collection."""

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

CLASSES = ["open_circuit", "short", "mouse_bite", "spur", "spurious_copper", "pin_hole"]
INFER = {"conf_floor": 0.001, "iou": 0.7, "max_det": 300, "imgsz": 640,
         "agnostic_nms": False, "half": False}
PREDICTION_KEYS = set("run_id sample_id image_sha256 split source_group model_id "
                      "checkpoint_sha256 config_hash preprocessing_version inference_params "
                      "detections yolo_image_score error_reason timing_ms".split())
ARTIFACT_KEYS = set("schema_version model_id seed smoke checkpoint_best_sha256 "
                    "checkpoint_last_sha256 class_order config_hash preprocessing_version "
                    "preprocessing_hash inference_params view_signature run_manifest "
                    "created_by_git_commit".split())
RUN_KEYS = set("schema_version run_id model_id architecture seed smoke config_hash recipe_id "
               "git_commit git_dirty ultralytics_version torch_version cuda_version device python "
               "dataset_release_sha256 view_signature link_mode partitions_used pretrained args_used "
               "epochs_run best_epoch early_stopped train_time_s peak_vram_mb peak_ram_mb "
               "checkpoints resumed_from".split())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(root):
    path = Path(root) / "benchmarks/deeppcb/manifests/samples.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def class_order(root):
    return read_json(Path(root) / "benchmarks/deeppcb/configs/classes.json")["names"]


def field(value, name):
    # Contract specifies fields, not dict versus dataclass representation.
    return value[name] if isinstance(value, dict) else getattr(value, name)


def dataset_snapshot(root):
    """Inventory *all* entries including caches, hashing everything except test images.

    os.walk is solely for detecting writes, never for selecting dataset samples.
    Test images are stat-only: WORKFLOW forbids opening them even for raw hashing.
    Access time/link count are excluded because read/hardlink legitimately changes them.
    """
    root = Path(root).resolve()
    all_rows = records(root)
    held_out = {(root / row[key]).resolve() for row in all_rows if row["split"] == "test"
                for key in ("image", "source_image") if row.get(key)}
    allowed_images = {(root / row["image"]).resolve() for row in all_rows if row["split"] != "test"}
    snapshot = {}
    for parent, dirs, files in os.walk(root):
        for name in sorted(dirs + files):
            path = Path(parent) / name
            stat = path.stat()
            unknown_image = (path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                             and path.resolve() not in allowed_images)
            digest = None if path.is_dir() or path.resolve() in held_out or unknown_image else sha256(path)
            snapshot[path.relative_to(root).as_posix()] = (
                path.is_dir(), stat.st_size if path.is_file() else None, stat.st_mtime_ns, digest
            )
    return snapshot


def assert_sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), value


def assert_number(value, minimum=0):
    assert type(value) in (int, float) and math.isfinite(value) and value >= minimum, value


def assert_measurement(value):
    assert isinstance(value, dict) and {"value", "method"} <= value.keys()
    assert isinstance(value["method"], str) and value["method"].strip()
    if value["value"] is not None:
        assert_number(value["value"])


def assert_run_schema(run):
    assert RUN_KEYS <= run.keys(), RUN_KEYS - run.keys()
    for key in ("run_id", "model_id", "architecture", "recipe_id", "git_commit",
                "ultralytics_version", "torch_version", "python", "link_mode"):
        assert isinstance(run[key], str) and run[key], key
    # schema_version's concrete scalar type is not fixed by the contract.
    assert type(run["schema_version"]) in (str, int)
    assert type(run["seed"]) is int
    for key in ("smoke", "git_dirty", "early_stopped"):
        assert type(run[key]) is bool, key
    for key in ("config_hash", "dataset_release_sha256", "view_signature"):
        assert_sha(run[key])
    assert run["cuda_version"] is None or isinstance(run["cuda_version"], str)
    assert type(run["device"]) in (str, int, list)
    for key in ("args_used", "partitions_used", "pretrained", "checkpoints"):
        assert isinstance(run[key], dict), key
    used = run["partitions_used"]
    assert used["val"] == "calibration" and type(used["train"]) is int
    assert used["train"] > 0 and "test" not in used and "fusion" not in used
    # Contract writes partitions_used{train:n, val:"calibration", n}; count key unresolved.
    assert {"path", "sha256", "source"} <= run["pretrained"].keys()
    if not run["smoke"]:
        for key in ("path", "source"):
            assert isinstance(run["pretrained"][key], str) and run["pretrained"][key]
        assert_sha(run["pretrained"]["sha256"])
    for key in ("epochs_run", "best_epoch"):
        assert type(run[key]) is int and run[key] >= 0
    assert run["epochs_run"] >= 1
    assert_number(run["train_time_s"])
    assert_measurement(run["peak_vram_mb"])
    # RAM shape is unspecified; accept measured scalar or value/method object.
    if isinstance(run["peak_ram_mb"], dict):
        assert_measurement(run["peak_ram_mb"])
    elif run["peak_ram_mb"] is not None:
        assert_number(run["peak_ram_mb"])
    # Null RAM's measurement method is checked during C review: contract does not name its key.
    assert run["resumed_from"] is None or isinstance(run["resumed_from"], str)
    for key in ("best_sha256", "last_sha256"):
        assert_sha(run["checkpoints"][key])


def assert_artifact_schema(meta):
    assert ARTIFACT_KEYS <= meta.keys(), ARTIFACT_KEYS - meta.keys()
    assert type(meta["schema_version"]) in (str, int)
    assert type(meta["seed"]) is int and type(meta["smoke"]) is bool
    assert meta["class_order"] == CLASSES
    for key in ("model_id", "preprocessing_version", "created_by_git_commit"):
        assert isinstance(meta[key], str) and meta[key]
    for key in ("checkpoint_best_sha256", "checkpoint_last_sha256", "config_hash",
                "preprocessing_hash", "view_signature"):
        assert_sha(meta[key])
    assert isinstance(meta["inference_params"], dict)
    assert INFER.keys() <= meta["inference_params"].keys()
    assert type(meta["inference_params"]["max_det"]) is int
    assert type(meta["inference_params"]["imgsz"]) is int
    for key in ("agnostic_nms", "half"):
        assert type(meta["inference_params"][key]) is bool
    for key in ("conf_floor", "iou"):
        assert_number(meta["inference_params"][key])
        assert meta["inference_params"][key] <= 1
    assert isinstance(meta["run_manifest"], dict)
    assert isinstance(meta["run_manifest"]["path"], str)
    assert_sha(meta["run_manifest"]["sha256"])


def assert_error_name(expected, call):
    # Exception module is not fixed; require the exact contracted class name.
    import pytest
    with pytest.raises(Exception) as caught:
        call()
    assert type(caught.value).__name__ == expected, repr(caught.value)


def approved_test_config():
    """Test-only explicit values; NOT approval of the production recipe."""
    return {
        "model_id": "B01_yolo11n", "recipe_id": "yolo_recipe_v1", "architecture": "yolo11n",
        "pretrained": "artifacts/pretrained/yolo11n.pt",
        "train": {"imgsz": 640, "epochs": 100, "patience": 20, "optimizer": "AdamW",
                  "lr0": 0.001, "batch": 2, "workers": 0, "deterministic": True,
                  "augment": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                              "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                              "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
                              "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0,
                              "copy_paste": 0.0}},
        "selection": {"split": "calibration", "metric": "ultralytics_fitness"},
        "infer": dict(INFER), "classes_ref": "benchmarks/deeppcb/configs/classes.json",
    }


def write_config(directory, value):
    import yaml
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Two files are the only configuration composition described by the contract.
    (directory / "yolo_common.yaml").write_text("{}\n", encoding="utf-8")
    path = directory / "yolo11n.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def reverse_keys(value):
    if isinstance(value, dict):
        return {key: reverse_keys(value[key]) for key in reversed(value)}
    return value


def detector_meta():
    # Exact schema approved by the supervisor during Phase A.
    return {"model_id": "B01_yolo11n", "checkpoint_sha256": "a" * 64,
            "class_order": list(CLASSES), "infer_params": dict(INFER), "input_color": "rgb"}


class FakeEngine:
    """Contract Engine.infer + supervisor-approved class_names, no torch dependency."""
    def __init__(self, detections=(), class_names=None, check=None):
        import numpy as np
        self.detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        self.class_names = list(CLASSES if class_names is None else class_names)
        self.check = check
        self.calls = 0

    def load(self):
        return self

    def infer(self, pixels):
        import numpy as np
        assert pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
        self.calls += 1
        if self.check:
            self.check(pixels)
        return self.detections.copy()


def detection_record(det):
    return {"class_id": field(det, "class_id"), "class_name": field(det, "class_name"),
            "confidence": float(field(det, "confidence")),
            "xyxy_original": [float(v) for v in field(det, "xyxy_original")]}


def fake_checkpoint_loader(monkeypatch, class_names=None):
    """B correction: from_artifact loads YOLO before constructing UltralyticsEngine.

    Replace only deserialization of synthetic bytes; retain real hash/class/smoke guards.
    Source evidence: adapter.py:119-138. The constructor signature was not fixed in A.
    """
    from types import SimpleNamespace
    import ultralytics
    names = list(CLASSES if class_names is None else class_names)
    model = SimpleNamespace(names=dict(enumerate(names)))
    monkeypatch.setattr(ultralytics, "YOLO", lambda *args, **kwargs: SimpleNamespace(model=model, names=model.names))


def resolve_recorded_path(value, anchor):
    """Contract does not fix relative-path anchor; require one unambiguous existing file."""
    value, anchor = Path(value), Path(anchor).resolve()
    if value.is_absolute():
        assert value.is_file(), value
        return value
    candidates = {(parent / value).resolve() for parent in [anchor, *anchor.parents]
                  if (parent / value).is_file()}
    assert len(candidates) == 1, (value, candidates)
    return candidates.pop()


def assert_prediction_row(row, sample, meta, run):
    assert PREDICTION_KEYS <= row.keys(), PREDICTION_KEYS - row.keys()
    assert row["sample_id"] == sample.sample_id and row["image_sha256"] == sample.sha256
    assert row["source_group"] == sample.source_group and row["split"] == sample.partition
    assert row["run_id"] == run["run_id"] and row["model_id"] == meta["model_id"]
    assert row["checkpoint_sha256"] == meta["checkpoint_best_sha256"]
    assert row["config_hash"] == meta["config_hash"]
    assert row["preprocessing_version"] == meta["preprocessing_version"]
    assert row["inference_params"] == meta["inference_params"]
    assert row["timing_ms"] is None and isinstance(row["detections"], list)
    if row["error_reason"] is not None:
        assert isinstance(row["error_reason"], str) and row["error_reason"].strip()
        # Contract requires an explicit error, but does not prescribe the failed score's type.
        assert row.get("final_status") != "GOOD"
        return
    scores = []
    for det in row["detections"]:
        assert {"class_id", "class_name", "confidence", "xyxy_original"} <= det.keys()
        assert type(det["class_id"]) is int and 0 <= det["class_id"] < len(CLASSES)
        assert det["class_name"] == CLASSES[det["class_id"]]
        assert_number(det["confidence"])
        assert meta["inference_params"]["conf_floor"] <= det["confidence"] <= 1
        x1, y1, x2, y2 = det["xyxy_original"]
        assert 0 <= x1 < x2 <= sample.width and 0 <= y1 < y2 <= sample.height
        scores.append(det["confidence"])
    assert row["yolo_image_score"] == max(scores, default=0.0)
    # JSONL rounds scores to six decimals, so equal serialized scores do not
    # imply equal original scores. C replay proved four such ties in B01 fusion.
    # Exact class/coordinate tie-breaking remains covered on raw adapter output.
    assert scores == sorted(scores, reverse=True)


def make_fake_artifact(directory, dataset_root):
    """Metadata-complete synthetic artifact. Checkpoint bytes are deliberately NOT torch files."""
    import numpy as np
    from pcb_lab.inference.preprocessing import prepare_input
    directory = Path(directory)
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"verifier-fake-best-weights\x00")
    (directory / "last.pt").write_bytes(b"verifier-fake-last-weights\x00")
    prep = prepare_input(np.zeros((640, 640, 3), dtype=np.uint8), "yolo").meta
    partition_counts = Counter(row["split"] for row in records(dataset_root))
    run = {
        "schema_version": 1, "run_id": "verifier-synthetic-run", "model_id": "B01_yolo11n",
        "architecture": "yolo11n", "seed": 42, "smoke": False, "config_hash": "b" * 64,
        "recipe_id": "yolo_recipe_v1", "git_commit": "c" * 40, "git_dirty": False,
        "ultralytics_version": "synthetic", "torch_version": "synthetic", "cuda_version": None,
        "device": "cpu", "python": "synthetic", "dataset_release_sha256": sha256(
            Path(dataset_root) / "benchmarks/deeppcb/release.json"),
        "view_signature": "d" * 64, "link_mode": "copy",
        "partitions_used": {"train": partition_counts["train"], "val": "calibration",
                            "n": partition_counts["calibration"]},
        "pretrained": {"path": "synthetic.pt", "sha256": "e" * 64, "source": "test fixture"},
        "args_used": {}, "epochs_run": 1, "best_epoch": 0, "early_stopped": False,
        "train_time_s": 1.0, "peak_vram_mb": {"value": None, "method": "CPU fixture"},
        "peak_ram_mb": {"value": None, "method": "fixture; not measured"},
        "checkpoints": {"best_sha256": sha256(directory / "best.pt"),
                        "last_sha256": sha256(directory / "last.pt")}, "resumed_from": None,
    }
    write_json(directory / "run_manifest.json", run)
    meta = {
        "schema_version": 1, "model_id": run["model_id"], "seed": 42, "smoke": False,
        "checkpoint_best_sha256": run["checkpoints"]["best_sha256"],
        "checkpoint_last_sha256": run["checkpoints"]["last_sha256"], "class_order": list(CLASSES),
        "config_hash": run["config_hash"], "preprocessing_version": prep["preprocessing_version"],
        "preprocessing_hash": prep["preprocessing_hash"], "inference_params": dict(INFER),
        "view_signature": run["view_signature"], "run_manifest": {
            "path": str((directory / "run_manifest.json").resolve()),
            "sha256": sha256(directory / "run_manifest.json")},
        "created_by_git_commit": run["git_commit"],
    }
    write_json(directory / "artifact.json", meta)
    return directory


def assert_boxes_close_unordered(actual, expected, tolerance=0.01):
    """Match class/geometry bijectively; rounding must not change a lexicographic pairing.

    B evidence: deeppcb_12100121_defect has two class-0 boxes at x1=261.
    Six-decimal labels give x1=260.99968 and 261.00032, reversing their sorted order.
    """
    assert len(actual) == len(expected)
    candidates = [[j for j, (gt_cls, gt_box) in enumerate(expected)
                   if cls == gt_cls and max(abs(a - b) for a, b in zip(box, gt_box)) <= tolerance]
                  for cls, box in actual]
    assigned = {}
    def match(index, seen):
        for target in candidates[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or match(assigned[target], seen):
                assigned[target] = index
                return True
        return False
    assert all(match(index, set()) for index in range(len(actual))), (actual, expected)


def assert_view(root, out, view):
    """Verify by image bytes, never assume generated image basenames equal sample_id."""
    import yaml
    from pcb_lab.data.samples import ManifestDataset
    root, out = Path(root), Path(out)
    data_path = Path(field(view, "data_yaml"))
    assert data_path.resolve() == (out / "data.yaml").resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    assert {"path", "train", "val", "names"} <= data.keys() and "test" not in data
    names = data["names"]
    if isinstance(names, dict):
        # B correction: YAML string keys "0".."5" and integer keys encode the same class order.
        indexed = {int(key): value for key, value in names.items()}
        assert len(indexed) == len(names) == 6 and set(indexed) == set(range(6))
        names = [indexed[key] for key in range(6)]
    assert names == class_order(root) == CLASSES
    base = Path(data["path"])
    if not base.is_absolute():
        base = data_path.parent / base
    hashes = set()
    all_rows = records(root)
    for partition, yaml_key in (("train", "train"), ("calibration", "val")):
        folder = out / "images" / partition
        assert (base / data[yaml_key]).resolve() == folder.resolve()
        samples = list(ManifestDataset(root, partition))
        rows = {row["sample_id"]: row for row in all_rows if row["split"] == partition}
        by_hash = {sample.sha256: sample for sample in samples}
        assert len(by_hash) == len(samples), "Fixture/real view needs unique file hashes"
        images = list(folder.iterdir())
        assert len(images) == len(samples)
        labels = list((out / "labels" / partition).iterdir())
        assert {p.name for p in labels} == {p.stem + ".txt" for p in images}
        counts = Counter()
        partition_hashes = []
        for image in images:
            digest = sha256(image)
            assert digest in by_hash
            sample = by_hash[digest]
            partition_hashes.append(digest)
            label = out / "labels" / partition / (image.stem + ".txt")
            raw = label.read_text(encoding="utf-8")
            lines = raw.splitlines()
            assert len(lines) == len(sample.boxes) == len(rows[sample.sample_id]["boxes"])
            if not sample.is_defect:
                assert label.read_bytes() == b""
            expected = sorted((b.class_id, tuple(b.xyxy)) for b in sample.boxes)
            source = sorted((b["class_id"], tuple(b["xyxy"])) for b in rows[sample.sample_id]["boxes"])
            assert expected == source, "Sample.boxes differs from independent manifest oracle"
            actual = []
            for line in lines:
                parts = line.split()
                assert len(parts) == 5 and parts[0] in {str(i) for i in range(6)}
                assert all(re.fullmatch(r"\d+\.\d{6}", token) for token in parts[1:])
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
                actual.append((cls, ((cx - w / 2) * sample.width, (cy - h / 2) * sample.height,
                                     (cx + w / 2) * sample.width, (cy + h / 2) * sample.height)))
                counts[CLASSES[cls]] += 1
            assert_boxes_close_unordered(actual, expected, tolerance=0.01)
        assert len(set(partition_hashes)) == len(samples)
        hashes.update(partition_hashes)
        good = sum(not sample.is_defect for sample in samples)
        assert field(view, "counts")[partition] == {
            "images": len(samples), "good": good, "defect": len(samples) - good}
        reported = field(view, "box_counts")[partition]
        # B correction: contract does not require explicitly storing zero-count classes.
        assert set(reported) <= set(CLASSES)
        assert {name: reported.get(name, 0) for name in CLASSES} == {name: counts[name] for name in CLASSES}
    forbidden = {row["sha256"] for row in all_rows if row["split"] in ("fusion", "test")}
    assert hashes.isdisjoint(forbidden)
    assert_sha(field(view, "view_signature"))
    return hashes
