"""Verifier-owned Step 4 expectations. No implementation imports or copied algorithms.

Counts below are acceptance assertions from the frozen contract, never production
constants. Synthetic images belong only to train/calibration/fusion; test is metadata.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODEL_ID = "B03_efficientad_s"
INFER = dict(tile_size=256, stride=224, stitch_method="linear", score_metric="max",
             tau_pixel=0.5, morph_k=3, min_area=20, nms_iou=0.45,
             box_class_name="anomaly_unclassified")
CONFIG = dict(
    model_id=MODEL_ID, recipe_id="efficientad_recipe_v1", architecture="efficientad_s",
    model_size="small", pretrained_teacher="artifacts/pretrained/efficientad_teacher_s.pt",
    train=dict(tile_size=256, stride=224, batch_size=1, epochs=70, lr=None,
               weight_decay=None, optimizer="AdamW", seed=42, deterministic=True, device=None),
    normalization=dict(teacher_stats_split="train", map_norm_split="calibration",
                       quantile_min=0.90, quantile_max=0.995),
    infer=INFER,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_hash(value):
    assert isinstance(value, str) and len(value) == 64
    assert all(char in "0123456789abcdef" for char in value)


def assert_number(value, nonnegative=False):
    assert type(value) in (int, float) and math.isfinite(value), value
    if nonnegative:
        assert value >= 0, value


def assert_named_error(error, name):
    # The contract names exceptions, but does not prescribe their module.
    assert name in {base.__name__ for base in type(error).__mro__}, repr(error)


def field(value, key):
    if isinstance(value, dict):
        return value[key]
    # a385e16: Detection object uses xyxy_original; JSONL boxes use xyxy.
    return getattr(value, "xyxy_original" if key == "xyxy" else key)


def assert_box(box):
    assert field(box, "class_name") == "anomaly_unclassified"
    if not isinstance(box, dict):
        assert field(box, "class_id") is None
    xyxy = field(box, "xyxy")
    assert len(xyxy) == 4
    assert all(math.isfinite(float(v)) for v in xyxy)
    x1, y1, x2, y2 = xyxy
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640
    assert math.isfinite(float(field(box, "confidence")))


def assert_config(config):
    assert isinstance(config, dict)
    assert set(config) == set(CONFIG)
    for key in ("model_id", "recipe_id", "architecture", "model_size"):
        assert config[key] == CONFIG[key]
    assert isinstance(config["pretrained_teacher"], str) and config["pretrained_teacher"]
    for section in ("train", "normalization", "infer"):
        assert set(config[section]) == set(CONFIG[section])
        for key, expected in CONFIG[section].items():
            if section == "train" and key in ("lr", "weight_decay", "device"):
                # null is valid in the UNAPPROVED template, never runnable.
                value = config[section][key]
                if value is not None and key != "device":
                    assert_number(value, nonnegative=True)
                    if key == "lr":
                        assert value > 0
                elif value is not None:
                    assert type(value) in (str, int)
            else:
                assert config[section][key] == expected, (section, key)


def write_config(path, mutate=None):
    import yaml
    config = copy.deepcopy(CONFIG)
    # Synthetic values only for negative validation tests, not recipe approval.
    config["train"].update(lr=1e-4, weight_decay=1e-5, device="cpu")
    if mutate:
        mutate(config)
    Path(path).write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def assert_schema_version(value):
    assert type(value) in (int, str) and bool(value)


def assert_artifact(meta):
    required = {"schema_version", "model_id", "seed", "smoke", "checkpoint_sha256",
                "config_hash", "preprocessing_version", "tile_params", "normalization_stats",
                "inference_params", "run_manifest", "created_by_git_commit"}
    assert required <= meta.keys()
    assert_schema_version(meta["schema_version"])
    assert meta["model_id"] == MODEL_ID
    assert type(meta["seed"]) is int and type(meta["smoke"]) is bool
    for key in ("checkpoint_sha256", "config_hash"):
        assert_hash(meta[key])
    for key in ("preprocessing_version", "created_by_git_commit"):
        assert isinstance(meta[key], str) and meta[key]
    for key in ("tile_params", "normalization_stats"):
        assert isinstance(meta[key], dict) and meta[key]
    assert_normalization(meta["normalization_stats"])
    assert isinstance(meta["inference_params"], dict)
    for key, value in INFER.items():
        assert meta["inference_params"][key] == value, key
    assert isinstance(meta["run_manifest"]["path"], str) and meta["run_manifest"]["path"]
    assert_hash(meta["run_manifest"]["sha256"])


def assert_normalization(stats):
    assert {"source_split", "n_good_images", "quantile_min", "quantile_max", "mean", "std"} <= stats.keys()
    assert stats["source_split"] == "calibration"
    assert type(stats["n_good_images"]) is int and stats["n_good_images"] == 230
    for key in ("quantile_min", "quantile_max", "mean", "std"):
        assert type(stats[key]) is float
        assert_number(stats[key])
    assert stats["quantile_min"] <= stats["quantile_max"] and stats["std"] >= 0


def detector_meta():
    return dict(model_id=MODEL_ID, checkpoint_sha256="c" * 64,
                normalization_stats=dict(source_split="calibration", n_good_images=230,
                                         quantile_min=0.0, quantile_max=1.0, mean=0.0, std=1.0),
                inference_params=copy.deepcopy(INFER),
                tile_params=dict(tile_size=256, stride=224, pad_mode="reflect", stitch_method="linear"),
                preprocessing_version="prep_v1", input_color="rgb")


class FakeEngine:
    """Only the engine protocol authorized by contract a385e16, no torch dependency."""
    def __init__(self, maps):
        self.maps = np.asarray(maps, dtype=np.float32)
        self.inputs = []

    def infer_tiles(self, tiles_uint8_nhwc):
        assert isinstance(tiles_uint8_nhwc, np.ndarray)
        assert tiles_uint8_nhwc.dtype == np.uint8
        assert tiles_uint8_nhwc.shape == (9, 256, 256, 3)
        self.inputs.append(tiles_uint8_nhwc.copy())
        return self.maps.copy()


def assert_run(run):
    required = {"schema_version", "run_id", "model_id", "architecture", "seed", "smoke",
                "config_hash", "recipe_id", "git_commit", "git_dirty", "anomalib_version",
                "torch_version", "cuda_version", "device", "python", "dataset_release_sha256",
                "partitions_used", "normal_only_verified", "args_used", "epochs_run",
                "train_time_s", "peak_vram_mb", "peak_ram_mb", "checkpoint_sha256"}
    assert required <= run.keys()
    assert_schema_version(run["schema_version"])
    assert run["model_id"] == MODEL_ID and run["architecture"] == "efficientad_s"
    assert run["recipe_id"] == "efficientad_recipe_v1"
    assert type(run["seed"]) is int and type(run["smoke"]) is bool
    assert type(run["git_dirty"]) is bool
    assert run["normal_only_verified"] is True
    for key in ("config_hash", "dataset_release_sha256", "checkpoint_sha256"):
        assert_hash(run[key])
    for key in ("run_id", "git_commit", "torch_version", "python", "device"):
        assert isinstance(run[key], str) and run[key]
    for key in ("anomalib_version", "cuda_version"):
        assert run[key] is None or isinstance(run[key], str) and run[key]
    assert isinstance(run["args_used"], dict)
    assert type(run["epochs_run"]) is int and run["epochs_run"] > 0
    for key in ("train_time_s", "peak_ram_mb", "peak_vram_mb"):
        assert_number(run[key], nonnegative=True)
    partitions = run["partitions_used"]
    assert partitions["val"] == "calibration"
    assert not {"test", "fusion"} & partitions.keys()
    if not run["smoke"]:
        assert partitions["train"] == 895
        assert partitions["n_good"] == partitions["n_defect"] == 230
        assert all(type(partitions[k]) is int for k in ("train", "n_good", "n_defect"))


def assert_calibration(stats):
    assert {"split", "n_good", "n_defect", "good_scores", "defect_scores", "source"} <= stats.keys()
    assert stats["split"] == "calibration"
    assert stats["n_good"] == stats["n_defect"] == 230
    assert type(stats["n_good"]) is int and type(stats["n_defect"]) is int
    assert isinstance(stats["source"], str) and stats["source"]
    for population in ("good_scores", "defect_scores"):
        values = stats[population]
        order = ("min", "p50", "p90", "p95", "p99", "max")
        assert {*order, "mean", "std"} <= values.keys()
        for key in (*order, "mean", "std"):
            assert_number(values[key])
        assert values["std"] >= 0
        assert values["min"] <= values["mean"] <= values["max"]
        assert [values[k] for k in order] == sorted(values[k] for k in order)


def resolve_manifest(root, meta):
    path = Path(meta["run_manifest"]["path"])
    if path.is_absolute():
        assert path.is_file(), path
        return path
    # The contract permits recorded repo-relative or artifact-relative paths.
    for parent in (Path(root), *Path(root).parents):
        candidate = parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise AssertionError(f"Cannot resolve run manifest: {path}")


def assert_prediction(row, record, meta, run):
    required = {"run_id", "sample_id", "image_sha256", "split", "source_group", "model_id",
                "checkpoint_sha256", "config_hash", "preprocessing_version", "tile_params",
                "anomaly_score", "boxes", "error_reason", "timing_ms"}
    assert required <= row.keys()
    for key in ("sample_id", "source_group", "split"):
        assert row[key] == record[key]
    assert row["image_sha256"] == record["sha256"]
    assert row["run_id"] == run["run_id"]
    for key in ("model_id", "checkpoint_sha256", "config_hash", "preprocessing_version", "tile_params"):
        assert row[key] == meta[key]
    assert row["timing_ms"] is None
    assert isinstance(row["boxes"], list)
    for box in row["boxes"]:
        assert_box(box)
    if row["error_reason"] is None:
        assert_number(row["anomaly_score"])
    else:
        assert isinstance(row["error_reason"], str) and row["error_reason"]
        assert row["anomaly_score"] is None, "Failed inference cannot become a GOOD/zero score"
        assert row["boxes"] == []


def build_dataset(root, counts=None):
    """Fast real PNG fixtures; no file is created/read/decoded for split test."""
    from io import BytesIO
    from PIL import Image
    from helpers.dataset_builder import DatasetFixture, GROUPS, _sample_record

    counts = counts or {"train": 4, "calibration": 4, "fusion": 4}
    fixture = DatasetFixture(root)
    for split, count in counts.items():
        assert split in ("train", "calibration", "fusion")
        for index in range(count):
            sid = f"{split}_{index:04d}"
            stream = BytesIO()
            # Every image is distinguishable, while synthetic generation stays cheap.
            Image.new("RGB", (640, 640), (index % 256, index // 256, len(split))).save(stream, format="PNG")
            args = (sid, split, GROUPS[split][index % 2], sid)
            if index % 2:
                fixture.add_defect_sample(*args, boxes=[dict(class_id=0, class_name="open_circuit",
                                                           xyxy=[20, 30, 50, 60])], png_bytes=stream.getvalue())
            else:
                fixture.add_good_sample(*args, png_bytes=stream.getvalue())
    fixture.records.append(_sample_record("test_metadata_only", "test", GROUPS["test"][0],
        "test_pair", False, "DO_NOT_OPEN_TEST.png", "DO_NOT_OPEN_TEST.txt", "f" * 64,
        "e" * 64, "d" * 64, []))
    fixture.write_classes()
    fixture.write_protocol()
    fixture.write_holdout()
    fixture.write_manifest()
    fixture.write_release()
    return fixture


def make_extraction_artifact(root):
    """Metadata for extraction tests only; never pretend this is a model checkpoint."""
    root.mkdir(parents=True)
    (root / "model.pt").write_bytes(b"VERIFIER_EXTRACTION_STUB_NOT_A_CHECKPOINT")
    run = dict(run_id="step4-synthetic-extraction", model_id=MODEL_ID)
    write_json(root / "run_manifest.json", run)
    meta = detector_meta()
    meta.update(schema_version=1, seed=42, smoke=False,
                checkpoint_sha256=sha256(root / "model.pt"), config_hash="a" * 64,
                run_manifest=dict(path=str(root / "run_manifest.json"),
                                  sha256=sha256(root / "run_manifest.json")), created_by_git_commit="b" * 40)
    write_json(root / "artifact.json", meta)
    return root, meta, run


def tile_maps_from_global(heatmap):
    # Independent locked coordinates, not production split()/stitch().
    canvas = np.pad(np.asarray(heatmap, dtype=np.float32), ((0, 64), (0, 64)), mode="edge")
    return np.stack([canvas[y:y + 256, x:x + 256] for y in (0, 224, 448) for x in (0, 224, 448)])
