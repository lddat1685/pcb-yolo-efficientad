"""Strict B03 recipe validation and seed-independent canonical hashing."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Incomplete, unknown or incompatible recipe setting."""


SCHEMA = {
    "model_id": None, "recipe_id": None, "architecture": None,
    "model_size": None, "pretrained_teacher": None,
    "train": {k: None for k in (
        "tile_size", "stride", "batch_size", "epochs", "lr", "weight_decay",
        "optimizer", "seed", "deterministic", "device")},
    "normalization": {k: None for k in (
        "teacher_stats_split", "map_norm_split", "quantile_min", "quantile_max")},
    "infer": {k: None for k in (
        "tile_size", "stride", "stitch_method", "score_metric", "tau_pixel",
        "morph_k", "min_area", "nms_iou", "box_class_name")},
}


def _keys(value, schema, prefix=""):
    if not isinstance(value, dict):
        raise ConfigError(f"{prefix or 'config'} must be a mapping")
    if set(value) != set(schema):
        raise ConfigError(f"{prefix}: unknown={set(value)-set(schema)}, missing={set(schema)-set(value)}")
    for key, child in schema.items():
        if child is not None:
            _keys(value[key], child, prefix + key + ".")


def config_hash(config):
    def strip_seed(value):
        if isinstance(value, dict):
            return {k: strip_seed(v) for k, v in value.items() if k != "seed"}
        return value
    return hashlib.sha256(json.dumps(strip_seed(config), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path, *, seed=42, device=None, smoke=False):
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(str(exc)) from exc
    _keys(cfg, SCHEMA)
    cfg = deepcopy(cfg)
    train, norm, infer = cfg["train"], cfg["normalization"], cfg["infer"]
    for section, locked in (
        (cfg, {"model_id": "B03_efficientad_s", "architecture": "efficientad_s", "model_size": "small"}),
        (train, {"tile_size": 256, "stride": 224, "batch_size": 1, "optimizer": "AdamW", "deterministic": True}),
        (norm, {"teacher_stats_split": "train", "map_norm_split": "calibration"}),
        (infer, {"tile_size": 256, "stride": 224, "stitch_method": "linear", "score_metric": "max",
                 "morph_k": 3, "nms_iou": 0.45, "box_class_name": "anomaly_unclassified"}),
    ):
        for key, expected in locked.items():
            if type(section[key]) is not type(expected) or section[key] != expected:
                raise ConfigError(f"{key} is locked to {expected!r}")
    for section, keys in ((train, ("lr", "weight_decay")), (norm, ("quantile_min", "quantile_max")),
                          (infer, ("tau_pixel",))):
        for key in keys:
            if type(section[key]) not in (float, int) or not math.isfinite(section[key]):
                raise ConfigError(f"{key} must be a finite number, not null")
            section[key] = float(section[key])
    if train["lr"] <= 0 or train["weight_decay"] < 0:
        raise ConfigError("lr must be positive and weight_decay nonnegative")
    if not 0 <= norm["quantile_min"] < norm["quantile_max"] <= 1:
        raise ConfigError("Invalid normalization quantile probabilities")
    for value, name, minimum in ((train["epochs"], "epochs", 1), (infer["min_area"], "min_area", 0),
                                 (train["seed"], "train.seed", 0), (seed, "seed", 0)):
        if type(value) is not int or value < minimum:
            raise ConfigError(f"{name} must be an integer >= {minimum}")
    for key in ("recipe_id", "pretrained_teacher"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ConfigError(f"{key} must be nonempty")
    selected = device if device is not None else ("cpu" if smoke else train["device"])
    if selected is None or isinstance(selected, bool):
        raise ConfigError("Specify train.device or --device (cpu or CUDA index)")
    selected = str(selected).lower()
    if selected.isdigit():
        selected = "cuda:" + selected
    if selected != "cpu" and selected != "cuda" and not (selected.startswith("cuda:") and selected[5:].isdigit()):
        raise ConfigError("Only cpu or cuda devices are supported")
    train.update(seed=seed, device=selected, epochs=1 if smoke else train["epochs"])
    return cfg
