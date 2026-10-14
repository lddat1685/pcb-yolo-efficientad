"""Canonical metadata adapter. No directory discovery or inferred pairing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath

import yaml

PARTITIONS = ("train", "calibration", "fusion", "test")
BENCHMARK = "benchmarks/deeppcb"
METADATA_PATHS = {
    "manifest": f"{BENCHMARK}/manifests/samples.jsonl",
    "holdout": f"{BENCHMARK}/splits/holdout.json",
    "classes": f"{BENCHMARK}/configs/classes.json",
    "protocol": f"{BENCHMARK}/configs/protocol.json",
    "release": f"{BENCHMARK}/release.json",
}


class DatasetInputError(ValueError):
    """Required metadata cannot be loaded; the CLI returns 2."""


def resolve_dataset_root(explicit: Path | None = None, config: Path | None = None) -> Path:
    """CLI > environment > YAML; default: repo/data, then sibling dataset.

    Relative YAML paths are based on the config's parent directory, never cwd.
    Relative CLI/environment paths retain normal shell semantics.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("DATASET_ROOT"):
        return Path(os.environ["DATASET_ROOT"]).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    config = Path(config) if config is not None else repo / "configs/dataset.yaml"
    try:
        settings = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
        value = settings["DATASET_ROOT"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("DATASET_ROOT must be a nonempty string")
        for key, path in METADATA_PATHS.items():
            if settings.get(key, path) != path:
                raise ValueError(f"{key} must retain the canonical contract path: {path}")
        if value == "auto":
            local = repo / "data"
            return local if local.is_dir() else repo.parent / "DatasetVer4_Public"
        path = Path(value).expanduser()
        return (path if path.is_absolute() else config.resolve().parent / path).resolve()
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise DatasetInputError(f"Cannot read dataset config {config}: {exc}") from exc


def dataset_path(root: Path, relative: str) -> Path:
    """Accept only relative paths resolving inside the read-only dataset."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Dataset path must be a nonempty relative string")
    windows = PureWindowsPath(relative)
    if windows.drive or windows.root or Path(relative).is_absolute():
        raise ValueError(f"Absolute dataset path is not allowed: {relative}")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


def output_path(root: Path, out_dir: Path) -> Path:
    """Check before any writes, including symlink/junction resolution."""
    root, out_dir = root.resolve(), out_dir.resolve()
    if out_dir.is_relative_to(root) or root.is_relative_to(out_dir):
        raise DatasetInputError("Output and dataset directories must not overlap")
    return out_dir


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_path(out_dir: Path, relative: str) -> Path:
    """Reject existing output symlinks/junctions leading outside out_dir."""
    path = (out_dir / relative).resolve()
    if not path.is_relative_to(out_dir.resolve()):
        raise DatasetInputError(f"Output path escapes report directory: {relative}")
    return path


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetInputError(f"Cannot read metadata {path.name}: {exc}") from exc


def load_manifest(dataset_root: Path) -> list[dict]:
    path = dataset_path(dataset_root, METADATA_PATHS["manifest"])
    rows = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise DatasetInputError(f"Invalid manifest JSON at line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    row = {"_schema_error": f"Manifest line {line_number} is not an object"}
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise DatasetInputError(f"Cannot read manifest: {exc}") from exc
    if not rows:
        raise DatasetInputError("Manifest has no samples")
    return sorted(rows, key=lambda r: (str(r.get("sample_id", "")), json.dumps(r, sort_keys=True)))


@dataclass
class Dataset:
    root: Path
    rows: list[dict]
    holdout: dict
    classes: dict
    protocol: dict
    release: dict

    @property
    def names(self) -> list[str]:
        return self.classes["names"]


def load_dataset(dataset_root: Path) -> Dataset:
    root = Path(dataset_root).resolve()
    meta = {key: read_json(dataset_path(root, relative))
            for key, relative in METADATA_PATHS.items() if key != "manifest"}
    holdout, classes, release = meta["holdout"], meta["classes"], meta["release"]
    try:
        if sorted(holdout["partitions"]) != sorted(PARTITIONS):
            raise ValueError("holdout must contain the four canonical partitions exactly once")
        for key in ("samples", "groups", "pair_ids"):
            mapping = holdout[key]
            if set(mapping) != set(PARTITIONS):
                raise ValueError(f"holdout.{key} has incorrect partition keys")
            for partition, values in mapping.items():
                if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                    raise ValueError(f"holdout.{key}.{partition} must be a list of strings")
        if set(holdout["actual_pair_counts"]) != set(PARTITIONS) or any(
            type(v) is not int or v < 0 for v in holdout["actual_pair_counts"].values()
        ):
            raise ValueError("holdout.actual_pair_counts must contain nonnegative integers")
        names = classes["names"]
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n or not all(c.isalnum() or c == "_" for c in n)
                       for n in names)):
            raise ValueError("classes.names must contain unique safe class names")
        if not isinstance(release["files"], dict) or not release["files"]:
            raise ValueError("release.files must be a nonempty path-to-SHA256 map")
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetInputError(f"Invalid metadata schema: {exc}") from exc
    return Dataset(root, load_manifest(root), holdout, classes, meta["protocol"], release)


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def parse_yolo(text: str, names: list[str]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Parse every nonblank line, retaining diagnostics for each bad line."""
    boxes, issues = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.append(("box_valid", f"Label line {number}: expected 5 fields"))
            continue
        try:
            values = [float(v) for v in fields]
        except ValueError:
            issues.append(("box_valid", f"Label line {number}: nonnumeric value"))
            issues.append(("class_id_valid", f"Label line {number}: cannot parse class/coordinates"))
            continue
        c, cx, cy, w, h = values
        class_ok = math.isfinite(c) and c.is_integer() and 0 <= c < len(names)
        if not class_ok:
            issues.append(("class_id_valid", f"Label line {number}: invalid class ID {fields[0]}"))
        xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        geometry_ok = (all(math.isfinite(v) for v in values[1:]) and w > 0 and h > 0
                       and min(xyxy) >= -1e-9 and max(xyxy) <= 1 + 1e-9)
        if not geometry_ok:
            issues.append(("box_valid", f"Label line {number}: nonfinite, nonpositive or out-of-bounds box"))
        if class_ok and geometry_ok:
            boxes.append({"class_id": int(c), "xyxy_normalized": xyxy})
    return boxes, issues
