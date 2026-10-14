"""Immutable metadata samples and manifest-only CPU loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import islice
import json
from pathlib import Path
from numbers import Integral, Real

import numpy as np

from .manifest import PARTITIONS, dataset_path, load_dataset
from pcb_lab.inference.preprocessing import load_preprocessing_config, prepare_input, preprocessing_hash, to_canonical
from pcb_lab.inference.tiling import TileManager


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]

    def __post_init__(self):
        xy = tuple(self.xyxy)
        if (len(xy) != 4 or not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) for v in xy)
                or xy[2] <= xy[0] or xy[3] <= xy[1]):
            raise ValueError("Box requires four finite coordinates and positive area")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral) or self.class_id < 0 or not isinstance(self.class_name, str):
            raise ValueError("Box requires a nonnegative integer class_id and class_name")
        object.__setattr__(self, "xyxy", tuple(float(v) for v in xy))
        object.__setattr__(self, "class_id", int(self.class_id))


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_group: str
    pair_id: str
    partition: str
    image_path: Path
    width: int
    height: int
    is_defect: bool
    boxes: tuple[Box, ...]
    mask_status: str
    sha256: str
    pair_ids: tuple[str, ...] = ()
    source: str = "deeppcb"
    _allow_test: bool = field(default=False, repr=False, compare=False)
    _dataset_root: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "boxes", tuple(self.boxes))
        pairs = tuple(sorted(self.pair_ids or (self.pair_id,)))
        if not pairs or any(not isinstance(p, str) or not p for p in pairs) or self.pair_id != pairs[0]:
            raise ValueError("pair_id must be the lexical first member of pair_ids")
        object.__setattr__(self, "pair_ids", pairs)

    def load_image(self) -> np.ndarray:
        if self.partition == "test" and not self._allow_test:
            raise PermissionError("Test Sample.load_image requires allow_test=True")
        image = to_canonical(self.image_path, dataset_root=self._dataset_root, allow_test=self._allow_test)
        if image.shape != (self.height, self.width, 3):
            raise ValueError(f"{self.sample_id}: decoded size does not match manifest")
        return image


class ManifestDataset:
    def __init__(self, dataset_root, partition, *, allow_test=False):
        partition = "fusion" if partition == "development_selection" else partition
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        if type(allow_test) is not bool:
            raise ValueError("allow_test must be boolean")
        data = load_dataset(Path(dataset_root))
        self.dataset_root, self.partition, self.allow_test = data.root, partition, allow_test
        rows = [r for r in data.rows if r.get("split") == partition]
        ids = [r.get("sample_id") for r in rows]
        if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Manifest must have unique nonempty sample IDs")
        if sorted(ids) != sorted(data.holdout["samples"][partition]):
            raise ValueError(f"{partition}: sample membership differs from holdout")
        groups = set(data.holdout["groups"][partition])
        pairs_allowed = set(data.holdout["pair_ids"][partition])
        samples = []
        for row in sorted(rows, key=lambda r: r["sample_id"]):
            try:
                pairs = row["pair_ids"]
                if (not isinstance(pairs, list) or not pairs
                        or any(not isinstance(p, str) or not p for p in pairs)
                        or len(set(pairs)) != len(pairs) or not set(pairs) <= pairs_allowed):
                    raise ValueError("Invalid pair aliases or membership")
                if row["source_group"] not in groups:
                    raise ValueError("source_group differs from holdout")
                if (type(row["is_defect"]) is not bool or type(row["width"]) is not int
                        or type(row["height"]) is not int or min(row["width"], row["height"]) <= 0):
                    raise ValueError("Invalid image metadata")
                boxes = tuple(Box(b["class_id"], b["class_name"], b["xyxy"]) for b in row["boxes"])
                for box in boxes:
                    if box.class_id >= len(data.names) or data.names[box.class_id] != box.class_name:
                        raise ValueError("Invalid class schema")
                    x1, y1, x2, y2 = box.xyxy
                    if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
                        raise ValueError("Box outside image")
                if bool(boxes) != row["is_defect"]:
                    raise ValueError("Good/defect metadata contradicts boxes")
                # Full aliases remain available: representative pair_id is not an isolation key.
                samples.append(Sample(row["sample_id"], row["source_group"], min(pairs), partition,
                    dataset_path(data.root, row["image"]), row["width"], row["height"], row["is_defect"],
                    boxes, row.get("mask_status", "unavailable"), row["sha256"], tuple(sorted(pairs)),
                    row.get("source", "deeppcb"), allow_test, data.root))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample {row.get('sample_id')}: {exc}") from exc
        self._samples = tuple(samples)

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


class YoloTrainSet(ManifestDataset):
    def __init__(self, root):
        super().__init__(root, "train")


class AdTrainSet(ManifestDataset):
    def __init__(self, root, *, tile_manager=None):
        super().__init__(root, "train")
        self._samples = tuple(sample for sample in self._samples if not sample.is_defect)
        self.tile_manager = tile_manager or TileManager()
        counts = {len(self.tile_manager.plan(s.height, s.width)) for s in self._samples}
        if len(counts) > 1:
            raise ValueError("AdTrainSet requires a uniform tile count per image for num_tiles")
        self.tiles_per_image = next(iter(counts), 0)
        self.num_tiles = len(self) * self.tiles_per_image

    def iter_tiles(self):
        for sample in self:
            tiles, specs = self.tile_manager.split(sample.load_image())
            for tile, spec in zip(tiles, specs):
                yield sample, spec, tile


class EvalSet(ManifestDataset):
    def __init__(self, root, partition, allow_test=False):
        super().__init__(root, partition, allow_test=allow_test)


def check_loaders(dataset_root, cfg=None, *, batch_size=2) -> dict:
    """Shared implementation for the CLI: pixels on development, metadata on test."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    config = load_preprocessing_config(cfg)
    tile = config["ad"]["tile"]
    tm = TileManager(tile["size"], tile["stride"], tile["pad_mode"], tile["global_nms_iou"])
    loaders = {"yolo_train": (YoloTrainSet(dataset_root), "yolo"),
               "ad_train": (AdTrainSet(dataset_root, tile_manager=tm), "ad_tile"),
               "eval_calibration": (EvalSet(dataset_root, "calibration"), "yolo"),
               "eval_fusion": (EvalSet(dataset_root, "fusion"), "yolo"),
               "eval_test": (EvalSet(dataset_root, "test"), "yolo")}
    report = {"schema_version": 1, "preprocessing_version": config["preprocessing_version"],
              "preprocessing_hash": preprocessing_hash(config), "loaders": {}, "errors": []}
    for name, (loader, mode) in loaders.items():
        entry = {"n_samples": len(loader), "batch_size": 0, "shape": None, "dtype": None,
                 "min": None, "max": None, "color_order": "rgb", "normalized": False, "hash": None,
                 "preprocessing_version": config["preprocessing_version"],
                 "preprocessing_hash": report["preprocessing_hash"]}
        report["loaders"][name] = entry
        if loader.partition == "test":
            entry["metadata_only"] = True
            entry["sample_ids"] = [s.sample_id for s in loader]
            entry["access_blocked"] = True
            # This call must fail before Pillow reads even a header.
            for sample in islice(loader, 1):
                try:
                    sample.load_image()
                    entry["access_blocked"] = False
                    report["errors"].append({"loader": name, "message": "Test image gate failed"})
                except PermissionError:
                    pass
            continue
        selected = list(islice(loader, batch_size))
        try:
            if not selected:
                raise ValueError("No sample available for a batch")
            prepared = [prepare_input(s, mode, config) for s in selected]
            pixels = np.concatenate([p.pixels for p in prepared]) if mode == "ad_tile" else np.stack([p.pixels for p in prepared])
            if isinstance(loader, AdTrainSet):
                actual_tiles = list(islice(loader.iter_tiles(), len(selected) * loader.tiles_per_image))
                if not np.array_equal(pixels, np.stack([x[2] for x in actual_tiles])):
                    raise ValueError("AD iter_tiles differs from prepare_input")
                if any(s.partition != "train" or s.is_defect for s, _, _ in actual_tiles):
                    raise ValueError("AD tile isolation failed")
                entry.update(tiles_per_image=loader.tiles_per_image, num_tiles=loader.num_tiles)
            hashes = [p.sha256() for p in prepared]
            entry.update(batch_size=len(selected), shape=list(pixels.shape), dtype=pixels.dtype.name,
                         min=int(pixels.min()), max=int(pixels.max()), sample_ids=[s.sample_id for s in selected],
                         sample_hashes=hashes, hash=hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest())
        except (OSError, ValueError, PermissionError) as exc:
            report["errors"].append({"loader": name, "message": str(exc)})
    alias = EvalSet(dataset_root, "development_selection")
    report["development_selection_matches_fusion"] = [s.sample_id for s in alias] == [s.sample_id for s in loaders["eval_fusion"][0]]
    if not report["development_selection_matches_fusion"]:
        report["errors"].append({"loader": "development_selection", "message": "Alias differs from fusion"})
    report["overall"] = "FAIL" if report["errors"] else "PASS"
    return report
