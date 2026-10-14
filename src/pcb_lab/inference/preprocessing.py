"""One CPU preprocessing path for API/CLI, with explicit geometry/provenance."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from pcb_lab.data.manifest import METADATA_PATHS, dataset_path, load_dataset, resolve_dataset_root
from .tiling import TileManager, image_hw, positive_int


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_preprocessing_config(cfg=None) -> dict:
    path = Path(__file__).resolve().parents[3] / "configs/preprocessing.yaml"
    defaults = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if cfg is None:
        incoming = {}
    elif isinstance(cfg, (str, Path)):
        incoming = yaml.safe_load(Path(cfg).read_text(encoding="utf-8-sig"))
    else:
        incoming = deepcopy(cfg)
    if not isinstance(incoming, dict):
        raise ValueError("Preprocessing configuration must be a mapping")

    def merge(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    result = merge(defaults, incoming)
    try:
        if not isinstance(result["preprocessing_version"], str) or not result["preprocessing_version"]:
            raise ValueError("preprocessing_version must be nonempty")
        if result["canonical"] != {"color": "rgb", "dtype": "uint8", "layout": "HWC"}:
            raise ValueError("Canonical format must be RGB uint8 HWC")
        yolo, ad = result["yolo"], result["ad"]
        yolo["imgsz"] = positive_int(yolo["imgsz"], "yolo.imgsz")
        if type(yolo["pad_value"]) is not int or not 0 <= yolo["pad_value"] <= 255:
            raise ValueError("yolo.pad_value must be an integer in [0,255]")
        ad["resize"]["size"] = positive_int(ad["resize"]["size"], "ad.resize.size")
        if ad["resize"]["interpolation"] != "bilinear" or ad["tile"]["blend"] != "linear":
            raise ValueError("Only bilinear resize and linear blending are supported")
        tile = ad["tile"]
        manager = TileManager(tile["size"], tile["stride"], tile["pad_mode"], tile["global_nms_iou"])
        tile.update(size=manager.tile_size, stride=manager.stride, global_nms_iou=manager.nms_iou)
        if result["partition_alias"].get("development_selection") != "fusion":
            raise ValueError("development_selection must remain an alias for fusion")
        if result["augment"]["recipe_id"] != "aug_v1":
            raise ValueError("Only the approved aug_v1 recipe is available")
        if (result["augment"]["rotations_ccw"] != [0, 90, 180, 270]
                or result["augment"]["horizontal_flip_probability"] != 0.5
                or result["augment"]["photometric"] is not False):
            raise ValueError("aug_v1 parameters are frozen; changes require a separately approved recipe")
        _json(result)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid preprocessing configuration: {exc}") from exc
    return result


def preprocessing_hash(cfg=None) -> str:
    return hashlib.sha256(_json(load_preprocessing_config(cfg)).encode("utf-8")).hexdigest()


class TransformMeta(dict):
    """JSON-compatible metadata supporting both meta['normalized'] and attributes."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _meta(cfg, **values) -> TransformMeta:
    return TransformMeta(preprocessing_version=cfg["preprocessing_version"],
                         preprocessing_hash=preprocessing_hash(cfg), color_order="rgb", normalized=False, **values)


@lru_cache(maxsize=8)
def _canonical_partitions(root: str, manifest_stamp: tuple) -> dict:
    # The stamp invalidates the cache if the manifest is modified in a fixture.
    data = load_dataset(Path(root))
    result = {}
    for row in data.rows:
        path = dataset_path(data.root, row["image"])
        result.setdefault(path, set()).add(row["split"])
        # Holdout can veto a conflicting manifest row before any pixel I/O.
        if row["sample_id"] in data.holdout["samples"]["test"]:
            result[path].add("test")
    return result


def guard_image_path(path: Path, dataset_root=None, *, allow_test=False) -> Path:
    """Reject canonical test paths before Pillow opens them, including aliases."""
    path = Path(path).resolve()
    root = resolve_dataset_root(Path(dataset_root) if dataset_root is not None else None)
    if path.is_relative_to(root):
        manifest = dataset_path(root, METADATA_PATHS["manifest"])
        holdout = dataset_path(root, METADATA_PATHS["holdout"])
        stamp = (manifest.stat().st_mtime_ns, manifest.stat().st_size,
                 holdout.stat().st_mtime_ns, holdout.stat().st_size)
        partitions = _canonical_partitions(str(root), stamp).get(path)
        if partitions is None:
            raise PermissionError("Image under dataset root is not a canonical manifest image")
        if "test" in partitions and not allow_test:
            raise PermissionError("Test image access requires allow_test=True")
    return path


def to_canonical(image_or_path, *, dataset_root=None, allow_test=False) -> np.ndarray:
    """RGB uint8 HWC; alpha is discarded, grayscale replicated, no BGR swap."""
    if hasattr(image_or_path, "load_image"):
        return to_canonical(image_or_path.load_image())
    if isinstance(image_or_path, (str, Path)):
        path = guard_image_path(Path(image_or_path), dataset_root, allow_test=allow_test)
        with Image.open(path) as image:
            image.load()
            return to_canonical(image)
    if isinstance(image_or_path, Image.Image):
        if image_or_path.mode not in {"1", "L", "LA", "RGB", "RGBA", "P"}:
            raise ValueError("Expected an 8-bit image (L, LA, RGB or RGBA)")
        with image_or_path.convert("RGB") as rgb:
            arr = np.array(rgb, dtype=np.uint8, copy=True)
    else:
        arr = np.asarray(image_or_path)
        if arr.dtype != np.uint8:
            raise ValueError("Canonical input must be uint8; normalized/float input is not accepted")
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] in (1, 2):
            arr = np.repeat(arr[..., :1], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = arr[..., :3]
        else:
            raise ValueError("Expected HW grayscale or HWC image with 1/2/3/4 channels")
    image_hw(arr.shape[:2])
    return np.array(arr, dtype=np.uint8, order="C", copy=True)


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_left: int
    pad_top: int
    orig_w: int
    orig_h: int
    preprocessing_version: str = "prep_v1"
    preprocessing_hash: str = field(default_factory=preprocessing_hash)
    color_order: str = "rgb"
    normalized: bool = False


def letterbox(img, target=640, pad_value=114, *, cfg=None):
    config = load_preprocessing_config(cfg)
    target = positive_int(target, "target")
    if type(pad_value) is not int or not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be an integer in [0,255]")
    config["yolo"].update(imgsz=target, pad_value=pad_value)
    arr = to_canonical(img)
    h, w = arr.shape[:2]
    scale = min(target / w, target / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (target - rw) // 2, (target - rh) // 2
    resized = np.asarray(Image.fromarray(arr).resize((rw, rh), Image.Resampling.BILINEAR))
    canvas = np.full((target, target, 3), pad_value, dtype=np.uint8)
    canvas[top:top + rh, left:left + rw] = resized
    meta = LetterboxMeta(scale, left, top, w, h, config["preprocessing_version"], preprocessing_hash(config))
    return canvas, meta


def unletterbox_boxes(boxes_xyxy, meta) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must be finite [N,4]")
    get = (lambda key: meta[key]) if isinstance(meta, dict) else (lambda key: getattr(meta, key))
    scale = get("scale")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Letterbox scale must be positive")
    result = (boxes - np.array([get("pad_left"), get("pad_top")] * 2)) / scale
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, get("orig_w"))
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, get("orig_h"))
    return result


def normalize(img, mean, std, *, cfg=None):
    arr = to_canonical(img)  # Rejects the float32 output of any earlier normalization.
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("mean/std must contain three finite RGB values; std must be positive")
    pixels = (arr.astype(np.float32) / np.float32(255) - mean) / std
    meta = _meta(load_preprocessing_config(cfg), orig_hw=list(arr.shape[:2]), mean=mean.tolist(), std=std.tolist())
    meta["normalized"] = True
    return pixels, meta


def resize_for_ad(img, size=256, *, cfg=None):
    config = load_preprocessing_config(cfg)
    size = positive_int(size, "size")
    config["ad"]["resize"]["size"] = size
    arr = to_canonical(img)
    h, w = arr.shape[:2]
    resized = np.array(Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR), copy=True)
    return resized, _meta(config, orig_hw=[h, w], scale_xy=[size / w, size / h], interpolation="bilinear")


def upsample_map(map2d, out_hw) -> np.ndarray:
    arr = np.asarray(map2d, dtype=np.float32)
    h, w = image_hw(out_hw)
    if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
        raise ValueError("map2d must be a nonempty finite 2D array")
    resized = np.array(Image.fromarray(arr).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.clip(resized, arr.min(), arr.max())


@dataclass(frozen=True)
class PreparedInput:
    pixels: np.ndarray
    meta: dict

    def sha256(self) -> str:
        header = {"dtype": self.pixels.dtype.name, "shape": list(self.pixels.shape), "meta": self.meta}
        digest = hashlib.sha256(_json(header).encode("utf-8") + b"\0")
        digest.update(np.ascontiguousarray(self.pixels).tobytes())
        return digest.hexdigest()


def prepare_input(image_or_path, mode, cfg=None) -> PreparedInput:
    if mode not in {"yolo", "ad_tile", "ad_resize"}:
        raise ValueError("mode must be yolo, ad_tile or ad_resize")
    config = load_preprocessing_config(cfg)
    arr = to_canonical(image_or_path)
    meta = _meta(config, mode=mode, orig_hw=list(arr.shape[:2]))
    if mode == "yolo":
        pixels, geometry = letterbox(arr, config["yolo"]["imgsz"], config["yolo"]["pad_value"], cfg=config)
        meta.update(asdict(geometry))
    elif mode == "ad_resize":
        pixels, geometry = resize_for_ad(arr, config["ad"]["resize"]["size"], cfg=config)
        meta.update(geometry)
    else:
        tile = config["ad"]["tile"]
        manager = TileManager(tile["size"], tile["stride"], tile["pad_mode"], tile["global_nms_iou"])
        pixels, specs = manager.split(arr)
        meta.update(tile_specs=[asdict(s) for s in specs], pad_mode=manager.pad_mode,
                    canvas_hw=[specs[-1].y0 + manager.tile_size, specs[-1].x0 + manager.tile_size],
                    pad_bottom=specs[-1].y0 + manager.tile_size - arr.shape[0],
                    pad_right=specs[-1].x0 + manager.tile_size - arr.shape[1], blend="linear")
    return PreparedInput(pixels, dict(meta))


def preview_input(image_or_path, mode, cfg=None) -> dict:
    """Shared JSON result for CLI and direct API calls."""
    prepared = prepare_input(image_or_path, mode, cfg)
    return {"meta": prepared.meta, "sha256": prepared.sha256(),
            "shape": list(prepared.pixels.shape), "dtype": prepared.pixels.dtype.name}
