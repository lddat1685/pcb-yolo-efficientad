"""Native 256px tiles with normalized linear feathering and box geometry."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def image_hw(out_hw) -> tuple[int, int]:
    if len(out_hw) != 2:
        raise ValueError("out_hw must contain height and width")
    return positive_int(out_hw[0], "height"), positive_int(out_hw[1], "width")


def _detections(dets) -> np.ndarray:
    arr = np.asarray(dets)
    if arr.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("Detections must have shape [N,5]: x1,y1,x2,y2,score")
    if not np.isfinite(arr).all():
        raise ValueError("Detections must contain finite values")
    return arr.astype(np.float64, copy=True)


def global_nms(dets: np.ndarray, iou: float = 0.45) -> np.ndarray:
    """Stable class-agnostic NMS: suppress only IoU strictly above threshold."""
    if not np.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError("NMS IoU must lie in [0,1]")
    arr = _detections(dets)
    if np.any(arr[:, 2:4] <= arr[:, :2]):
        raise ValueError("NMS requires positive-area boxes")
    order = np.argsort(-arr[:, 4], kind="stable")
    kept = []
    areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    while order.size:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        lo = np.maximum(arr[current, :2], arr[rest, :2])
        hi = np.minimum(arr[current, 2:4], arr[rest, 2:4])
        wh = np.maximum(hi - lo, 0)
        intersection = wh[:, 0] * wh[:, 1]
        overlap = intersection / (areas[current] + areas[rest] - intersection)
        order = rest[overlap <= iou]
    return arr[kept].reshape(-1, 5)


@dataclass(frozen=True)
class TileSpec:
    tile_id: int
    row: int
    col: int
    x0: int
    y0: int
    size: int


class TileManager:
    def __init__(self, tile_size=256, stride=224, pad_mode="reflect", nms_iou=0.45):
        self.tile_size = positive_int(tile_size, "tile_size")
        if self.tile_size != 256:
            raise ValueError("Only native tile_size=256 is supported")
        self.stride = positive_int(stride, "stride")
        if self.stride > self.tile_size:
            raise ValueError("stride cannot exceed tile_size (coverage gaps)")
        if pad_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError("pad_mode must be reflect, replicate or constant")
        if not np.isfinite(nms_iou) or not 0 <= nms_iou <= 1:
            raise ValueError("nms_iou must lie in [0,1]")
        self.pad_mode, self.nms_iou = pad_mode, float(nms_iou)

    def plan(self, h, w) -> list[TileSpec]:
        h, w = image_hw((h, w))
        # Integer ceiling avoids the arithmetic error in the original plan.
        rows = 1 + (max(0, h - self.tile_size) + self.stride - 1) // self.stride
        cols = 1 + (max(0, w - self.tile_size) + self.stride - 1) // self.stride
        return [TileSpec(r * cols + c, r, c, c * self.stride, r * self.stride, self.tile_size)
                for r in range(rows) for c in range(cols)]

    def split(self, img: np.ndarray) -> tuple[np.ndarray, list[TileSpec]]:
        arr = np.asarray(img)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("split requires canonical RGB uint8 [H,W,3]")
        h, w = image_hw(arr.shape[:2])
        specs = self.plan(h, w)
        bottom = specs[-1].y0 + self.tile_size - h
        right = specs[-1].x0 + self.tile_size - w
        padding = ((0, bottom), (0, right), (0, 0))
        # NumPy reflect on a singleton axis repeats its sole value.
        mode = "edge" if self.pad_mode == "replicate" else self.pad_mode
        padded = np.pad(arr, padding, mode=mode)
        tiles = np.stack([padded[s.y0:s.y0 + s.size, s.x0:s.x0 + s.size] for s in specs])
        return tiles, specs

    def _specs(self, specs, out_hw) -> tuple[list[TileSpec], int, int]:
        h, w = image_hw(out_hw)
        specs = list(specs)
        expected = self.plan(h, w)
        if (any(not isinstance(s, TileSpec) for s in specs)
                or sorted(specs, key=lambda s: s.tile_id) != expected):
            raise ValueError("specs must cover out_hw exactly once with this manager's complete plan")
        return specs, h, w

    def _axis_weights(self, origin: int, last_origin: int) -> np.ndarray:
        overlap = self.tile_size - self.stride
        if overlap <= 1:
            return np.ones(self.tile_size, dtype=np.float64)
        center = (self.tile_size - 1) / 2
        distance_from_edge = center - np.abs(np.arange(self.tile_size) - center)
        weights = np.clip(distance_from_edge / min(overlap - 1, center), 0, 1)
        # No taper against the outer canvas: even corner pixels have weight 1.
        if origin == 0:
            weights[:self.tile_size // 2] = 1
        if origin == last_origin:
            weights[self.tile_size // 2:] = 1
        return weights

    def stitch(self, tile_maps: np.ndarray, specs, out_hw) -> np.ndarray:
        specs, h, w = self._specs(specs, out_hw)
        maps = np.asarray(tile_maps)
        if maps.shape != (len(specs), self.tile_size, self.tile_size):
            raise ValueError("tile_maps must have shape [len(specs),256,256]")
        if not np.issubdtype(maps.dtype, np.number) or not np.isfinite(maps).all():
            raise ValueError("tile_maps must contain finite numeric values")
        last_x, last_y = max(s.x0 for s in specs), max(s.y0 for s in specs)
        numerator = np.zeros((h, w), dtype=np.float64)
        denominator = np.zeros((h, w), dtype=np.float64)
        for tile_map, spec in zip(maps, specs):
            ph, pw = min(spec.size, h - spec.y0), min(spec.size, w - spec.x0)
            weights = (self._axis_weights(spec.y0, last_y)[:, None]
                       * self._axis_weights(spec.x0, last_x)[None, :])[:ph, :pw]
            region = np.s_[spec.y0:spec.y0 + ph, spec.x0:spec.x0 + pw]
            numerator[region] += tile_map[:ph, :pw].astype(np.float64) * weights
            denominator[region] += weights
        if np.any(denominator <= 0):
            raise ValueError("Invalid tile coverage: zero accumulated weight")
        return (numerator / denominator).astype(np.float32)

    def boxes_to_global(self, per_tile_dets, specs, out_hw) -> np.ndarray:
        specs, h, w = self._specs(specs, out_hw)
        if len(per_tile_dets) != len(specs):
            raise ValueError("One detection array is required for each TileSpec")
        result = []
        for dets, spec in zip(per_tile_dets, specs):
            arr = _detections(dets)
            arr[:, [0, 2]] += spec.x0
            arr[:, [1, 3]] += spec.y0
            arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0, w)
            arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, h)
            result.append(arr[(arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])])
        return np.concatenate(result, axis=0)

    def global_nms(self, dets: np.ndarray, iou: float | None = None) -> np.ndarray:
        return global_nms(dets, self.nms_iou if iou is None else iou)
