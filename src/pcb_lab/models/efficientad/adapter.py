"""Tile Engine protocol, feathering, R8 components and verified artifact loading."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pcb_lab.data.manifest import sha256_file
from pcb_lab.inference.tiling import TileManager, global_nms


class ArtifactMismatchError(ValueError):
    """Artifact provenance or checkpoint content does not match metadata."""


class SmokeArtifactError(ValueError):
    """A smoke artifact was used without an explicit opt-in."""


class Engine(Protocol):
    def infer_tiles(self, tiles_uint8_nhwc: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Detection:
    class_id: int | None
    class_name: str
    confidence: float
    xyxy_original: tuple[float, float, float, float]


@dataclass(frozen=True)
class EfficientAdResult:
    anomaly_score: float
    anomaly_map: np.ndarray
    boxes: list[Detection]


class EfficientAdEngine:
    def __init__(self, model, device="cpu"):
        import torch
        self.device = torch.device("cuda:" + str(device) if str(device).isdigit() else device)
        self.model = model.to(self.device).eval()

    def infer_tiles(self, tiles_uint8_nhwc):
        import torch
        tiles = np.asarray(tiles_uint8_nhwc)
        if tiles.dtype != np.uint8 or tiles.ndim != 4 or tiles.shape[1:] != (256, 256, 3) or len(tiles) == 0:
            raise ValueError("Engine expects nonempty RGB uint8 [N,256,256,3]")
        maps = []
        with torch.inference_mode():
            if self.device.type == "cuda":
                pixels = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).to(
                    self.device, dtype=torch.float32) / 255.0
                return self.model(pixels).cpu().numpy().astype(np.float32)
            # Native batch=1 bounds CPU/GPU memory even for nine tiles.
            for tile in tiles:
                pixels = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).to(
                    self.device, dtype=torch.float32).unsqueeze(0) / 255
                maps.append(self.model(pixels)[0].cpu().numpy())
        return np.stack(maps).astype(np.float32)


DEFAULT_INFER = dict(tile_size=256, stride=224, stitch_method="linear", score_metric="max",
    tau_pixel=0.5, morph_k=3, min_area=20, nms_iou=0.45, box_class_name="anomaly_unclassified")


class EfficientAdDetector:
    def __init__(self, engine: Engine, meta: dict):
        self.engine, self.meta = engine, deepcopy(meta)
        self.infer_params = dict(DEFAULT_INFER, **meta.get("inference_params", meta.get("infer_params", {})))
        p = self.infer_params
        for key in ("tile_size", "stride", "stitch_method", "score_metric", "morph_k", "nms_iou", "box_class_name"):
            if p[key] != DEFAULT_INFER[key]:
                raise ValueError(f"Locked inference parameter differs: {key}")
        if not np.isfinite(p["tau_pixel"]) or type(p["min_area"]) is not int or p["min_area"] < 0:
            raise ValueError("Invalid component threshold/area")
        self.tile_manager = TileManager(256, 224, "reflect", 0.45)

    def predict(self, image_rgb_640x640, *, skip_boxes=False):
        pixels = np.asarray(image_rgb_640x640)
        tiles, specs = self.tile_manager.split(pixels)
        tile_maps = self.engine.infer_tiles(tiles)
        heatmap = self.tile_manager.stitch(tile_maps, specs, pixels.shape[:2])
        score = float(heatmap.max())
        if skip_boxes:
            return EfficientAdResult(score, heatmap, [])
        p = self.infer_params
        mask = (heatmap > p["tau_pixel"]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            if area > p["min_area"]:
                confidence = float(heatmap[labels == component].max())
                candidates.append((x, y, x + w, y + h, confidence))
        kept = global_nms(np.asarray(candidates, dtype=np.float64).reshape(-1, 5), p["nms_iou"])
        boxes = [Detection(None, "anomaly_unclassified", float(row[4]), tuple(map(float, row[:4]))) for row in kept]
        return EfficientAdResult(score, heatmap, boxes)

    def describe(self):
        return {"model_id": self.meta.get("model_id"), "checkpoint_sha256": self.meta.get("checkpoint_sha256"),
                "norm_stats": deepcopy(self.meta.get("normalization_stats", {})),
                "infer_params": deepcopy(self.infer_params), "input_color": "rgb"}

    @classmethod
    def from_artifact(cls, artifact_dir, allow_smoke=False, *, device="cpu"):
        import torch
        from .architecture import EfficientAdModel
        path = Path(artifact_dir)
        meta = json.loads((path / "artifact.json").read_text(encoding="utf-8-sig"))
        if type(meta.get("smoke")) is not bool:
            raise ArtifactMismatchError("Artifact smoke flag must be boolean")
        if meta["smoke"] and not allow_smoke:
            raise SmokeArtifactError("Pass allow_smoke=True only for smoke validation")
        checkpoint = path / "model.pt"
        if sha256_file(checkpoint) != meta.get("checkpoint_sha256"):
            raise ArtifactMismatchError("Checkpoint SHA256 differs from artifact.json")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key in ("config_hash", "model_id", "seed", "smoke", "tile_params", "normalization_stats", "inference_params"):
            if payload.get(key) != meta.get(key):
                raise ArtifactMismatchError(f"Checkpoint metadata differs: {key}")
        model = EfficientAdModel()
        model.load_state_dict(payload["model_state"], strict=True)
        return cls(EfficientAdEngine(model, device), meta)
