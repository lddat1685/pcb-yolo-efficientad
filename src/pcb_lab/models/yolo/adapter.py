"""Inference adapter: a deterministic engine over a YOLO checkpoint.

The engine accepts an **RGB** uint8 HWC array (canonical, per Step 2) and returns
detections in the original image coordinate frame. It converts RGB->BGR internally
because Ultralytics' numpy input path expects BGR (engine/predictor.py:179 flips
BGR->RGB). ``YoloDetector`` reuses Step 2's ``prepare_input``/``unletterbox_boxes``
as the single path from pixels to model input and back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from pcb_lab.data.manifest import (
    METADATA_PATHS,
    DatasetInputError,
    load_dataset,
    dataset_path,
)
from pcb_lab.inference.preprocessing import LetterboxMeta, prepare_input, unletterbox_boxes


class ArtifactMismatchError(ValueError):
    """Checkpoint sha256 differs from artifact.json."""


class ClassOrderError(ValueError):
    """Model class order differs from classes.json."""


class SmokeArtifactError(ValueError):
    """A smoke artifact was loaded without allow_smoke=True."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy_original: tuple[float, float, float, float]


class Engine:
    """Protocol: infer(rgb_uint8_hwc) -> ndarray[N,6] (x1,y1,x2,y2,conf,cls)."""

    def infer(self, pixels_rgb_uint8_hwc: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UltralyticsEngine(Engine):
    """Wraps a loaded ultralytics YOLO model; expects RGB arrays, returns original-frame boxes."""

    def __init__(self, model, infer_params: dict, class_names: list[str], device="cpu"):
        self.model = model
        self.infer_params = dict(infer_params)
        self.class_names = list(class_names)
        self.device = device

    def infer(self, pixels_rgb_uint8_hwc: np.ndarray) -> np.ndarray:
        arr = np.asarray(pixels_rgb_uint8_hwc)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Engine.infer requires uint8 HWC RGB array")
        # Ultralytics numpy path expects BGR; convert so it matches the path-based reader.
        bgr = arr[..., ::-1]
        params = self.infer_params
        results = self.model.predict(
            source=bgr,
            conf=params["conf_floor"],
            iou=params["iou"],
            max_det=params["max_det"],
            agnostic_nms=params["agnostic_nms"],
            imgsz=params["imgsz"],
            device=self.device,
            half=params.get("half", False),
            verbose=False,
            stream=False,
            retina_masks=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float64)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = boxes.conf.cpu().numpy().astype(np.float64).reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().astype(np.float64).reshape(-1, 1)
        return np.hstack([xyxy, conf, cls])


class YoloDetector:
    """High-level detector: prepares input via Step 2, runs the engine, unletterboxes."""

    def __init__(self, engine: Engine, meta: dict):
        self.engine = engine
        self.meta = dict(meta)

    @classmethod
    def from_artifact(cls, artifact_dir, allow_smoke: bool = False, *, class_names=None) -> "YoloDetector":
        artifact_dir = Path(artifact_dir)
        artifact_json = artifact_dir / "artifact.json"
        if not artifact_json.exists():
            raise FileNotFoundError(f"artifact.json not found in {artifact_dir}")
        meta = yaml.safe_load(artifact_json.read_text(encoding="utf-8-sig"))
        if meta.get("smoke") and not allow_smoke:
            raise SmokeArtifactError(f"{artifact_dir} is a smoke artifact; pass allow_smoke=True")
        best = artifact_dir / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"best.pt not found in {artifact_dir}")
        import ultralytics
        from ultralytics import YOLO
        # Verify checkpoint sha matches artifact record.
        check_sha(best, meta.get("checkpoint_best_sha256"), artifact_dir)
        # Evaluation keeps its canonical dataset guard. Standalone inference can
        # use artifact labels, still checked against the actual checkpoint below.
        if class_names is None:
            root = resolve_classes_root(meta)
            names = load_canonical_names(root)
        else:
            if (not isinstance(class_names, list) or not class_names or
                    any(not isinstance(n, str) or not n.strip() for n in class_names) or
                    len(set(class_names)) != len(class_names) or class_names != meta.get("class_order")):
                raise ClassOrderError("Invalid artifact class_order for standalone inference")
            names = list(class_names)
        model = YOLO(str(best))
        raw = getattr(model.model, "names", []) or getattr(model, "names", []) or []
        # names may be a {idx: name} dict or a list; normalize to a list by index.
        if isinstance(raw, dict):
            model_names = [raw[i] for i in sorted(raw)]
        else:
            model_names = list(raw)
        if model_names != names:
            raise ClassOrderError(f"Model class order {model_names} != expected class order {names}")
        infer_params = dict(meta["inference_params"])
        engine = UltralyticsEngine(model, infer_params, names, device=meta.get("device", "cpu"))
        detector_meta = {
            "model_id": meta.get("model_id"),
            "checkpoint_sha256": meta.get("checkpoint_best_sha256"),
            "class_order": meta.get("class_order"),
            "infer_params": infer_params,
            "input_color": "rgb",
        }
        return cls(engine, detector_meta)

    def predict(self, image_rgb) -> list[Detection]:
        """image_rgb: RGB uint8 HWC array or path. Returns detections sorted by confidence desc."""
        prepared = prepare_input(image_rgb, "yolo", cfg=None)
        prepared_meta = prepared.meta
        letterbox_meta = LetterboxMeta(
            scale=prepared_meta["scale"], pad_left=prepared_meta["pad_left"],
            pad_top=prepared_meta["pad_top"], orig_w=prepared_meta["orig_w"],
            orig_h=prepared_meta["orig_h"],
            preprocessing_version=prepared_meta.get("preprocessing_version", "prep_v1"),
            preprocessing_hash=prepared_meta.get("preprocessing_hash", ""),
            color_order="rgb", normalized=False,
        )
        dets = self.engine.infer(prepared.pixels)  # [N,6] in letterboxed 640 space
        if dets.size == 0:
            return []
        original = unletterbox_boxes(dets[:, :4], letterbox_meta)  # back to original frame
        detections = []
        names = self.engine.class_names
        for (x1, y1, x2, y2), (conf, cid) in zip(original, dets[:, 4:6]):
            cid_int = int(round(float(cid)))
            detections.append(Detection(
                class_id=cid_int,
                class_name=names[cid_int] if cid_int < len(names) else str(cid_int),
                confidence=float(conf),
                xyxy_original=(float(x1), float(y1), float(x2), float(y2)),
            ))
        detections.sort(key=lambda d: (-d.confidence, d.class_id, *d.xyxy_original))
        return detections

    def load(self) -> "YoloDetector":
        return self

    def describe(self) -> dict:
        return dict(self.meta)

    def image_score(self, detections: list[Detection]) -> float:
        if not detections:
            return 0.0
        return max(d.confidence for d in detections)


# --- helpers ---------------------------------------------------------------

def check_sha(path: Path, expected: str | None, context: Path) -> None:
    if expected is None:
        return
    from pcb_lab.data.manifest import sha256_file
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactMismatchError(
            f"Checkpoint {path.name} sha256 {actual} != artifact.json {expected} (in {context})")


def resolve_classes_root(meta: dict) -> Path:
    """Find dataset root from configs/dataset.yaml or environment."""
    from pcb_lab.data.manifest import resolve_dataset_root
    try:
        return resolve_dataset_root()
    except Exception:
        return Path.cwd()


def load_canonical_names(root: Path | None = None) -> list[str]:
    from pcb_lab.data.manifest import resolve_dataset_root
    data = load_dataset(resolve_dataset_root(root))
    return data.names
