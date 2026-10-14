"""One inference path for CLI and Streamlit, with actual conditional execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from threading import RLock
from time import perf_counter
from uuid import uuid4

import numpy as np

from pcb_lab.data.manifest import sha256_file
from pcb_lab.fusion.rules import predict_cascade, predict_or
from .preprocessing import preprocessing_hash
from .registry import ModelAsset, ModelRegistry
from .sources import BaseImageSource, validate_pixels


@dataclass
class LoadedModel:
    detector: object
    lock: RLock = field(default_factory=RLock)


def load_model(asset: ModelAsset, device: str = "cpu") -> LoadedModel:
    """Verify freeze + artifact before loading; never download a checkpoint."""
    if sha256_file(asset.checkpoint) != asset.sha256:
        raise ValueError(f"Checkpoint không khớp cấu hình đã khóa: {asset.model_id}")
    meta = json.loads((asset.checkpoint.parent / "artifact.json").read_text(encoding="utf-8-sig"))
    if meta.get("model_id") != asset.model_id:
        raise ValueError("Model ID không khớp registry")
    if asset.model_id.startswith("B03"):
        from pcb_lab.models.efficientad.adapter import EfficientAdDetector
        detector = EfficientAdDetector.from_artifact(asset.checkpoint.parent, device=device)
    else:
        from pcb_lab.models.yolo.adapter import YoloDetector
        if meta.get("preprocessing_hash") != preprocessing_hash():
            raise ValueError("Tiền xử lý YOLO không khớp artifact")
        class_names = meta.get("class_order")
        if not isinstance(class_names, list) or not class_names:
            raise ValueError("Artifact YOLO thiếu class_order cho suy luận độc lập")
        detector = YoloDetector.from_artifact(asset.checkpoint.parent, class_names=class_names)
        detector.engine.device = device
    return LoadedModel(detector)


@dataclass
class InspectionResult:
    model_id: str
    sample_id: str = "unknown"
    decision: str = "ERROR"
    mode: str = ""
    reason: str = ""
    routing_trace: list[str] = field(default_factory=list)
    yolo_executed: bool = False
    efficientad_executed: bool = False
    early_exit: bool = False
    yolo_image_score: float | None = None
    anomaly_score: float | None = None
    detections: list[dict] = field(default_factory=list)
    detection_candidates: list[dict] = field(default_factory=list)
    anomaly_regions: list[dict] = field(default_factory=list)
    timing_ms: dict = field(default_factory=lambda: dict(decode=0., model_load=0., yolo=0., efficientad=0., total_core=0., total=0.))
    metadata: dict = field(default_factory=dict)
    error_reason: str | None = None
    run_id: str = field(default_factory=lambda: "inspection-" + uuid4().hex)
    image: np.ndarray | None = field(default=None, repr=False)
    anomaly_map: np.ndarray | None = field(default=None, repr=False)

    @property
    def latency_ms(self):
        return self.timing_ms["total_core"]

    def to_dict(self):
        # Avoid deep-copying image/heatmap into every CSV/JSON export.
        result = {k: v for k, v in vars(self).items() if k not in {"image", "anomaly_map"}}
        heatmap = self.anomaly_map
        result["heatmap"] = None if heatmap is None else {
            "shape": list(heatmap.shape), "dtype": str(heatmap.dtype),
            "min": float(heatmap.min()), "max": float(heatmap.max()),
            "sha256": hashlib.sha256(heatmap.tobytes()).hexdigest(),
        }
        return result

    def summary(self):
        return {"sample_id": self.sample_id, "model_id": self.model_id,
                "decision": self.decision, "reason": self.reason,
                "route_ad": self.efficientad_executed, "early_exit": self.early_exit,
                "yolo_score": self.yolo_image_score, "ad_score": self.anomaly_score,
                "box_count": len(self.detections), "latency_ms": self.latency_ms,
                "ground_truth": self.metadata.get("ground_truth"), "error": self.error_reason}


class InferenceService:
    def __init__(self, registry=None, *, device="cpu", model_loader=load_model):
        self.registry = registry or ModelRegistry()
        self.device, self.model_loader = device, model_loader
        self._models = {}
        self._load_lock = RLock()

    def _model(self, asset):
        key = (asset.fingerprint(), self.device)
        with self._load_lock:
            if key not in self._models:
                self._models[key] = self.model_loader(asset, self.device)
            return self._models[key]

    def inspect(self, source: BaseImageSource, model_id="H01") -> InspectionResult:
        result = InspectionResult(model_id)
        started = perf_counter()
        try:
            spec = self.registry.get(model_id)
            result.mode = spec.mode
            tick = perf_counter()
            result.sample_id, image = source.get_frame()
            result.image = image = validate_pixels(image)
            result.timing_ms["decode"] = (perf_counter() - tick) * 1000
            result.metadata = dict(source.describe(), config_hash=spec.config_hash,
                tau_yolo=spec.tau_yolo, tau_ad=spec.tau_ad, device=self.device,
                backend="pytorch", precision="float32", configuration="locked",
                input_size=[image.shape[1], image.shape[0]], preprocessing_hash=preprocessing_hash(),
                pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                checkpoint_hashes={a.model_id: a.sha256 for a in (spec.yolo, spec.ad) if a},
                score_definition={"yolo": "max box confidence", "ad": "max stitched anomaly map"})
            if image.shape[:2] != (640, 640):
                result.metadata["warning"] = "Ảnh khác kích thước benchmark 640×640; kiểm tra miền PCB trần trước khi diễn giải kết quả."
            if spec.yolo:
                result.routing_trace.append("YOLO")
                tick = perf_counter()
                loaded = self._model(spec.yolo)
                result.timing_ms["model_load"] += (perf_counter() - tick) * 1000
                with loaded.lock:
                    tick = perf_counter()
                    result.yolo_executed = True
                    detections = loaded.detector.predict(image)
                    result.timing_ms["yolo"] = (perf_counter() - tick) * 1000
                if any(not np.isfinite(d.confidence) for d in detections):
                    raise ValueError("YOLO trả score không hữu hạn")
                result.yolo_image_score = max((float(d.confidence) for d in detections), default=0.)
                result.detection_candidates = [asdict(d) for d in detections]
                result.detections = [asdict(d) for d in detections if d.confidence >= spec.tau_yolo]
                result.metadata["yolo_infer_params"] = loaded.detector.describe().get("infer_params", {})
            yolo_hit = spec.yolo is not None and result.yolo_image_score >= spec.tau_yolo
            result.early_exit = spec.mode == "CASCADE" and yolo_hit
            if spec.ad and not result.early_exit:
                result.routing_trace.append("EfficientAD")
                tick = perf_counter()
                loaded = self._model(spec.ad)
                result.timing_ms["model_load"] += (perf_counter() - tick) * 1000
                with loaded.lock:
                    tick = perf_counter()
                    result.efficientad_executed = True
                    prediction = loaded.detector.predict(image)
                    result.timing_ms["efficientad"] = (perf_counter() - tick) * 1000
                heatmap = np.asarray(prediction.anomaly_map)
                if not np.isfinite(prediction.anomaly_score) or heatmap.shape != image.shape[:2] or not np.isfinite(heatmap).all():
                    raise ValueError("EfficientAD trả score/heatmap không hợp lệ")
                result.anomaly_score = float(prediction.anomaly_score)
                result.anomaly_map = heatmap.copy()
                result.anomaly_regions = [asdict(b) for b in prediction.boxes] if result.anomaly_score >= spec.tau_ad else []
                params = loaded.detector.describe().get("infer_params", {})
                result.metadata["tiling_params"] = params
                result.metadata["tile_count"] = len(loaded.detector.tile_manager.plan(*image.shape[:2]))
            tick = perf_counter()
            if spec.mode in {"CASCADE", "OR"}:
                base = dict(sample_id=result.sample_id, split=result.metadata.get("split", "unlabelled"),
                            is_defect=result.metadata.get("ground_truth") == "DEFECT")
                yolo = dict(base, yolo_image_score=result.yolo_image_score, detections=result.detections)
                ad = dict(base, anomaly_score=result.anomaly_score or 0., boxes=result.anomaly_regions)
                rule = predict_cascade if spec.mode == "CASCADE" else predict_or
                decision = rule(yolo, ad, spec.tau_yolo, spec.tau_ad,
                                allow_test=result.metadata.get("split") == "test")
                result.decision, result.reason = decision.decision, decision.reason
                if spec.mode == "OR" and yolo_hit and result.anomaly_score >= spec.tau_ad:
                    result.reason = "both_hit"
            else:
                positive = yolo_hit if spec.mode == "YOLO" else result.anomaly_score >= spec.tau_ad
                result.decision = "DEFECT" if positive else "GOOD"
                result.reason = ("yolo_hit" if spec.mode == "YOLO" else "ad_hit") if positive else "all_pass"
            result.routing_trace.append("Early-exit" if result.early_exit else result.decision)
            result.timing_ms["total_core"] = result.timing_ms["yolo"] + result.timing_ms["efficientad"] + (perf_counter() - tick) * 1000
        except Exception as exc:
            result.decision, result.reason = "ERROR", "inference_error"
            result.error_reason = f"{type(exc).__name__}: {exc}"
            result.routing_trace.append("ERROR")
        result.timing_ms["total"] = (perf_counter() - started) * 1000
        return result

    def inspect_batch(self, sources, model_id="H01", *, progress=None, limit=20):
        sources = list(sources)
        if not 1 <= len(sources) <= limit:
            raise ValueError(f"Batch phải có 1–{limit} ảnh")
        results = []
        for index, source in enumerate(sources):
            results.append(self.inspect(source, model_id))
            if progress:
                progress(index + 1, len(sources))
        return results
