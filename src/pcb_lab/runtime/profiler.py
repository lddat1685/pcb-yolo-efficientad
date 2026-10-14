"""Fine-grained runtime profiler for the PCB hybrid inference pipeline.

Stage breakdown (PLAN Step 9):
- decode: image source get_frame (I/O + PIL decode + canonical validation)
- preprocess_canonical: to_canonical (RGB uint8 HWC)
- preprocess_letterbox: YOLO letterbox (resize + pad)
- yolo_infer: Ultralytics engine forward (predict on letterboxed input)
- yolo_postprocess: unletterbox + sort detections
- ad_split: TileManager.split (pad + 9 tiles for 640x640)
- ad_tile_infer: EfficientAdEngine.infer_tiles (9x forward)
- ad_stitch: TileManager.stitch (linear feathering to 640x640)
- ad_postprocess: threshold + morph open + connectedComponents + global NMS
- fusion_decision: routing rule (predict_cascade / predict_or)
- total_core: yolo + efficientad + fusion overhead (matches InferenceService)
- total: wall clock including model load + decode

Design: replicate adapter internals with timers instead of calling
predict() twice. Results are bit-identical to InferenceService because
the same functions are called in the same order.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Generator, Optional

import numpy as np

# Canonical stage order for reports.
STAGES = [
    "decode",
    "model_load",
    "preprocess_canonical",
    "preprocess_letterbox",
    "yolo",
    "yolo_postprocess",
    "ad_split",
    "ad_tile_infer",
    "ad_stitch",
    "ad_postprocess",
    "efficientad",
    "fusion_decision",
    "total_core",
    "total",
]

# Stages that roll up into InferenceService.timing_ms["yolo"].
YOLO_SUBSTAGES = ["preprocess_letterbox", "yolo", "yolo_postprocess"]
# Stages that roll up into InferenceService.timing_ms["efficientad"].
AD_SUBSTAGES = ["ad_split", "ad_tile_infer", "ad_stitch", "ad_postprocess"]


def _peak_rss_mb() -> float:
    """Current process RSS in MB; 0.0 when psutil/resource is unavailable."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource  # type: ignore

        # ru_maxrss is KiB on Linux, bytes on macOS.
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / 1024.0 if rss > 10_000_000 else rss / (1024 * 1024)
    except Exception:
        return 0.0


@dataclass
class StageStats:
    """Aggregated timing for one stage across N samples."""

    name: str
    samples_ms: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0

    def add(self, ms: float, memory_mb: float = 0.0) -> None:
        self.samples_ms.append(float(ms))
        if memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = float(memory_mb)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def total_ms(self) -> float:
        return float(sum(self.samples_ms))

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def _quantile(self, q: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    @property
    def p50_ms(self) -> float:
        return self._quantile(0.50)

    @property
    def p95_ms(self) -> float:
        return self._quantile(0.95)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms) if self.samples_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms) if self.samples_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }


@dataclass
class ProfileRun:
    """Per-sample fine-grained timings plus the inference outcome."""

    sample_id: str
    model_id: str
    timings_ms: dict = field(default_factory=dict)
    decision: str = "ERROR"
    reason: str = ""
    routing_trace: list[str] = field(default_factory=list)
    yolo_executed: bool = False
    efficientad_executed: bool = False
    early_exit: bool = False
    yolo_image_score: float | None = None
    anomaly_score: float | None = None
    n_yolo_boxes: int = 0
    n_ad_boxes: int = 0
    tile_count: int = 0
    error_reason: str | None = None
    peak_memory_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "model_id": self.model_id,
            "timings_ms": {k: round(float(v), 3) for k, v in self.timings_ms.items()},
            "decision": self.decision,
            "reason": self.reason,
            "routing_trace": list(self.routing_trace),
            "yolo_executed": self.yolo_executed,
            "efficientad_executed": self.efficientad_executed,
            "early_exit": self.early_exit,
            "yolo_image_score": self.yolo_image_score,
            "anomaly_score": self.anomaly_score,
            "n_yolo_boxes": self.n_yolo_boxes,
            "n_ad_boxes": self.n_ad_boxes,
            "tile_count": self.tile_count,
            "error_reason": self.error_reason,
            "peak_memory_mb": round(float(self.peak_memory_mb), 2),
        }


@dataclass
class ProfileSummary:
    """Aggregated profile across samples for one model/backend."""

    model_id: str
    backend: str = "pytorch"
    precision: str = "float32"
    device: str = "cpu"
    n_samples: int = 0
    stages: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    route_rate: float | None = None
    early_exit_rate: float | None = None
    n_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "precision": self.precision,
            "device": self.device,
            "n_samples": self.n_samples,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "decisions": dict(self.decisions),
            "route_rate": self.route_rate,
            "early_exit_rate": self.early_exit_rate,
            "n_errors": self.n_errors,
        }

    @classmethod
    def aggregate(
        cls,
        runs: list[ProfileRun],
        model_id: str,
        backend: str = "pytorch",
        precision: str = "float32",
        device: str = "cpu",
    ) -> "ProfileSummary":
        summary = cls(model_id=model_id, backend=backend, precision=precision, device=device)
        summary.n_samples = len(runs)
        for stage in STAGES:
            stats = StageStats(name=stage)
            summary.stages[stage] = stats
        for run in runs:
            for stage, ms in run.timings_ms.items():
                if stage not in summary.stages:
                    summary.stages[stage] = StageStats(name=stage)
                summary.stages[stage].add(ms, run.peak_memory_mb)
            summary.decisions[run.decision] = summary.decisions.get(run.decision, 0) + 1
            if run.error_reason:
                summary.n_errors += 1
        routed = sum(1 for r in runs if r.efficientad_executed)
        early = sum(1 for r in runs if r.early_exit)
        if runs:
            summary.route_rate = routed / len(runs)
            summary.early_exit_rate = early / len(runs)
        return summary


class _Timer:
    """Accumulates per-stage milliseconds for one sample."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.marks[name] = self.marks.get(name, 0.0) + (perf_counter() - start) * 1000.0

    def get(self, name: str) -> float:
        return self.marks.get(name, 0.0)


def profile_yolo_detector(detector: Any, image_rgb: np.ndarray, timer: _Timer) -> list:
    """Run YoloDetector.predict with sub-stage timers. Returns detections."""
    from pcb_lab.inference.preprocessing import (
        LetterboxMeta,
        prepare_input,
        to_canonical,
        unletterbox_boxes,
    )

    with timer.stage("preprocess_canonical"):
        canonical = to_canonical(image_rgb)
    with timer.stage("preprocess_letterbox"):
        prepared = prepare_input(canonical, "yolo", cfg=None)
        meta = prepared.meta
        letterbox_meta = LetterboxMeta(
            scale=meta["scale"],
            pad_left=meta["pad_left"],
            pad_top=meta["pad_top"],
            orig_w=meta["orig_w"],
            orig_h=meta["orig_h"],
            preprocessing_version=meta.get("preprocessing_version", "prep_v1"),
            preprocessing_hash=meta.get("preprocessing_hash", ""),
            color_order="rgb",
            normalized=False,
        )
    with timer.stage("yolo"):
        dets = detector.engine.infer(prepared.pixels)
    with timer.stage("yolo_postprocess"):
        if dets.size == 0:
            return []
        original = unletterbox_boxes(dets[:, :4], letterbox_meta)
        names = detector.engine.class_names
        detections = []
        for (x1, y1, x2, y2), (conf, cid) in zip(original, dets[:, 4:6]):
            cid_int = int(round(float(cid)))
            detections.append(
                {
                    "class_id": cid_int,
                    "class_name": names[cid_int] if cid_int < len(names) else str(cid_int),
                    "confidence": float(conf),
                    "xyxy_original": (float(x1), float(y1), float(x2), float(y2)),
                }
            )
        detections.sort(key=lambda d: (-d["confidence"], d["class_id"], *d["xyxy_original"]))
    return detections


def profile_ad_detector(detector: Any, image_rgb: np.ndarray, timer: _Timer) -> tuple[float, Any, list]:
    """Run EfficientAdDetector.predict with sub-stage timers.

    Returns (anomaly_score, heatmap, boxes) with Detection objects identical
    to EfficientAdDetector.predict.
    """
    import cv2

    from pcb_lab.inference.tiling import global_nms

    pixels = np.asarray(image_rgb)
    with timer.stage("ad_split"):
        tiles, specs = detector.tile_manager.split(pixels)
    with timer.stage("ad_tile_infer"):
        tile_maps = detector.engine.infer_tiles(tiles)
    with timer.stage("ad_stitch"):
        heatmap = detector.tile_manager.stitch(tile_maps, specs, pixels.shape[:2])
        score = float(heatmap.max())
    with timer.stage("ad_postprocess"):
        p = detector.infer_params
        mask = (heatmap > p["tau_pixel"]).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            if area > p["min_area"]:
                confidence = float(heatmap[labels == component].max())
                candidates.append((x, y, x + w, y + h, confidence))
        kept = global_nms(np.asarray(candidates, dtype=np.float64).reshape(-1, 5), p["nms_iou"])
        from pcb_lab.models.efficientad.adapter import Detection

        boxes = [
            Detection(None, "anomaly_unclassified", float(row[4]), tuple(map(float, row[:4])))
            for row in kept
        ]
    return score, heatmap, boxes


def profile_inspect(
    service: Any,
    source: Any,
    model_id: str = "H01",
) -> ProfileRun:
    """Profile one InferenceService.inspect call with fine-grained stages.

    Uses the service's own loaded detectors and registry spec so thresholds,
    routing and guards match production exactly. Returns a ProfileRun whose
    decision/scores equal InferenceService.inspect on the same input.
    """
    from dataclasses import asdict

    from pcb_lab.fusion.rules import predict_cascade, predict_or
    from pcb_lab.inference.preprocessing import preprocessing_hash
    from pcb_lab.inference.sources import validate_pixels

    run = ProfileRun(sample_id="unknown", model_id=model_id)
    timer = _Timer()
    total_start = perf_counter()
    try:
        spec = service.registry.get(model_id)
        with timer.stage("decode"):
            sample_id, image = source.get_frame()
            image = validate_pixels(image)
        run.sample_id = sample_id
        with timer.stage("model_load"):
            yolo_loaded = service._model(spec.yolo) if spec.yolo else None
            # For CASCADE, AD loads lazily only when routed; for OR/YOLO/AD-only
            # preload here so model_load timing stays comparable.
            ad_loaded = None
            if spec.ad and spec.mode in ("OR", "AD"):
                ad_loaded = service._model(spec.ad)

        yolo_dets: list = []
        yolo_score: float | None = None
        ad_score: float | None = None
        ad_boxes: list = []
        tile_count = 0

        if spec.yolo:
            run.routing_trace.append("YOLO")
            with yolo_loaded.lock:
                raw = profile_yolo_detector(yolo_loaded.detector, image, timer)
            run.yolo_executed = True
            yolo_score = max((float(d["confidence"]) for d in raw), default=0.0)
            run.yolo_image_score = yolo_score
            run.n_yolo_boxes = len([d for d in raw if d["confidence"] >= spec.tau_yolo])
            yolo_dets = [d for d in raw if d["confidence"] >= spec.tau_yolo]

        yolo_hit = spec.yolo is not None and (yolo_score or 0.0) >= spec.tau_yolo
        run.early_exit = spec.mode == "CASCADE" and bool(yolo_hit)

        if spec.ad and not run.early_exit:
            run.routing_trace.append("EfficientAD")
            if ad_loaded is None:
                with timer.stage("model_load"):
                    ad_loaded = service._model(spec.ad)
            with ad_loaded.lock:
                ad_score, heatmap, boxes = profile_ad_detector(ad_loaded.detector, image, timer)
            run.efficientad_executed = True
            run.anomaly_score = float(ad_score)
            run.tile_count = len(ad_loaded.detector.tile_manager.plan(*image.shape[:2]))
            tile_count = run.tile_count
            if ad_score >= spec.tau_ad:
                ad_boxes = [asdict(b) for b in boxes]
                run.n_ad_boxes = len(ad_boxes)

        with timer.stage("fusion_decision"):
            if spec.mode in {"CASCADE", "OR"}:
                base = dict(
                    sample_id=sample_id,
                    split=source.describe().get("split", "unlabelled"),
                    is_defect=source.describe().get("ground_truth") == "DEFECT",
                )
                yolo_row = dict(base, yolo_image_score=yolo_score, detections=yolo_dets)
                ad_row = dict(base, anomaly_score=ad_score or 0.0, boxes=ad_boxes)
                rule = predict_cascade if spec.mode == "CASCADE" else predict_or
                decision = rule(
                    yolo_row,
                    ad_row,
                    spec.tau_yolo,
                    spec.tau_ad,
                    allow_test=base.get("split") == "test",
                )
                run.decision, run.reason = decision.decision, decision.reason
                if spec.mode == "OR" and yolo_hit and (ad_score or 0.0) >= spec.tau_ad:
                    run.reason = "both_hit"
            else:
                positive = (
                    yolo_hit if spec.mode == "YOLO" else (ad_score or 0.0) >= spec.tau_ad
                )
                run.decision = "DEFECT" if positive else "GOOD"
                run.reason = (
                    ("yolo_hit" if spec.mode == "YOLO" else "ad_hit")
                    if positive
                    else "all_pass"
                )
        run.routing_trace.append("Early-exit" if run.early_exit else run.decision)

        # Roll up InferenceService-compatible totals.
        timer.marks["efficientad"] = sum(timer.get(s) for s in AD_SUBSTAGES)
        timer.marks["yolo_total"] = sum(timer.get(s) for s in YOLO_SUBSTAGES)
        # service["yolo"] historically = whole YOLO branch incl. preprocess.
        timer.marks["yolo"] = timer.get("preprocess_canonical") + timer.marks["yolo_total"]
        timer.marks["total_core"] = (
            timer.marks["yolo_total"] + timer.marks["efficientad"] + timer.get("fusion_decision")
        )
        void = (preprocessing_hash, tile_count)
        _ = void
    except Exception as exc:  # noqa: BLE001 - profiler must not crash the batch
        run.decision, run.reason = "ERROR", "inference_error"
        run.error_reason = f"{type(exc).__name__}: {exc}"
        run.routing_trace.append("ERROR")
        timer.marks.setdefault("total_core", 0.0)
    timer.marks["total"] = (perf_counter() - total_start) * 1000.0
    run.timings_ms = dict(timer.marks)
    run.peak_memory_mb = _peak_rss_mb()
    return run
