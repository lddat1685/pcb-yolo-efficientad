#!/usr/bin/env python3
"""Export YOLO and EfficientAD models and verify real TorchScript execution.

Attempts ultralytics export for YOLO (ONNX/OpenVINO via model.export()).
For EfficientAD, exports the torchscript via torch.jit.trace.
If export fails, keeps PyTorch float32 as reference standard.

Parity compares maps, scores, boxes and decisions for the AD TorchScript backend.
YOLO ONNX/OpenVINO are explicitly unverified by this checker; fallback never passes.
Target: score error Delta < 1e-3, 100% decision parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcb_lab.inference.registry import ModelRegistry
from pcb_lab.inference.sources import GalleryImageSource
from pcb_lab.inference.service import InferenceService
from pcb_lab.models.yolo.adapter import YoloDetector
from pcb_lab.models.efficientad.adapter import EfficientAdDetector, EfficientAdEngine
from pcb_lab.models.efficientad.architecture import EfficientAdModel


def export_yolo_onnx(artifact_dir: Path, output_dir: Path, device: str = "cpu") -> dict:
    """Export YOLO to ONNX via ultralytics. Returns export metadata."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        return {"status": "skipped", "reason": "ultralytics or torch not available"}

    model_path = artifact_dir / "best.pt"
    if not model_path.exists():
        return {"status": "skipped", "reason": "best.pt not found"}

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "yolo.onnx"

    try:
        model = YOLO(str(model_path))
        # Ultralytics export to ONNX
        exported_path = Path(model.export(
            imgsz=640,
            batch=1,
            device=device,
            format="onnx",
            half=False,
            save_dir=str(output_dir),
            name="yolo_onnx",
            opset=12
        ))
        onnx_path = exported_path.resolve()
        return {"status": "exported", "path": str(onnx_path), "backend": "onnx"}
    except Exception as e:
        return {"status": "fallback", "reason": str(e), "backend": "pytorch"}


def export_yolo_openvino(artifact_dir: Path, output_dir: Path, device: str = "cpu") -> dict:
    """Export YOLO to OpenVINO via ultralytics model.export(format='openvino')."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "skipped", "reason": "ultralytics not available"}

    model_path = artifact_dir / "best.pt"
    if not model_path.exists():
        return {"status": "skipped", "reason": "best.pt not found"}

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(str(model_path))
        exported_path = Path(model.export(
            imgsz=640,
            batch=1,
            device=device,
            format="openvino",
            half=False,
            save_dir=str(output_dir),
            name="yolo_openvino"
        ))
        xml_path = next(exported_path.glob("*.xml")) if exported_path.is_dir() else exported_path
        return {"status": "exported", "path": str(xml_path), "backend": "openvino"}
    except Exception as e:
        return {"status": "fallback", "reason": str(e), "backend": "pytorch"}


def export_efficientad_torchscript(artifact_dir: Path, output_dir: Path) -> dict:
    """Export EfficientAD to TorchScript via torch.jit.trace."""
    try:
        import torch
    except ImportError:
        return {"status": "skipped", "reason": "torch not available"}

    model_path = artifact_dir / "model.pt"
    if not model_path.exists():
        return {"status": "skipped", "reason": "model.pt not found"}

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "efficientad.pt"

    try:
        # Load model
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        model = EfficientAdModel()
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()

        # Trace with dummy input (1 tile, 256x256, 3 channels, float32, normalized)
        dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
        traced = torch.jit.trace(model, dummy_input, strict=False)
        traced.save(str(script_path))

        return {"status": "exported", "path": str(script_path), "backend": "torchscript"}
    except Exception as e:
        return {"status": "fallback", "reason": str(e), "backend": "pytorch"}


def _max_delta(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Backend returned non-finite output")
    return float(np.abs(reference - candidate).max()) if reference.size else 0.0


def verify_parity(samples_list, model_id, registry, artifact_dir_yolo, artifact_dir_ad,
                  delta_threshold=1e-3, n_samples=20, *, exported_ad_path=None):
    """Verify the actual TorchScript AD backend through the production pipeline.

    Hybrid YOLO remains PyTorch and is explicitly outside this export claim.
    YOLO-only ONNX/OpenVINO parity is UNVERIFIED until a checker runs that backend.
    """
    from pcb_lab.inference.service import LoadedModel
    import torch

    selected = list(samples_list)[:max(0, n_samples)]
    spec = registry.get(model_id)
    path = Path(exported_ad_path) if exported_ad_path else (
        REPO / "artifacts/runtime" / model_id / "efficientad/efficientad.pt")
    parity = dict(passed=False, status="UNVERIFIED", backend="torchscript",
                  backend_path=str(path), backend_sha256=None,
                  scope="EfficientAD TorchScript; hybrid YOLO remains PyTorch",
                  verified_samples=0, backend_executed_samples=0, failed_samples=0,
                  decision_matches=0, decision_flips=0, decision_parity=0.,
                  score_delta_max=None, score_delta_mean=None, tilemap_delta_max=None,
                  heatmap_delta_max=None, box_delta_max=None, details=[], failures=[])
    result = dict(total=n_samples, actual_samples=len(selected), parity=parity,
                  pytorch={"runs": []}, optimized={"runs": []})
    if spec.ad is None:
        parity["error"] = "No target AD backend; PyTorch fallback is not export verification"
        return result
    if n_samples <= 0 or len(selected) != n_samples:
        parity.update(status="FAILED", error="Insufficient parity sample coverage")
        return result
    try:
        scripted = torch.jit.load(str(path), map_location="cpu").eval()
        parity["backend_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as exc:
        parity.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        parity["failures"].append(dict(status="FAILED", error=parity["error"]))
        return result

    reference = InferenceService(registry)
    class RecordingEngine:
        def __init__(self, engine):
            self.engine, self.maps = engine, None
        def infer_tiles(self, tiles):
            self.maps = self.engine.infer_tiles(tiles)
            return self.maps
    try:
        original = reference._model(spec.ad).detector
        original.engine = RecordingEngine(original.engine)
        target_engine = RecordingEngine(EfficientAdEngine(scripted, "cpu"))
        target_ad = LoadedModel(EfficientAdDetector(target_engine, original.meta))
        def load_target(asset, device):
            return target_ad if asset.model_id == spec.ad.model_id else reference._model(asset)
        optimized = InferenceService(registry, model_loader=load_target)
    except Exception as exc:
        parity.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        parity["failures"].append(dict(status="FAILED", error=parity["error"]))
        return result

    deltas = []
    for sample in selected:
        detail = dict(sample_id=sample.sample_id, status="FAILED", error=None,
                      score_delta=None, decision_ts=None)
        try:
            source = GalleryImageSource(sample)
            py_run = reference.inspect(source, model_id)
            if py_run.decision == "ERROR":
                raise RuntimeError("Reference: " + str(py_run.error_reason))
            opt_run = optimized.inspect(source, model_id)
            if opt_run.decision == "ERROR":
                raise RuntimeError("Target: " + str(opt_run.error_reason))
            if py_run.efficientad_executed != opt_run.efficientad_executed:
                raise ValueError("Backend routing differs")
            sample_deltas = dict(score=0., tilemap=0., heatmap=0., box=0.)
            if opt_run.efficientad_executed:
                sample_deltas["tilemap"] = _max_delta(original.engine.maps, target_engine.maps)
                sample_deltas["heatmap"] = _max_delta(py_run.anomaly_map, opt_run.anomaly_map)
                sample_deltas["score"] = _max_delta(py_run.anomaly_score, opt_run.anomaly_score)
                def box_values(run):
                    return [[*b["xyxy_original"], b["confidence"]] for b in run.anomaly_regions]
                sample_deltas["box"] = _max_delta(box_values(py_run), box_values(opt_run))
                parity["backend_executed_samples"] += 1
            # Early exits are rerun in the target pipeline, never copied from reference.
            result["pytorch"]["runs"].append(py_run.to_dict())
            result["optimized"]["runs"].append(opt_run.to_dict())
            parity["verified_samples"] += 1
            parity["decision_matches"] += int(py_run.decision == opt_run.decision)
            parity["decision_flips"] += int(py_run.decision != opt_run.decision)
            deltas.append(sample_deltas)
            detail.update(status="VERIFIED", score_delta=sample_deltas["score"],
                          decision_py=py_run.decision, decision_ts=opt_run.decision,
                          early_exit=opt_run.early_exit, backend_executed=opt_run.efficientad_executed,
                          deltas=sample_deltas)
        except Exception as exc:
            detail["error"] = f"{type(exc).__name__}: {exc}"
            parity["failed_samples"] += 1
            parity["failures"].append(dict(detail))
        parity["details"].append(detail)
    if deltas:
        for key in ("score", "tilemap", "heatmap", "box"):
            parity[key + "_delta_max"] = max(d[key] for d in deltas)
        parity["score_delta_mean"] = float(np.mean([d["score"] for d in deltas]))
    parity["decision_parity"] = parity["decision_matches"] / n_samples
    parity["passed"] = bool(
        parity["verified_samples"] == n_samples and parity["backend_executed_samples"] > 0
        and parity["failed_samples"] == 0 and parity["decision_matches"] == n_samples
        and all(parity[k + "_delta_max"] is not None and
                parity[k + "_delta_max"] < min(delta_threshold, 1e-3)
                for k in ("score", "tilemap", "heatmap", "box")))
    parity["status"] = "PASSED" if parity["passed"] else "FAILED"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="H01", help="Model ID (H01, H00, B01, B03, etc.)")
    parser.add_argument("--split", default="calibration", choices=["calibration", "fusion", "test"],
                        help="Data split for parity verification")
    parser.add_argument("--output-dir", type=Path, default=REPO / "artifacts" / "runtime",
                        help="Output directory for exported models")
    parser.add_argument("--verify", action="store_true", help="Run parity verification after export")
    parser.add_argument("--n-samples", type=int, default=20, help="Samples for parity check")
    parser.add_argument("--onnx", action="store_true", help="Attempt ONNX export")
    parser.add_argument("--openvino", action="store_true", help="Attempt OpenVINO export")
    args = parser.parse_args()

    # Avoid double nesting when caller passes artifacts/runtime/<model> as --output-dir.
    if args.output_dir.name == args.model:
        output_dir = args.output_dir
    else:
        output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(REPO)
    spec = registry.get(args.model)

    print(f"[Step 9] Exporting {args.model} to runtime-optimized formats...")

    export_results = {}

    # YOLO export
    if spec.yolo:
        yolo_artifact = spec.yolo.checkpoint.parent

        if args.onnx or args.openvino:
            print(f"  Attempting ONNX export...")
            onnx_result = export_yolo_onnx(yolo_artifact, output_dir / "yolo")
            print(f"    ONNX: {onnx_result['status']}")
            export_results["yolo_onnx"] = onnx_result

            print(f"  Attempting OpenVINO export...")
            ov_result = export_yolo_openvino(yolo_artifact, output_dir / "yolo")
            print(f"    OpenVINO: {ov_result['status']}")
            export_results["yolo_openvino"] = ov_result
        else:
            # Default: try both
            onnx_result = export_yolo_onnx(yolo_artifact, output_dir / "yolo")
            ov_result = export_yolo_openvino(yolo_artifact, output_dir / "yolo")
            export_results["yolo_onnx"] = onnx_result
            export_results["yolo_openvino"] = ov_result

    # EfficientAD export
    if spec.ad:
        ad_artifact = spec.ad.checkpoint.parent
        print(f"  Attempting TorchScript export for EfficientAD...")
        ad_result = export_efficientad_torchscript(ad_artifact, output_dir / "efficientad")
        print(f"    TorchScript: {ad_result['status']}")
        export_results["efficientad_torchscript"] = ad_result

    # Save export report
    report = {
        "model_id": args.model,
        "timestamp": int(Path(__file__).resolve().stat().st_mtime) if Path(__file__).resolve().exists() else None,
        "exports": export_results,
        "fallback_backend": "pytorch_float32"
    }

    report_path = output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {report_path}")

    # Parity verification
    if args.verify:
        print(f"\n[Step 9] Parity verification on {args.n_samples} {args.split} samples...")
        samples = registry.gallery(args.split) if args.split != "test" else registry.gallery(args.split, report_mode=True)
        parity = verify_parity(samples, args.model, registry, spec.yolo.checkpoint.parent if spec.yolo else None,
                              spec.ad.checkpoint.parent if spec.ad else None, n_samples=args.n_samples,
                              exported_ad_path=output_dir / "efficientad/efficientad.pt")

        print(f"  Decision matches: {parity['parity']['decision_matches']}/{parity['total']}")
        print(f"  Decision flips: {parity['parity']['decision_flips']}")
        print(f"  Max score delta: {parity['parity']['score_delta_max']}")
        print(f"  Tilemap max delta: {parity['parity'].get('tilemap_delta_max')}")
        print(f"  Passed (Delta<1e-3 and 0 flips): {parity['parity'].get('passed', False)}")

        report["parity"] = parity
        # Persist parity to the same report file (previous version wrote before parity)
        # Convert non-serializable runs if needed
        def _to_serializable(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(x) for x in obj]
            return obj
        report_path.write_text(json.dumps(_to_serializable(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> updated {report_path} with parity")
        if not parity["parity"]["passed"]:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
