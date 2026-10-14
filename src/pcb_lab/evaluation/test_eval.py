"""Step 7: Final Test Evaluation and Benchmarking on Canonical Holdout Test Set.

Enforces zero-p-hacking by freezing evaluation configuration before evaluating test data.
Runs real pipeline inference on the 440 canonical test images, measuring real latency,
and computing overall, per-group, per-class, and routing metrics.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pcb_lab.data.manifest import output_path, resolve_dataset_root, sha256_file

CLASSES = [
    "open_circuit",
    "short",
    "mouse_bite",
    "spur",
    "spurious_copper",
    "pin_hole",
]

OPERATING_POINTS = {
    "B01": {"tau_yolo": 0.14, "tau_ad": None},
    "B02": {"tau_yolo": 0.38, "tau_ad": None},
    "B03": {"tau_yolo": None, "tau_ad": 1.4598},
    "H00": {"tau_yolo": 0.14, "tau_ad": 1.4598},
    "H01": {"tau_yolo": 0.14, "tau_ad": 1.4598},
    "H02": {"tau_yolo": 0.38, "tau_ad": 1.4598},
    "H01_sec": {"tau_yolo": 0.48, "tau_ad": 1.4598},
}
# Defaults are used only when creating a new freeze, never during evaluation.


def _frozen_operating_point(frozen_cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Return the locked operating point for model_id from the frozen config.

    Raises PermissionError when the frozen config does not contain the model.
    """
    ops = frozen_cfg.get("operating_points") or {}
    if model_id not in ops:
        raise PermissionError(f"Operating point for {model_id!r} missing from frozen config")
    return ops[model_id]


def _compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def freeze_evaluation_config(
    out_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """Create and freeze evaluation configuration before test evaluation.
    
    Verifies that all checkpoints exist and match shortlist specifications.
    Records checksum of the 440 test samples from holdout.json.
    """
    repo = Path(repo_root) if repo_root else Path.cwd()
    ds_root = resolve_dataset_root(dataset_root)
    out_file = Path(out_path) if out_path else repo / "artifacts/test/frozen_evaluation.json"
    if out_file.exists():
        raise FileExistsError(f"Refuse to overwrite frozen configuration: {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    holdout_path = ds_root / "benchmarks/deeppcb/splits/holdout.json"
    if not holdout_path.is_file():
        raise FileNotFoundError(f"holdout.json missing at {holdout_path}")

    with open(holdout_path, "r", encoding="utf-8") as f:
        holdout_data = json.load(f)

    test_ids = holdout_data.get("samples", {}).get("test", [])
    if len(test_ids) != 440:
        raise ValueError(f"Expected 440 test samples, found {len(test_ids)}")

    sorted_test_ids = sorted(test_ids)
    test_split_sha = hashlib.sha256(json.dumps(sorted_test_ids).encode("utf-8")).hexdigest()

    ckpt_b01 = repo / "artifacts/yolo/B01_yolo11n/seed42/best.pt"
    ckpt_b02 = repo / "artifacts/yolo/B02_yolo11s/seed42/best.pt"
    ckpt_b03 = repo / "artifacts/efficientad/B03_efficientad_s/seed42/model.pt"

    for p, name in [(ckpt_b01, "B01"), (ckpt_b02, "B02"), (ckpt_b03, "B03")]:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint for {name} missing at {p}")

    config = {
        "schema_version": 1,
        "status": "FROZEN",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_access_authorized": True,
        "split_name": "test",
        "sample_count": len(test_ids),
        "test_split_sha256": test_split_sha,
        "checkpoints": {
            "B01_yolo11n": {"path": ckpt_b01.relative_to(repo).as_posix(), "sha256": sha256_file(ckpt_b01)},
            "B02_yolo11s": {"path": ckpt_b02.relative_to(repo).as_posix(), "sha256": sha256_file(ckpt_b02)},
            "B03_efficientad_s": {"path": ckpt_b03.relative_to(repo).as_posix(), "sha256": sha256_file(ckpt_b03)},
        },
        "operating_points": OPERATING_POINTS,
        "recommended_demo_model": "H01_cascade_yolo11n_efficientad_s",
        "runtime_specs": {
            "device": "cpu",
            "batch_size": 1,
            "warmup_runs": 30,
            "seed": 42,
        },
    }

    with open(out_file, "x", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return out_file


def _calculate_binary_metrics(
    tp: int, tn: int, fp: int, fn: int,
    cost_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Calculate balanced accuracy, recall, precision, FRR, miss rate, F1, MCC, ECPB."""
    total_pos = tp + fn
    total_neg = fp + tn

    defect_recall = tp / total_pos if total_pos > 0 else 0.0
    miss_rate = fn / total_pos if total_pos > 0 else 0.0
    frr = fp / total_neg if total_neg > 0 else 0.0
    tnr = 1.0 - frr
    balanced_accuracy = (defect_recall + tnr) / 2.0
    defect_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    denom_f1 = (2 * tp + fp + fn)
    f1 = (2.0 * tp / denom_f1) if denom_f1 > 0 else 0.0

    mcc_num = float(tp * tn - fp * fn)
    mcc_den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (mcc_num / mcc_den) if mcc_den > 0.0 else 0.0

    # ECPB: N=10000, 1% defect, C_fn=100, C_fp=1
    fn_sim = 100.0 * miss_rate
    fp_sim = 9900.0 * frr
    ecpb = (100.0 * fn_sim + 1.0 * fp_sim) / 10000.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "balanced_accuracy": balanced_accuracy,
        "defect_recall": defect_recall,
        "defect_precision": defect_precision,
        "frr": frr,
        "miss_rate": miss_rate,
        "f1": f1,
        "mcc": mcc,
        "ecpb": ecpb,
    }


def _validate_frozen(frozen_cfg, repo, ds_root):
    """Reject tampering before loading models or decoding any test image."""
    if frozen_cfg.get("status") != "FROZEN":
        raise PermissionError("Evaluation status must be FROZEN")
    if frozen_cfg.get("test_access_authorized") is not True or frozen_cfg.get("split_name") != "test":
        raise PermissionError("Frozen configuration did not authorize test access")
    try:
        holdout = ds_root / "benchmarks/deeppcb/splits/holdout.json"
        test_ids = json.loads(holdout.read_text(encoding="utf-8"))["samples"]["test"]
        digest = hashlib.sha256(json.dumps(sorted(test_ids)).encode("utf-8")).hexdigest()
        if len(test_ids) != 440 or len(set(test_ids)) != 440 or frozen_cfg["sample_count"] != 440:
            raise PermissionError("Frozen test sample count must be 440 unique IDs")
        if digest != frozen_cfg["test_split_sha256"]:
            raise PermissionError("Test split SHA256 mismatch")
        for key, filename in (("B01_yolo11n", "best.pt"), ("B02_yolo11s", "best.pt"),
                              ("B03_efficientad_s", "model.pt")):
            entry = frozen_cfg["checkpoints"][key]
            checkpoint = (repo / entry["path"]).resolve()
            if not checkpoint.is_relative_to(repo.resolve()) or checkpoint.name != filename:
                raise PermissionError(f"Invalid checkpoint path for {key}")
            if sha256_file(checkpoint) != entry["sha256"]:
                raise PermissionError(f"Checkpoint SHA256 mismatch for {key}")
        from pcb_lab.inference.registry import MODEL_MATRIX
        for mid in MODEL_MATRIX:
            _frozen_operating_point(frozen_cfg, mid)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PermissionError(f"Invalid frozen evaluation: {exc}") from exc
    return test_ids


def _measure_inspection(service, source, model_id):
    """Wall clock starts before decode; synchronize CUDA at both boundaries."""
    device = str(service.device)
    use_cuda = device.startswith("cuda") or device.isdigit()
    cuda_device = "cuda:" + device if device.isdigit() else device
    if use_cuda:
        torch.cuda.synchronize(cuda_device)
    started = time.perf_counter()
    result = service.inspect(source, model_id)
    if use_cuda:
        torch.cuda.synchronize(cuda_device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if result.decision == "ERROR":
        raise RuntimeError(f"{model_id}/{result.sample_id}: {result.error_reason}")
    return result, elapsed_ms


def run_test_evaluation(
    frozen_config_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    warmup_runs: int = 30,
    timing_rounds: int = 3,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate seven real pipelines using a validated freeze and versioned output."""
    from pcb_lab.data.samples import ManifestDataset
    from pcb_lab.inference.registry import ModelRegistry
    from pcb_lab.inference.service import InferenceService
    from pcb_lab.inference.sources import GalleryImageSource

    repo = (Path(repo_root) if repo_root else Path.cwd()).resolve()
    ds_root = resolve_dataset_root(dataset_root)
    frozen_config_path = Path(frozen_config_path) if frozen_config_path else repo / "artifacts/test/frozen_evaluation.json"
    frozen_cfg = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    test_ids = _validate_frozen(frozen_cfg, repo, ds_root)
    if warmup_runs < 0 or timing_rounds < 1:
        raise ValueError("warmup_runs >= 0 and timing_rounds >= 1 required")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts_dir = Path(output_dir) if output_dir else repo / "artifacts/test" / ("e2e_" + stamp)
    artifacts_dir = output_path(ds_root, artifacts_dir)
    if artifacts_dir.exists():
        raise FileExistsError(f"Refuse to overwrite evaluation output: {artifacts_dir}")
    registry = ModelRegistry(repo, frozen_config_path=frozen_config_path)
    if registry.frozen != frozen_cfg:
        raise PermissionError("Frozen configuration changed during validation")
    device = frozen_cfg["runtime_specs"]["device"]
    if frozen_cfg["runtime_specs"].get("batch_size") != 1:
        raise PermissionError("Only the frozen batch=1 protocol is supported")
    torch.manual_seed(frozen_cfg["runtime_specs"]["seed"])
    service = InferenceService(registry, device=device)
    samples = sorted(ManifestDataset(ds_root, "test", allow_test=True), key=lambda sample: sample.sample_id)
    if [sample.sample_id for sample in samples] != sorted(test_ids):
        raise PermissionError("Manifest samples differ from frozen test split")
    # Warm every registered pipeline on a development image, after loading all assets.
    warm_sample = next(iter(ManifestDataset(ds_root, "calibration")))
    cold_start = {}
    for key in ("B01", "B02", "B03"):
        asset = registry.get(key).yolo or registry.get(key).ad
        tick = time.perf_counter()
        service._model(asset)
        cold_start[key] = (time.perf_counter() - tick) * 1000
    cold_pipeline = {}
    for model_id in registry.specs:
        _, cold_pipeline[model_id] = _measure_inspection(service, GalleryImageSource(warm_sample), model_id)
        for _ in range(warmup_runs):
            _measure_inspection(service, GalleryImageSource(warm_sample), model_id)
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=2, status="RUNNING", measurement="wall_clock_e2e_including_decode",
                    frozen_config_sha256=sha256_file(frozen_config_path), frozen_config=frozen_cfg,
                    warmup_runs=warmup_runs, timing_rounds=timing_rounds, device=device,
                    precision="float32", batch_size=1, cold_start_model_load_ms=cold_start,
                    cold_pipeline_first_call_ms=cold_pipeline,
                    torch_threads=torch.get_num_threads())
    (artifacts_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw_eval_data = []
    raw_path = artifacts_dir / "pipeline_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for idx, sample in enumerate(samples):
            row = dict(sample_id=sample.sample_id, source_group=sample.source_group,
                       is_defect=sample.is_defect,
                       gt_boxes=[dict(class_id=b.class_id, class_name=b.class_name, xyxy=list(b.xyxy)) for b in sample.boxes],
                       pipelines={})
            for model_id in registry.specs:
                runs, latencies = [], []
                for timing_round in range(timing_rounds):
                    result, elapsed_ms = _measure_inspection(service, GalleryImageSource(sample), model_id)
                    payload = result.to_dict()
                    payload.update(timing_round=timing_round, latency_e2e_ms=elapsed_ms)
                    stream.write(json.dumps(payload, allow_nan=False) + "\n")
                    # Keep compact records; pixel arrays are never accumulated.
                    runs.append(payload)
                    latencies.append(elapsed_ms)
                if any((r["decision"], r["efficientad_executed"], r["yolo_image_score"], r["anomaly_score"]) !=
                       (runs[0]["decision"], runs[0]["efficientad_executed"], runs[0]["yolo_image_score"], runs[0]["anomaly_score"]) for r in runs[1:]):
                    raise RuntimeError(f"Nondeterministic pipeline: {model_id}/{sample.sample_id}")
                row["pipelines"][model_id] = dict(result=runs[0], latencies=latencies)
            row.update(b01_score=row["pipelines"]["B01"]["result"]["yolo_image_score"],
                       b02_score=row["pipelines"]["B02"]["result"]["yolo_image_score"])
            raw_eval_data.append(row)
            stream.flush()
            if (idx + 1) % 10 == 0:
                print(f"Evaluated {idx + 1}/{len(samples)} samples, {timing_rounds} rounds", flush=True)

    # Evaluate each method
    models = ["B01", "B02", "B03", "H00", "H01", "H02", "H01_sec"]
    summary_rows = []
    group_rows = []
    predictions_rows = []

    for model_id in models:
        # M01 FIX: thresholds come from the frozen config, never the module global.
        op = _frozen_operating_point(frozen_cfg, model_id)
        tau_y = op["tau_yolo"]
        tau_a = op["tau_ad"]

        sample_results = []
        for r in raw_eval_data:
            pipeline = r["pipelines"][model_id]
            result = pipeline["result"]
            sample_results.append(dict(
                sample_id=r["sample_id"], model_id=model_id, split="test", source_group=r["source_group"],
                is_defect=r["is_defect"], decision=result["decision"],
                yolo_score=result["yolo_image_score"], ad_score=result["anomaly_score"],
                route_ad=result["efficientad_executed"], early_exit=result["early_exit"],
                reason=result["reason"], latency_ms=float(np.mean(pipeline["latencies"])),
                latency_rounds_ms=pipeline["latencies"], timing_ms=result["timing_ms"],
                detections=result["detection_candidates"], anomaly_regions=result["anomaly_regions"],
                ground_truth_boxes=r["gt_boxes"], metadata=result["metadata"],
                config_hash=registry.config_hash, error_reason=result["error_reason"]))
            predictions_rows.append(sample_results[-1])

        # Overall confusion & metrics
        tp = sum(1 for s in sample_results if s["is_defect"] and s["decision"] == "DEFECT")
        tn = sum(1 for s in sample_results if not s["is_defect"] and s["decision"] == "GOOD")
        fp = sum(1 for s in sample_results if not s["is_defect"] and s["decision"] == "DEFECT")
        fn = sum(1 for s in sample_results if s["is_defect"] and s["decision"] == "GOOD")

        base_metrics = _calculate_binary_metrics(tp, tn, fp, fn)

        # Routing metrics
        routed_total = sum(1 for s in sample_results if s["route_ad"])
        routed_good = sum(1 for s in sample_results if not s["is_defect"] and s["route_ad"])
        routed_defect = sum(1 for s in sample_results if s["is_defect"] and s["route_ad"])

        route_rate_total = routed_total / len(sample_results) if sample_results else 0.0
        route_rate_good = routed_good / (tn + fp) if tn + fp else 0.
        route_rate_defect = routed_defect / (tp + fn) if tp + fn else 0.

        rescued_defects = 0
        rescue_rate = "N/A"
        added_false_rejects = 0

        if model_id in ("H01", "H01_sec", "H02"):
            # Rescued: YOLO missed (score < tau_y) but AD hit (score >= tau_a) on defect
            rescued_defects = sum(1 for s in sample_results if s["is_defect"] and s["reason"] == "ad_hit")
            # YOLO fn at this tau_y
            y_fn = sum(1 for s in sample_results if s["is_defect"] and s["yolo_score"] < tau_y)
            rescue_rate = (rescued_defects / y_fn) if y_fn > 0 else "N/A"
            # Added false rejects: good samples where YOLO < tau_y but AD >= tau_a
            added_false_rejects = sum(1 for s in sample_results if not s["is_defect"] and s["reason"] == "ad_hit")

        latencies_ms = [latency for s in sample_results for latency in s["latency_rounds_ms"]]
        lat_mean = float(np.mean(latencies_ms))
        lat_p50 = float(np.median(latencies_ms))
        lat_p95 = float(np.percentile(latencies_ms, 95))

        unclassified_pos = sum(1 for s in sample_results if s["decision"] == "DEFECT" and s["reason"] == "ad_hit")
        total_pos_dec = sum(1 for s in sample_results if s["decision"] == "DEFECT")
        unclass_rate = (unclassified_pos / total_pos_dec) if total_pos_dec > 0 else 0.0

        summary_rows.append({
            "model_id": model_id,
            "balanced_accuracy": base_metrics["balanced_accuracy"],
            "defect_recall": base_metrics["defect_recall"],
            "defect_precision": base_metrics["defect_precision"],
            "f1": base_metrics["f1"],
            "mcc": base_metrics["mcc"],
            "frr": base_metrics["frr"],
            "miss_rate": base_metrics["miss_rate"],
            "ecpb": base_metrics["ecpb"],
            "latency_mean_ms": lat_mean,
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "route_rate_total": route_rate_total if model_id != "B01" and model_id != "B02" else "N/A",
            "route_rate_good": route_rate_good if model_id != "B01" and model_id != "B02" else "N/A",
            "route_rate_defect": route_rate_defect if model_id != "B01" and model_id != "B02" else "N/A",
            "rescued_defects": rescued_defects if "h" in model_id.lower() else "N/A",
            "rescue_rate": rescue_rate if "h" in model_id.lower() else "N/A",
            "added_false_rejects": added_false_rejects if "h" in model_id.lower() else "N/A",
            "unclassified_positive_rate": unclass_rate if "h" in model_id.lower() or model_id == "B03" else 0.0,
        })

        # Derive group counts from the executed records.
        grp_metrics_map = {}
        groups = sorted({s["source_group"] for s in sample_results})
        for grp in groups:
            grp_samples = [s for s in sample_results if s["source_group"] == grp]
            g_tp = sum(1 for s in grp_samples if s["is_defect"] and s["decision"] == "DEFECT")
            g_tn = sum(1 for s in grp_samples if not s["is_defect"] and s["decision"] == "GOOD")
            g_fp = sum(1 for s in grp_samples if not s["is_defect"] and s["decision"] == "DEFECT")
            g_fn = sum(1 for s in grp_samples if s["is_defect"] and s["decision"] == "GOOD")
            g_base = _calculate_binary_metrics(g_tp, g_tn, g_fp, g_fn)
            grp_metrics_map[grp] = g_base

        worst_recall = min(gm["defect_recall"] for gm in grp_metrics_map.values())
        worst_frr = max(gm["frr"] for gm in grp_metrics_map.values())

        for grp in groups:
            gm = grp_metrics_map[grp]
            group_rows.append({
                "model_id": model_id,
                "source_group": grp,
                "sample_count": gm["tp"] + gm["tn"] + gm["fp"] + gm["fn"],
                "good_count": gm["tn"] + gm["fp"],
                "defect_count": gm["tp"] + gm["fn"],
                "defect_recall": gm["defect_recall"],
                "frr": gm["frr"],
                "balanced_accuracy": gm["balanced_accuracy"],
                "worst_group_recall": worst_recall,
                "worst_group_frr": worst_frr,
            })

    from pcb_lab.evaluation.detection import evaluate_detections
    per_class_rows = []
    for model_id in ("B01", "B02", "H00", "H01", "H02", "H01_sec"):
        rows = [r for r in predictions_rows if r["model_id"] == model_id]
        per_class_rows.extend(evaluate_detections(rows, CLASSES,
            frozen_cfg["operating_points"][model_id]["tau_yolo"], model_id=model_id))
    for summary in summary_rows:
        classes = [r for r in per_class_rows if r["model_id"] == summary["model_id"]]
        summary["map50"] = float(np.mean([r["ap_50"] for r in classes])) if classes else None
        summary["map50_95"] = float(np.mean([r["ap_50_95"] for r in classes])) if classes else None

    # 1. predictions_test.jsonl
    pred_path = artifacts_dir / "predictions_test.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions_rows:
            f.write(json.dumps(p) + "\n")

    # 2. benchmark_summary
    sum_json = artifacts_dir / "benchmark_summary.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    sum_csv = artifacts_dir / "benchmark_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3. benchmark_per_group
    grp_json = artifacts_dir / "benchmark_per_group.json"
    with open(grp_json, "w", encoding="utf-8") as f:
        json.dump(group_rows, f, indent=2)

    grp_csv = artifacts_dir / "benchmark_per_group.csv"
    with open(grp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    # 4. benchmark_per_class
    cls_json = artifacts_dir / "benchmark_per_class.json"
    with open(cls_json, "w", encoding="utf-8") as f:
        json.dump(per_class_rows, f, indent=2)

    cls_csv = artifacts_dir / "benchmark_per_class.csv"
    with open(cls_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_class_rows)

    # 5. Markdown Report
    report_path = artifacts_dir / "step7_final_test_evaluation_report.md"
    _write_report(report_path, summary_rows, group_rows, per_class_rows, frozen_cfg)

    protocol["status"] = "COMPLETE"
    protocol["raw_runs_sha256"] = sha256_file(raw_path)
    (artifacts_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(f"Artifacts successfully written to {artifacts_dir} and {report_path}")

    return {
        "output_dir": str(artifacts_dir),
        "summary": summary_rows,
        "per_group": group_rows,
        "per_class": per_class_rows,
        "predictions_count": len(predictions_rows),
    }


def _write_report(
    report_path: Path,
    summary: List[Dict[str, Any]],
    per_group: List[Dict[str, Any]],
    per_class: List[Dict[str, Any]],
    frozen_cfg: Dict[str, Any],
) -> None:
    lines = []
    lines.append("# Báo cáo Đánh giá Test Cuối cùng (Step 7: Final Test Evaluation Report)")
    lines.append("")
    lines.append(f"Ngày lập: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}. Trạng thái: **Đã hoàn tất kiểm chứng test**.")
    lines.append("")
    lines.append("## 1. Xác thực Khóa Cấu hình (Frozen Evaluation Verification)")
    lines.append(f"- **Split:** `{frozen_cfg['split_name']}` ({frozen_cfg['sample_count']} ảnh: 220 good, 220 defect).")
    lines.append(f"- **SHA-256 tập test:** `{frozen_cfg['test_split_sha256']}`")
    lines.append(f"- **Checkpoint B01 (YOLO11n):** `{frozen_cfg['checkpoints']['B01_yolo11n']['sha256'][:16]}...`")
    lines.append(f"- **Checkpoint B02 (YOLO11s):** `{frozen_cfg['checkpoints']['B02_yolo11s']['sha256'][:16]}...`")
    lines.append(f"- **Checkpoint B03 (EfficientAD-S):** `{frozen_cfg['checkpoints']['B03_efficientad_s']['sha256'][:16]}...`")
    lines.append(f"- **Mô hình đề xuất cho demo:** `{frozen_cfg['recommended_demo_model']}`")
    lines.append("")
    lines.append("## 2. Bảng Tổng hợp Kết quả Test ở Mức Ảnh (PLAN §7.2)")
    lines.append("")
    lines.append("| Model | Bal. Acc | Recall | Precision | FRR | Miss Rate | F1 | MCC | ECPB | Mean Latency (ms) | p95 Latency (ms) | Route % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        route_str = f"{r['route_rate_total']*100:.1f}%" if isinstance(r['route_rate_total'], (int, float)) else str(r['route_rate_total'])
        lines.append(
            f"| **{r['model_id']}** | {r['balanced_accuracy']*100:.2f}% | {r['defect_recall']*100:.2f}% | "
            f"{r['defect_precision']*100:.2f}% | {r['frr']*100:.2f}% | {r['miss_rate']*100:.2f}% | "
            f"{r['f1']:.4f} | {r['mcc']:.4f} | {r['ecpb']:.4f} | {r['latency_mean_ms']:.1f} | "
            f"{r['latency_p95_ms']:.1f} | {route_str} |"
        )
    lines.append("")
    lines.append("## 3. Phân tích Routing và Hiệu quả Định tuyến (H01 vs H00)")
    h00 = next(r for r in summary if r["model_id"] == "H00")
    h01 = next(r for r in summary if r["model_id"] == "H01")
    delta_rec = (h01["defect_recall"] - h00["defect_recall"]) * 100.0
    delta_frr = (h01["frr"] - h00["frr"]) * 100.0
    delta_lat = h01["latency_mean_ms"] - h00["latency_mean_ms"]

    lines.append(f"- **Δ Recall (H01 - H00):** {delta_rec:+.2f}%")
    lines.append(f"- **Δ FRR (H01 - H00):** {delta_frr:+.2f}%")
    lines.append(f"- **Δ Latency Mean (H01 - H00):** {delta_lat:+.1f} ms")
    lines.append(f"- **Số lượng mẫu Early-Exit:** {int(440 * (1.0 - h01['route_rate_total']))} / 440 ảnh ({ (1.0 - h01['route_rate_total'])*100:.1f}%)")
    lines.append(f"- **Khuyết tật được cứu (Rescued Defects):** {h01['rescued_defects']}")
    lines.append(f"- **Số Good bị báo nhầm tăng thêm (Added False Rejects):** {h01['added_false_rejects']}")
    lines.append("")
    lines.append("## 4. Phân tích Phân rã theo Nhóm Nguồn (Source Group Breakdown)")
    lines.append("")
    lines.append("| Model | Source Group | Samples | Good | Defect | Recall | FRR | Bal. Acc | Worst-Group Recall | Worst-Group FRR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in per_group:
        lines.append(
            f"| {g['model_id']} | {g['source_group']} | {g['sample_count']} | {g['good_count']} | {g['defect_count']} | "
            f"{g['defect_recall']*100:.2f}% | {g['frr']*100:.2f}% | {g['balanced_accuracy']*100:.2f}% | "
            f"{g['worst_group_recall']*100:.2f}% | {g['worst_group_frr']*100:.2f}% |"
        )
    lines.append("")
    lines.append("## 5. Phân tích Phát hiện 6 Lớp Khuyết tật (Per-Class Detection)")
    lines.append("")
    lines.append("| Model | Class Name | GT Boxes | Detected Boxes | AP@0.5 | AP@0.5:0.95 | Recall at frozen threshold |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for c in per_class:
        lines.append(
            f"| {c['model_id']} | `{c['class_name']}` | {c['gt_box_count']} | {c['detected_box_count']} | "
            f"{c['ap_50']:.4f} | {c['ap_50_95']:.4f} | {c['class_recall']*100:.2f}% |"
        )
    lines.append("")
    lines.append("## 6. Kết luận và Khuyến nghị Triển khai")
    lines.append("- Toàn bộ 440 ảnh test đã được đánh giá độc lập theo đúng thỏa ước frozen evaluation.")
    b01 = next(r for r in summary if r["model_id"] == "B01")
    saving = 100 * (1 - h01["latency_mean_ms"] / h00["latency_mean_ms"])
    speedup = h00["latency_p50_ms"] / h01["latency_p50_ms"]
    lines.append(f"- H01 cứu thêm {h01['rescued_defects']} ảnh lỗi; recall YOLO11n = {b01['defect_recall']*100:.2f}% tại ngưỡng frozen.")
    lines.append(f"- So với H00 OR chạy cả hai nhánh, H01 tiết kiệm {saving:.1f}% latency mean; p50 nhanh hơn {speedup:.1f}× nhờ early-exit (E2E có decode).")
    lines.append("- Đây là phép đo mới có version; số latency cộng nhánh lịch sử không phải phép đo E2E. Không có bằng chứng H01 tối ưu hơn YOLO đơn.")


    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Public spelling retained for callers using the audit terminology.
evaluate_test_split = run_test_evaluation
