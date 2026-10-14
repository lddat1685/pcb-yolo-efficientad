#!/usr/bin/env python3
"""Profile runtime performance across calibration samples using fine-grained profiler.

Measures decode, preprocessing, YOLO inference, EfficientAD tiling/inference/stitch/NMS,
and fusion decision. Results bit-identical to InferenceService.inspect().
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcb_lab.inference.registry import ModelRegistry
from pcb_lab.inference.sources import GalleryImageSource
from pcb_lab.inference.service import InferenceService
from pcb_lab.runtime.profiler import profile_inspect, ProfileSummary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="H01", help="Model ID (H01, H00, B01, B03, etc.)")
    parser.add_argument("--split", default="calibration", choices=["calibration", "fusion", "test"],
                        help="Data split to profile")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0 = no limit)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs before profiling")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    # Registry and gallery setup
    registry = ModelRegistry(REPO)

    # Test mode requires report_mode flag
    allow_test = args.split == "test"
    if allow_test:
        samples = registry.gallery(args.split, report_mode=True)
    else:
        samples = registry.gallery(args.split)

    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"[Step 9] Profiling {len(samples)} samples with {args.model} ({args.split})")

    # Profile using InferenceService
    service = InferenceService(registry)

    # Warmup: run unprofiled inspects so model load + first-inference
    # overhead (oneDNN/cuDNN autotune, tile cache) does not pollute timings.
    n_warmup = min(args.warmup, len(samples))
    if n_warmup > 0:
        print(f"  Warmup: {n_warmup} unprofiled runs...")
        for sample in samples[:n_warmup]:
            try:
                service.inspect(GalleryImageSource(sample), args.model)
            except Exception:
                pass

    from pcb_lab.runtime.profiler import _peak_rss_mb
    runs = []
    for idx, sample in enumerate(samples):
        source = GalleryImageSource(sample)
        run = profile_inspect(service, source, args.model)
        runs.append(run)
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(samples)} samples...")

    # Aggregate summary
    summary = ProfileSummary.aggregate(runs, args.model)

    # Output results
    result = {
        "model_id": args.model,
        "split": args.split,
        "n_samples": len(runs),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "registry_config_hash": registry.config_hash,
        "tau_yolo": registry.get(args.model).tau_yolo,
        "tau_ad": registry.get(args.model).tau_ad,
        "summary": summary.to_dict(),
        "runs": [r.to_dict() for r in runs]
    }

    # Write JSON
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {args.output}")

    # Write CSV summary
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "count", "total_ms", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "peak_memory_mb"])
            writer.writeheader()
            for name, stats in summary.stages.items():
                row = {"stage": name}
                row.update(stats.to_dict())
                writer.writerow(row)
        print(f"  -> {args.csv}")

    # Print key metrics
    print(f"\n[Step 9] Profile Results Summary:")
    print(f"  Samples profiled: {len(runs)}")
    print(f"  Model load time: {summary.stages['model_load'].mean_ms:.1f}ms (p50={summary.stages['model_load'].p50_ms:.1f}, p95={summary.stages['model_load'].p95_ms:.1f})")
    print(f"  YOLO latency (yolo): {summary.stages['yolo'].mean_ms:.1f}ms")
    print(f"  EfficientAD latency: {summary.stages['efficientad'].mean_ms:.1f}ms")
    print(f"  Total latency: {summary.stages['total'].mean_ms:.1f}ms")
    print(f"  Total core latency: {summary.stages['total_core'].mean_ms:.1f}ms")
    print(f"  Route rate (AD executed): {summary.route_rate:.1%}" if summary.route_rate else "  Route rate: N/A")
    print(f"  Early-exit rate: {summary.early_exit_rate:.1%}" if summary.early_exit_rate else "  Early-exit rate: N/A")
    decisions = summary.decisions
    print(f"  Decisions: {decisions}")

    return 0


if __name__ == "__main__":
    sys.exit(main())