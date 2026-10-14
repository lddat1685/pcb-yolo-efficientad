#!/usr/bin/env python
"""Extract raw YOLO predictions for calibration/fusion from a trained artifact.

Example:
    python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B01_yolo11n/seed42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pcb_lab.models.yolo import extract_predictions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract YOLO predictions (calibration/fusion).")
    ap.add_argument("--artifact", required=True, help="Artifact directory (e.g. artifacts/yolo/B01_yolo11n/seed42)")
    ap.add_argument("--dataset-root", default=None, help="DATASET_ROOT (defaults to config/env)")
    ap.add_argument("--partitions", nargs="+", default=["calibration", "fusion"],
                    help="Partitions to extract (default calibration fusion)")
    ap.add_argument("--out-dir", default=None, help="Output dir for preds_*.jsonl (default: artifact dir)")
    args = ap.parse_args()

    try:
        summary = extract_predictions(
            artifact_dir=args.artifact, dataset_root=args.dataset_root,
            partitions=tuple(args.partitions), out_dir=args.out_dir,
        )
    except Exception as exc:
        print(f"EXTRACT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
