#!/usr/bin/env python
"""Train a YOLO baseline (B01 YOLO11n / B02 YOLO11s) per the locked config.

Example:
    python scripts/train_yolo.py --config configs/models/yolo11n.yaml --seed 42 --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo's src importable when run as a plain script (no editable install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pcb_lab.models.yolo import train_yolo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a YOLO baseline for PCB defect detection.")
    ap.add_argument("--config", required=True, help="Path to a model config (e.g. configs/models/yolo11n.yaml)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--device", default=None, help="Torch device (e.g. 0, 'cpu'); default cpu")
    ap.add_argument("--out-root", default=".", help="Output root for runs/ and artifacts/ (default .)")
    ap.add_argument("--smoke", action="store_true", help="Smoke run: random init, 1 epoch, small view, no artifacts")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run dir")
    ap.add_argument("--allow-download", action="store_true", help="Fetch pretrained weights if missing")
    ap.add_argument("--amp", default=None, choices=("true", "false"),
                   help="Force AMP on/off; default: auto-enable on CUDA GPU, off on CPU")
    args = ap.parse_args()

    try:
        result = train_yolo(
            config_path=args.config, seed=args.seed, out_root=args.out_root,
            device=args.device, smoke=args.smoke, resume=args.resume,
            allow_download=args.allow_download,
            amp=(args.amp == "true") if args.amp is not None else None,
        )
    except Exception as exc:
        print(f"TRAIN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "run_dir": str(result.run_dir),
        "artifact_dir": str(result.artifact_dir) if result.artifact_dir else None,
        "smoke": result.smoke,
        "run_id": result.run_manifest.get("run_id"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
