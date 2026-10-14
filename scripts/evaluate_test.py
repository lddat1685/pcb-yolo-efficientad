#!/usr/bin/env python3
"""CLI for Step 7: Freeze Configuration and Final Test Evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pcb_lab.evaluation.test_eval import freeze_evaluation_config, run_test_evaluation


def main():
    parser = argparse.ArgumentParser(description="Step 7: Freeze Configuration and Final Test Evaluation")
    parser.add_argument("--freeze-only", action="store_true", help="Only freeze evaluation configuration")
    parser.add_argument("--run", action="store_true", help="Run test evaluation using frozen config")
    parser.add_argument("--all", action="store_true", help="Freeze configuration and run test evaluation")
    parser.add_argument("--frozen-config", type=Path, default=None, help="Path to frozen configuration JSON")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to dataset root")
    parser.add_argument("--warmup", type=int, default=30, help="Number of warmup inference passes")
    parser.add_argument("--timing-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing output is never overwritten")

    args = parser.parse_args()

    if not (args.freeze_only or args.run or args.all):
        args.run = True

    frozen_path = args.frozen_config
    if args.freeze_only or args.all:
        print("[Step 7] Freezing evaluation configuration...")
        frozen_path = freeze_evaluation_config(
            out_path=args.frozen_config,
            dataset_root=args.dataset_root,
        )
        print(f"[Step 7] Configuration frozen at: {frozen_path}")
        if args.freeze_only:
            return 0

    if args.run or args.all:
        print(f"[Step 7] Running test evaluation with config: {frozen_path}...")
        result = run_test_evaluation(
            frozen_config_path=frozen_path,
            dataset_root=args.dataset_root,
            warmup_runs=args.warmup,
            timing_rounds=args.timing_rounds,
            output_dir=args.output_dir,
        )
        print(f"[Step 7] Test evaluation completed! Processed {result['predictions_count']} prediction records.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
