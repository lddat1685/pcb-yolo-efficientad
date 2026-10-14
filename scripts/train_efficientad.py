"""Train B03 with a locked recipe; --smoke is an isolated one-epoch CPU run."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/efficientad_s.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--out-root", default=".")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-steps", type=int, default=70000, help="Max optimizer steps (default: 70000; 0 for no limit)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit samples per class in calibration report (e.g. 20)")
    args = parser.parse_args()
    if args.dataset_root:
        os.environ["DATASET_ROOT"] = args.dataset_root
    from pcb_lab.models.efficientad.train import train_efficientad
    max_steps = args.max_steps if args.max_steps > 0 else None
    result = train_efficientad(args.config, args.seed, args.out_root, args.device,
                              args.smoke, args.resume, args.allow_download,
                              max_steps=max_steps, val_limit=args.val_limit)
    print(json.dumps(asdict(result), default=str, indent=2))


if __name__ == "__main__":
    main()
