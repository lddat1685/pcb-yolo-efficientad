"""Extract calibration/fusion rows; smoke requires an explicit opt-in."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--partitions", nargs="+", default=["calibration", "fusion"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from pcb_lab.data.manifest import resolve_dataset_root
    from pcb_lab.models.efficientad.extract import extract_efficientad_predictions
    result = extract_efficientad_predictions(args.artifact, resolve_dataset_root(args.dataset_root),
        args.partitions, args.out_dir, allow_smoke=args.allow_smoke, limit=args.limit, device=args.device)
    print(json.dumps(result, indent=2))
    return 1 if any(p["n_errors"] for p in result.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
