"""Write the shared prepare_input metadata and digest (no image export)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcb_lab.data.manifest import artifact_path, output_path, resolve_dataset_root
from pcb_lab.inference.preprocessing import preview_input


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", choices=("yolo", "ad_tile", "ad_resize"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        parent = output_path(resolve_dataset_root(), args.out_json.parent)
        destination = artifact_path(parent, args.out_json.name)
        if destination == args.image.resolve():
            raise ValueError("Output must not overwrite the input image")
        report = preview_input(args.image, args.mode)
        parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8", newline="\n")
        print(report["sha256"])
        return 0
    except (OSError, ValueError, PermissionError) as exc:
        print(f"Preview cannot run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
