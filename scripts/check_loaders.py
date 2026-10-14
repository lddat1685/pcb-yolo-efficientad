"""Check one development batch per loader without decoding test images."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcb_lab.data.manifest import artifact_path, output_path, resolve_dataset_root
from pcb_lab.data.samples import check_loaders


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/loader_check.json"))
    args = parser.parse_args(argv)
    try:
        root = resolve_dataset_root(args.dataset_root)
        parent = output_path(root, args.out.parent)
        destination = artifact_path(parent, args.out.name)
        report = check_loaders(root)
        parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8", newline="\n")
        print(json.dumps({"overall": report["overall"], "errors": len(report["errors"])}, ensure_ascii=True))
        return 0 if report["overall"] == "PASS" else 1
    except (OSError, ValueError, PermissionError) as exc:
        print(f"Loader check cannot run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
