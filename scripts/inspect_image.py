"""Inspect a local image or manifest sample using the same service as the UI."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcb_lab.inference.registry import ModelRegistry
from pcb_lab.inference.service import InferenceService
from pcb_lab.inference.sources import FileImageSource, GalleryImageSource, URLImageSource


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--sample-id")
    source.add_argument("--url")
    parser.add_argument("--split", choices=["calibration", "fusion", "test"], default="calibration")
    parser.add_argument("--report-mode", action="store_true", help="Allow locked test gallery")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--model", default="H01")
    args = parser.parse_args(argv)
    try:
        registry = ModelRegistry(args.repo_root)
        if args.sample_id:
            samples = registry.gallery(args.split, dataset_root=args.dataset_root, report_mode=args.report_mode)
            matches = [s for s in samples if s.sample_id == args.sample_id]
            if not matches:
                raise ValueError("Sample ID không có trong partition đã chọn")
            image_source = GalleryImageSource(matches[0])
        elif args.url:
            image_source = URLImageSource(args.url)
        else:
            image_source = FileImageSource(args.image, dataset_root=args.dataset_root)
        with redirect_stdout(sys.stderr):
            result = InferenceService(registry).inspect(image_source, args.model)
        print(json.dumps(result.to_dict(), ensure_ascii=True, allow_nan=False))
        return 2 if result.decision == "ERROR" else 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"decision": "ERROR", "error_reason": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
