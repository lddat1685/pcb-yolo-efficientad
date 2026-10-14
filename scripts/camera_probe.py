"""Diagnose IMV SDK / GigE enumeration; optionally capture an explicit camera key."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from pcb_lab.cameras.imv import IMVCamera


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", metavar="CAMERA_KEY", help="Key returned by enumeration")
    parser.add_argument("--output", type=Path, help="New PNG path; existing files are never overwritten")
    parser.add_argument("--software-trigger", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--exposure-us", type=float)
    args = parser.parse_args(argv)
    if bool(args.capture) != bool(args.output):
        parser.error("--capture and --output must be supplied together")
    try:
        if args.output and (args.output.suffix.lower() != ".png" or args.output.exists() or
                            args.output.with_suffix(".json").exists()):
            raise ValueError("Choose a new .png output path (PNG and JSON must not already exist)")
        with IMVCamera() as camera:
            devices = camera.enumerate()
            report = dict(status="OK" if devices else "NO_CAMERA", sdk_version=camera.version,
                          library=str(camera.library), cameras=[asdict(c) for c in devices])
            if args.capture:
                frame = camera.capture(args.capture, software_trigger=args.software_trigger,
                                       timeout_ms=args.timeout_ms, exposure_us=args.exposure_us)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as output:
                    Image.fromarray(frame.pixels).save(output, format="PNG")
                metadata = dict(frame.metadata, sensor_size=[frame.pixels.shape[1], frame.pixels.shape[0]])
                with args.output.with_suffix(".json").open("x", encoding="utf-8") as output:
                    json.dump(metadata, output, ensure_ascii=False, indent=2)
                report.update(status="CAPTURED", output=str(args.output), frame=metadata)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if devices else 2
    except Exception as exc:
        print(json.dumps(dict(status="ERROR", error=str(exc)), ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
