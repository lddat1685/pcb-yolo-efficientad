"""CPU audit entry point; usable directly without an editable install."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcb_lab.data.audit import run_audit, write_reports
from pcb_lab.data.eda import render_overlays
from pcb_lab.data.manifest import DatasetInputError, output_path, resolve_dataset_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only PCB dataset audit and development EDA")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = resolve_dataset_root(args.dataset_root)
        out = output_path(root, args.out_dir)
        report = run_audit(root, out)
        if not args.no_overlays:
            overlays = render_overlays(root, out)
            overlay_errors = json.loads((out / "overlay_errors.json").read_text(encoding="utf-8"))
            overlay_warnings = json.loads((out / "overlay_warnings.json").read_text(encoding="utf-8"))
            report["overlays"] = {"n_rendered": len(overlays), "partitions": ["train", "calibration"]}
            report["checks"].append({"id": "overlay_render", "status": "FAIL" if overlay_errors else "PASS",
                                     "n_violations": len(overlay_errors), "detail": "Development overlay coverage and rendering."})
            report["checks"].append({"id": "overlay_coverage", "status": "WARN" if overlay_warnings else "PASS",
                                     "n_violations": len(overlay_warnings), "detail": "Up to five examples per class per development partition."})
            report["checks"].sort(key=lambda c: c["id"])
            report["errors"] = sorted(report["errors"] + overlay_errors,
                                      key=lambda e: (e["sample_id"] or "", e["stage"], e["message"]))
            profile = (out / "data_profile.md").read_text(encoding="utf-8")
            if overlay_errors:
                report["overall"] = "FAIL"
                profile = profile.replace("Kết quả kiểm toán: **PASS**", "Kết quả kiểm toán: **FAIL**")
            profile += f"\nOverlay: {len(overlays)} ảnh train/calibration; {len(overlay_errors)} lỗi; {len(overlay_warnings)} cảnh báo thiếu mẫu. "
            profile += "Chi tiết: overlay_errors.json, overlay_warnings.json và overlay_index.json.\n"
            write_reports(report, profile, out)
        print(json.dumps({"overall": report["overall"], "images": report["counts"]["total"]["images"],
                          "errors": len(report["errors"]), "out_dir": str(out)}, ensure_ascii=True))
        return 0 if report["overall"] == "PASS" else 1
    except (DatasetInputError, OSError, UnicodeError) as exc:
        print(f"Audit cannot run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
