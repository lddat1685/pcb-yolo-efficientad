"""Stdlib-only verifier measurements. Never imports pcb_lab, numpy, torch or Ultralytics.

Run: python tests/helpers/step3_independent.py --dataset-root PATH [--smoke-run PATH]
Only manifest metadata, train labels, release.json and explicitly supplied checkpoints are opened.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(root, smoke_run=None):
    root = Path(root).resolve()
    manifest = root / "benchmarks/deeppcb/manifests/samples.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    counts, boxes, label_lines = {}, Counter(), Counter()
    for row in rows:
        part = row["split"]
        counts.setdefault(part, {"images": 0, "good": 0, "defect": 0})
        counts[part]["images"] += 1
        counts[part]["defect" if row["is_defect"] else "good"] += 1
        if part != "train":
            continue
        boxes.update(str(box["class_id"]) for box in row["boxes"])
        path = (root / row["label"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Train label escapes dataset")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid label: {path}")
                label_lines[fields[0]] += 1
    result = {"counts_from_manifest": counts, "train_boxes_manifest": dict(sorted(boxes.items())),
              "train_label_lines": dict(sorted(label_lines.items())),
              "train_box_total": sum(boxes.values()), "label_and_manifest_counts_match": boxes == label_lines,
              "manifest_file_sha256": digest(manifest),
              "release_file_sha256": digest(root / "benchmarks/deeppcb/release.json")}
    if smoke_run:
        smoke_run = Path(smoke_run).resolve()
        run = json.loads((smoke_run / "run_manifest.json").read_text(encoding="utf-8"))
        result["smoke_checkpoints"] = {}
        for name in ("best", "last"):
            checkpoint = smoke_run / "weights" / f"{name}.pt"
            actual = digest(checkpoint)
            expected = run["checkpoints"][f"{name}_sha256"]
            result["smoke_checkpoints"][name] = {"sha256": actual, "bytes": checkpoint.stat().st_size,
                                                    "matches_run_manifest": actual == expected}
        result["smoke_view_signature"] = run["view_signature"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--smoke-run")
    args = parser.parse_args()
    print(json.dumps(measure(args.dataset_root, args.smoke_run), indent=2, sort_keys=True))
