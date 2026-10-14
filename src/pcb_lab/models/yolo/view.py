"""Build a YOLO data view from the canonical manifest (train + calibration only).

The view is the *only* thing Ultralytics ever sees: image hard links, YOLO-format
label files, and a ``data.yaml``. The dataset itself is never written to. The number
of base images is exactly the manifest count (1.792 train, 460 calibration) so the
mandatory invariant (Step 3 contract §"Bất biến") holds; augmentation is applied
online at train time, never by expanding the view.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pcb_lab.data.manifest import (
    METADATA_PATHS,
    PARTITIONS,
    DatasetInputError,
    dataset_path,
    load_dataset,
    output_path,
    resolve_dataset_root,
    sha256_file,
)
from pcb_lab.data.samples import ManifestDataset


class ViewError(PermissionError):
    """Raised when a forbidden partition (calibration/fusion/test) is requested."""


# Only train and calibration may feed YOLO training/selection.
ALLOWED_PARTITIONS = ("train", "calibration")
FORBIDDEN = ("fusion", "test")


def _reject_forbidden(partitions) -> None:
    bad = [p for p in partitions if p in FORBIDDEN]
    if bad:
        raise ViewError(f"YOLO view refuses partition(s) {bad}; only {ALLOWED_PARTITIONS} allowed")


@dataclass(frozen=True)
class YoloView:
    data_yaml: Path
    counts: dict  # partition -> {images, good, defect}
    box_counts: dict  # partition -> {class_name: n}
    link_mode: str
    view_signature: str

    def to_dict(self) -> dict:
        return {
            "data_yaml": str(self.data_yaml),
            "counts": self.counts,
            "box_counts": self.box_counts,
            "link_mode": self.link_mode,
            "view_signature": self.view_signature,
        }


def _write_labels_and_collect(out_dir: Path, samples, partition: str, limit, link):
    """Write label files and hard-link/copy images; return counts + per-class box counts."""
    img_out = out_dir / "images" / partition
    lbl_out = out_dir / "labels" / partition
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    selected = list(samples)
    if limit is not None:
        good_n = limit.get("good")
        defect_n = limit.get("defect")
        good = [s for s in selected if not s.is_defect]
        defect = [s for s in selected if s.is_defect]
        if good_n is not None:
            good = good[:good_n]
        if defect_n is not None:
            defect = defect[:defect_n]
        selected = good + defect
    selected.sort(key=lambda s: s.sample_id)

    expected_images = {s.image_path.name for s in selected}
    expected_labels = {s.image_path.stem + ".txt" for s in selected}
    # Remove stale entries only within the two generated partition directories.
    for directory, expected in ((img_out, expected_images), (lbl_out, expected_labels)):
        if not directory.resolve().is_relative_to(out_dir.resolve()):
            raise ValueError("View directory escapes output root")
        for entry in directory.iterdir():
            if entry.name not in expected and (entry.is_file() or entry.is_symlink()):
                entry.unlink()

    counts = {"images": len(selected), "good": 0, "defect": 0}
    box_counts: dict[str, int] = {}
    label_hashes: list[str] = []
    sample_lines: list[str] = []
    link_modes: set[str] = set()

    for sample in selected:
        if sample.is_defect:
            counts["defect"] += 1
        else:
            counts["good"] += 1
        stem = Path(sample.image_path).stem
        img_target = img_out / f"{stem}{sample.image_path.suffix}"
        # Replace the directory entry, never overwrite a previous hardlink's
        # bytes: rebuilding with copy must detach the view from its source.
        if img_target.exists() or img_target.is_symlink():
            img_target.unlink()
        if link == "copy":
            shutil.copy2(sample.image_path, img_target)
            link_modes.add("copy")
        else:
            try:
                os.link(sample.image_path, img_target)
            except OSError:
                shutil.copy2(sample.image_path, img_target)
                link_modes.add("copy")
            else:
                link_modes.add("hardlink")
        # Labels (empty file for good samples).
        lbl_path = lbl_out / f"{stem}.txt"
        lines: list[str] = []
        for box in sample.boxes:
            w = sample.width
            h = sample.height
            cx = (box.xyxy[0] + box.xyxy[2]) / 2.0 / w
            cy = (box.xyxy[1] + box.xyxy[3]) / 2.0 / h
            bw = (box.xyxy[2] - box.xyxy[0]) / w
            bh = (box.xyxy[3] - box.xyxy[1]) / h
            lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            box_counts[box.class_name] = box_counts.get(box.class_name, 0) + 1
        # Canonical LF bytes keep view_signature identical on Windows and Colab.
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
        label_hashes.append(sha256_file(lbl_path))
        sample_lines.append(f"{sample.sample_id}\t{sample.sha256}\t{sha256_file(lbl_path)}")

    return counts, box_counts, sample_lines, label_hashes, link_modes


def _compute_view_signature(sample_lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in sorted(sample_lines):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_yolo_view(dataset_root, out_dir, partitions=("train", "calibration"),
                    link="hardlink", limit=None) -> YoloView:
    """Build a YOLO view from the manifest.

    Only ``train`` and ``calibration`` are allowed; ``fusion``/``test`` raise
    ``PermissionError``. Nothing is ever written under ``DATASET_ROOT``.
    """
    _reject_forbidden(partitions)
    for p in partitions:
        if p not in ALLOWED_PARTITIONS:
            raise ViewError(f"Partition {p!r} not allowed in YOLO view")

    dataset_root = resolve_dataset_root(dataset_root if dataset_root is not None else None)
    out_dir = output_path(dataset_root, Path(out_dir))
    out_dir = out_dir.resolve()
    if link not in {"hardlink", "copy"}:
        raise ValueError("link must be 'hardlink' or 'copy'")

    data = load_dataset(dataset_root)
    names = data.names  # 0..5 in canonical order

    counts: dict = {}
    box_counts: dict = {}
    all_sample_lines: list[str] = []
    link_modes: set[str] = set()

    for partition in partitions:
        samples = ManifestDataset(dataset_root, partition)
        c, bc, lines, _, modes = _write_labels_and_collect(out_dir, samples, partition, limit, link)
        link_modes.update(modes)
        counts[partition] = c
        box_counts[partition] = bc
        all_sample_lines.extend(lines)

    # Keep link_mode truthful even if linking fails only for some images: make
    # the complete selected view independent copies once any fallback occurs.
    if len(link_modes) > 1:
        for partition in partitions:
            _write_labels_and_collect(out_dir, ManifestDataset(dataset_root, partition),
                                      partition, limit, "copy")
    link_mode_used = "copy" if "copy" in link_modes else link

    view_signature = _compute_view_signature(all_sample_lines)

    data_yaml = out_dir / "data.yaml"
    yaml_data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/calibration",
        "names": {str(i): n for i, n in enumerate(names)},
    }
    # Explicit contract guarantee: no 'test' key.
    assert "test" not in yaml_data
    data_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True),
                         encoding="utf-8", newline="\n")

    # Persist a deterministic manifest of what was linked (for re-build idempotency checks).
    (out_dir / "view_manifest.txt").write_text(
        "\n".join(sorted(all_sample_lines)) + "\n", encoding="utf-8", newline="\n")

    return YoloView(data_yaml, counts, box_counts, link_mode_used, view_signature)
