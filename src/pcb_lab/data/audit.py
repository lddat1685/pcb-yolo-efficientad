"""Deterministic full canonical audit; individual sample failures are data."""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

from PIL import Image

from .manifest import (
    PARTITIONS, artifact_path, dataset_path, finite_number, load_dataset,
    output_path, parse_yolo, sha256_file,
)

CHECKS = {
    "manifest_paths_exist": "Canonical image and label paths exist inside dataset root.",
    "release_sha256": "Every release.files entry matches its SHA-256.",
    "partition_counts": "Sample membership, groups and all pair aliases match holdout.",
    "image_decode": "Every canonical image fully decodes.",
    "image_size_640": "Every decoded image is 640 x 640.",
    "label_good_empty": "Good labels and manifest boxes are empty.",
    "label_defect_nonempty": "Defect labels and manifest boxes are nonempty.",
    "box_valid": "Finite positive boxes remain inside the image in both annotation formats.",
    "class_id_valid": "YOLO and manifest class IDs/names match the class schema.",
    "leak_source_group": "All partition pairs have disjoint source groups.",
    "leak_pair": "All partition pairs have disjoint unions of pair_ids.",
    "leak_image_hash": "All partition pairs have disjoint recomputed file hashes.",
    "leak_pixel_hash": "All partition pairs have disjoint recomputed RGB pixel hashes.",
    "manifest_schema": "Individual manifest records have the required field types.",
    "manifest_unique_ids": "Canonical sample IDs are unique.",
    "image_sha256": "Canonical file hashes match manifest.sha256.",
    "label_sha256": "Label file hashes match manifest.label_sha256.",
    "pixel_sha256": "RGB pixel hashes match manifest.pixel_sha256.",
    "image_metadata": "Decoded dimensions match manifest; canonical images are RGB.",
    "label_manifest_match": "YOLO labels agree with manifest boxes (normalized tolerance 1e-9).",
}


class Findings:
    def __init__(self):
        self.errors: list[dict] = []
        self.violations = {key: 0 for key in CHECKS}

    def fail(self, check: str, sample_id: str | None, message: str) -> None:
        self.violations[check] += 1
        self.errors.append({"sample_id": sample_id, "stage": check, "message": message})

    def checks(self) -> list[dict]:
        return [{"id": key, "status": "FAIL" if self.violations[key] else "PASS",
                 "n_violations": self.violations[key], "detail": CHECKS[key]}
                for key in sorted(CHECKS)]


def _schema_issues(row: dict) -> list[str]:
    issues = []
    if "_schema_error" in row:
        issues.append(row["_schema_error"])
    for key in ("sample_id", "source_group", "image", "label", "sha256", "label_sha256", "pixel_sha256"):
        if not isinstance(row.get(key), str) or not row[key]:
            issues.append(f"{key} must be a nonempty string")
    if row.get("split") not in PARTITIONS:
        issues.append("split must be a canonical partition")
    if type(row.get("is_defect")) is not bool:
        issues.append("is_defect must be boolean")
    for key in ("width", "height"):
        if type(row.get(key)) is not int or row[key] <= 0:
            issues.append(f"{key} must be a positive integer")
    pairs = row.get("pair_ids")
    if (not isinstance(pairs, list) or not pairs
            or any(not isinstance(p, str) or not p for p in pairs)):
        issues.append("pair_ids must be a nonempty list of strings")
    elif len(set(pairs)) != len(pairs):
        issues.append("pair_ids contains duplicate aliases")
    if not isinstance(row.get("boxes"), list):
        issues.append("boxes must be a list")
    return issues


def _manifest_boxes(row: dict, names: list[str], source_map: dict, findings: Findings, sid: str | None) -> list[dict]:
    boxes = row.get("boxes")
    if not isinstance(boxes, list):
        findings.fail("box_valid", sid, "Manifest boxes is not a list")
        return []
    valid = []
    width, height = row.get("width"), row.get("height")
    for index, box in enumerate(boxes):
        if not isinstance(box, dict):
            findings.fail("box_valid", sid, f"Manifest box {index} is not an object")
            findings.fail("class_id_valid", sid, f"Manifest box {index} has no class")
            continue
        cls = box.get("class_id")
        class_ok = type(cls) is int and 0 <= cls < len(names)
        if class_ok:
            class_ok = box.get("class_name") == names[cls]
            if "source_class_id" in box and source_map:
                class_ok = class_ok and source_map.get(str(box["source_class_id"])) == cls
        if not class_ok:
            findings.fail("class_id_valid", sid, f"Manifest box {index} has inconsistent class ID/name/source ID")
        xy = box.get("xyxy")
        geometry_ok = (isinstance(xy, list) and len(xy) == 4 and all(finite_number(v) for v in xy)
                       and finite_number(width) and finite_number(height))
        if geometry_ok:
            x1, y1, x2, y2 = xy
            geometry_ok = 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        if not geometry_ok:
            findings.fail("box_valid", sid, f"Manifest box {index} has invalid xyxy/dimensions")
        if class_ok and geometry_ok:
            valid.append({"class_id": cls, "xyxy_normalized":
                          (xy[0] / width, xy[1] / height, xy[2] / width, xy[3] / height)})
    return valid


def _same_boxes(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    # Match a multiset within tolerance: floating point sort keys can swap two
    # boxes sharing x1 even when each annotation agrees to machine precision.
    candidates = [[j for j, y in enumerate(b) if x["class_id"] == y["class_id"]
                   and all(abs(u - v) <= 1e-9 for u, v in zip(x["xyxy_normalized"], y["xyxy_normalized"]))]
                  for x in a]
    matched = {}

    def assign(i, visited):
        for j in candidates[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in matched or assign(matched[j], visited):
                matched[j] = i
                return True
        return False

    return all(assign(i, set()) for i in range(len(a)))


def write_reports(report: dict, profile: str, out_dir: Path) -> None:
    """Stable UTF-8, LF and key ordering; never include runtime timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path(out_dir, "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    artifact_path(out_dir, "data_profile.md").write_text(profile, encoding="utf-8", newline="\n")


def run_audit(dataset_root: Path, out_dir: Path) -> dict:
    """Write data_audit.json + data_profile.md and return the JSON object.

    Raises DatasetInputError for unreadable required metadata. Sample failures
    populate errors and FAIL checks while subsequent samples are still audited.
    Test pixels are decoded/hashed only, never exported or retained for EDA.
    """
    root = Path(dataset_root).resolve()
    out_dir = output_path(root, Path(out_dir))
    data = load_dataset(root)
    findings = Findings()
    names = data.names
    counts = {p: {"images": 0, "good": 0, "defect": 0, "source_groups": 0} for p in PARTITIONS}
    total = {"images": len(data.rows), "good": 0, "defect": 0}
    class_counts = {p: dict.fromkeys(names, 0) for p in PARTITIONS}
    class_images = {p: dict.fromkeys(names, 0) for p in PARTITIONS}
    sets = {p: {key: set() for key in ("source_group", "pair", "image_hash", "pixel_hash")} for p in PARTITIONS}
    members = {p: [] for p in PARTITIONS}
    seen_ids: set[str] = set()
    observations = []
    release = {"n_locked": len(data.release["files"]), "n_verified": 0, "mismatches": []}
    for relative, expected in sorted(data.release["files"].items()):
        try:
            actual = sha256_file(dataset_path(root, relative))
            if actual != expected:
                raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
            release["n_verified"] += 1
        except (OSError, ValueError) as exc:
            release["mismatches"].append(relative)
            findings.fail("release_sha256", None, f"{relative}: {exc}")

    for row in data.rows:
        sid = row.get("sample_id") if isinstance(row.get("sample_id"), str) else None
        partition = row.get("split")
        known_partition = isinstance(partition, str) and partition in PARTITIONS
        issues = _schema_issues(row)
        for issue in issues:
            findings.fail("manifest_schema", sid, issue)
        if sid is not None:
            if sid in seen_ids:
                findings.fail("manifest_unique_ids", sid, "Repeated canonical sample_id")
            seen_ids.add(sid)
        defect = row.get("is_defect")
        if type(defect) is bool:
            total["defect" if defect else "good"] += 1
        if known_partition:
            counts[partition]["images"] += 1
            if type(defect) is bool:
                counts[partition]["defect" if defect else "good"] += 1
            if sid is not None:
                members[partition].append(sid)
            group = row.get("source_group")
            if isinstance(group, str) and group:
                sets[partition]["source_group"].add(group)
            else:
                findings.fail("leak_source_group", sid, "Cannot establish source-group isolation")
            pairs = row.get("pair_ids")
            if isinstance(pairs, list) and pairs and all(isinstance(p, str) and p for p in pairs):
                sets[partition]["pair"].update(pairs)
            else:
                findings.fail("leak_pair", sid, "Cannot establish pair isolation")
        else:
            findings.fail("partition_counts", sid, "Unknown or missing partition")
            for check in ("leak_source_group", "leak_pair", "leak_image_hash", "leak_pixel_hash"):
                findings.fail(check, sid, "Cannot establish isolation for unknown partition")

        paths = {}
        for field in ("image", "label"):
            try:
                path = dataset_path(root, row.get(field))
                if not path.is_file():
                    raise ValueError(f"Missing {field} file: {row.get(field)}")
                paths[field] = path
            except (OSError, ValueError) as exc:
                findings.fail("manifest_paths_exist", sid, str(exc))

        image_size = None
        uniform = None
        for field, hash_key in (("image", "sha256"), ("label", "label_sha256")):
            try:
                if field not in paths:
                    raise ValueError(f"Cannot hash unavailable {field}")
                digest = sha256_file(paths[field])
                if field == "image" and known_partition:
                    sets[partition]["image_hash"].add(digest)
                if digest != row.get(hash_key):
                    findings.fail(f"{field}_sha256", sid, f"{field} SHA-256 differs from manifest")
            except (OSError, ValueError) as exc:
                findings.fail(f"{field}_sha256", sid, str(exc))
                if field == "image":
                    findings.fail("leak_image_hash", sid, "Cannot compute image hash for isolation")
        try:
            if "image" not in paths:
                raise ValueError("Cannot decode unavailable image")
            with Image.open(paths["image"]) as image:
                image.load()
                image_size = image.size
                if image.size != (640, 640):
                    findings.fail("image_size_640", sid, f"Decoded size is {image.width} x {image.height}")
                if image.size != (row.get("width"), row.get("height")) or image.mode != "RGB":
                    findings.fail("image_metadata", sid, "Decoded dimensions/mode disagree with canonical metadata")
                rgb = image.convert("RGB")
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                if digest != row.get("pixel_sha256"):
                    findings.fail("pixel_sha256", sid, "RGB pixel SHA-256 differs from manifest")
                if known_partition:
                    sets[partition]["pixel_hash"].add(digest)
                if partition in ("train", "calibration"):
                    uniform = all(lo == hi for lo, hi in rgb.getextrema())
                rgb.close()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            for check in ("image_decode", "image_size_640", "image_metadata", "pixel_sha256", "leak_pixel_hash"):
                findings.fail(check, sid, f"Cannot inspect image: {exc}")

        manifest_boxes = _manifest_boxes(row, names, data.classes.get("source_to_class_id", {}), findings, sid)
        manifest_box_count = len(row["boxes"]) if isinstance(row.get("boxes"), list) else None
        label_boxes = []
        label_readable = False
        try:
            if "label" not in paths:
                raise ValueError("Cannot read unavailable label")
            label = paths["label"].read_text(encoding="utf-8-sig")
            label_readable = True
            if defect is False and (label.strip() or manifest_box_count != 0):
                findings.fail("label_good_empty", sid, "Good has nonempty label or manifest boxes")
            if defect is True and (not label.strip() or not manifest_box_count):
                findings.fail("label_defect_nonempty", sid, "Defect has empty label or manifest boxes")
            label_boxes, label_issues = parse_yolo(label, names)
            for check, message in label_issues:
                findings.fail(check, sid, message)
            if label_issues or manifest_box_count != len(manifest_boxes) or not _same_boxes(label_boxes, manifest_boxes):
                findings.fail("label_manifest_match", sid, "Label rows and manifest boxes differ or contain invalid boxes")
        except (OSError, UnicodeError, ValueError) as exc:
            for check in ("box_valid", "class_id_valid", "label_manifest_match"):
                findings.fail(check, sid, f"Cannot inspect label: {exc}")
            if type(defect) is bool:
                findings.fail("label_defect_nonempty" if defect else "label_good_empty", sid, str(exc))
        if known_partition:
            for box in label_boxes:
                class_counts[partition][names[box["class_id"]]] += 1
            for cls in {box["class_id"] for box in label_boxes}:
                class_images[partition][names[cls]] += 1
            observations.append({"sample_id": sid, "split": partition,
                "source_group": row.get("source_group") if isinstance(row.get("source_group"), str) else "UNKNOWN",
                "is_defect": defect, "boxes": label_boxes, "image_size": image_size,
                "label_readable": label_readable, "uniform": uniform,
                "image": row.get("image", ""), "source_mode": str(row.get("source_mode", "unknown")),
                "normal_evidence": str(row.get("normal_evidence")),
                "transform": json.dumps(row.get("transform"), sort_keys=True)})

    for p in PARTITIONS:
        counts[p]["source_groups"] = len(sets[p]["source_group"])
        for label, actual, expected in (
            ("samples", Counter(members[p]), Counter(data.holdout["samples"][p])),
            ("groups", Counter(sets[p]["source_group"]), Counter(data.holdout["groups"][p])),
            ("pair_ids", Counter(sets[p]["pair"]), Counter(data.holdout["pair_ids"][p])),
        ):
            if actual != expected:
                findings.fail("partition_counts", None,
                    f"{p}.{label}: missing={sorted((expected - actual).elements())}; "
                    f"unexpected={sorted((actual - expected).elements())}")
        if counts[p]["images"] != len(data.holdout["samples"][p]):
            findings.fail("partition_counts", None, f"{p}: image count differs from holdout sample list")
        if len(sets[p]["pair"]) != data.holdout["actual_pair_counts"][p]:
            findings.fail("partition_counts", None, f"{p}: pair count differs from actual_pair_counts")

    leakage_pairs = []
    for a, b in combinations(PARTITIONS, 2):
        pair = {"a": a, "b": b}
        for key in ("source_group", "pair", "image_hash", "pixel_hash"):
            common = sets[a][key] & sets[b][key]
            pair[key] = len(common)
            if common:
                # One violation per distinct common value in this partition pair.
                for value in sorted(common):
                    findings.fail(f"leak_{key}", None, f"{a}/{b}: shared {key} {value}")
        leakage_pairs.append(pair)
    checks = findings.checks()
    report = {"schema_version": 1, "dataset_root": str(root),
        "overall": "FAIL" if findings.errors else "PASS",
        "counts": {"total": total, "by_partition": counts},
        "class_box_counts": class_counts, "class_image_counts": class_images,
        "checks": checks, "leakage": {"pairs": leakage_pairs}, "release": release,
        "errors": sorted(findings.errors, key=lambda e: (e["sample_id"] or "", e["stage"], e["message"]))}
    # Import here keeps metadata and audit APIs usable without circular imports.
    from .eda import build_profile
    profile, eda_summary, warnings = build_profile(report, observations, names, data.protocol)
    report["eda"] = eda_summary
    report["checks"] = sorted(checks + warnings, key=lambda c: c["id"])
    write_reports(report, profile, out_dir)
    return report
