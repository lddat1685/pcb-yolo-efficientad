#!/usr/bin/env python
"""Package the Colab training bundle: a single zip with everything needed to run the
notebook on Google Colab, excluding .venv/, runs/, .cache/, datasets, and large .pt/zip files.

Usage:
    python scripts/package_colab.py [--out exports/colab_bundle.zip]

The bundle contains: src/, configs/, scripts/, notebooks/train_yolo_colab.ipynb,
requirements/step3.txt, pyproject.toml (and any other tracked repo source). Dataset and
checkpoints are NOT included — they are fetched at runtime (Cell 2 mounts Drive, Cell 4
downloads pretrained).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directory/pattern globs to exclude from the bundle entirely.
EXCLUDE_DIRS = {".venv", "venv", "runs", ".cache", ".git", "__pycache__",
                "artifacts", "data_refs", "exports", "scratch", "node_modules"}
# File suffixes/patterns to exclude.
EXCLUDE_SUFFIXES = {".pt", ".zip", ".pth", ".onnx", ".engine", ".bin", ".ckpt"}
EXCLUDE_NAMES = {"yolo_step3_artifacts.zip", "bundle_provenance.json"}

INCLUDE_EXACT = {
    "requirements/step1.txt",
    "requirements/step2.txt",
    "requirements/step3.txt",
    "requirements/step4.txt",
    "pyproject.toml",
    "notebooks/train_yolo_colab.ipynb",
    "notebooks/train_efficientad_colab.ipynb",
    "notebooks/train_efficientad_kaggle.ipynb",
}


def _should_include(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    rel_str = str(rel).replace("\\", "/")
    if rel_str in INCLUDE_EXACT:
        return True
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return False
    if p.name in EXCLUDE_NAMES:
        return False
    if p.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    # Never bundle dataset roots or large media.
    if "DatasetVer4" in parts or "DatasetVer" in parts:
        return False
    return True


def collect(root: Path) -> list[Path]:
    out = []
    # Always-include exact files first (create dirs if missing).
    for rel in sorted(INCLUDE_EXACT):
        fp = root / rel
        if not fp.exists():
            print(f"WARN missing expected file: {rel}")
            continue
        out.append(fp)
    # Walk the repo for everything else.
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.as_posix() in INCLUDE_EXACT:
            continue
        if _should_include(p, root):
            out.append(p)
    # De-dupe preserving order.
    seen = set()
    unique = []
    for p in out:
        key = p.relative_to(root).as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description="Package the Colab training bundle.")
    ap.add_argument("--out", default="exports/PCB_Lab_impl4.zip", help="Output zip path")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    files = collect(ROOT)
    git = shutil.which("git")
    if not git and os.environ.get("ProgramFiles"):
        candidate = Path(os.environ["ProgramFiles"]) / "Git/cmd/git.exe"
        if candidate.is_file():
            git = str(candidate)

    commit = "unknown"
    dirty = False
    if git:
        try:
            commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output([git, "status", "--porcelain"], cwd=ROOT, text=True).strip())
        except Exception:
            pass

    provenance = {
        "git_commit": commit,
        "git_dirty": dirty,
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())
        zf.writestr("bundle_provenance.json", json.dumps(provenance, indent=2) + "\n")

    total = sum(f.stat().st_size for f in files)
    print(f"Bundle written: {out}")
    print(f"  files: {len(files)}  uncompressed: {total/1024:.1f} KB")
    print(f"  size : {out.stat().st_size/1024:.1f} KB")
    print("  excludes: .venv/, runs/, .cache/, artifacts/, *.pt, *.zip, datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
