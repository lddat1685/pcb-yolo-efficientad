"""Step 10 Spec-First Test Suite: Packaging & Documentation Integrity.

Written from spec BEFORE reading step10/impl code or step10-impl.md.
Red in Phase A is normal: required docs/scripts do not exist yet on main.

Contract under test (from verifier task):
- README.md, docs/MODEL_CARD.md, docs/DATASET_CARD.md, docs/LIMITATIONS.md,
  docs/REPRODUCIBILITY.md must exist and be non-empty.
- LIMITATIONS.md must document: N=2 groups, bare PCB, domain shift,
  AD locked on YOLO positive.
- scripts/verify_reproducibility.py and scripts/export_thesis_assets.py
  must exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    "README.md",
    "docs/MODEL_CARD.md",
    "docs/DATASET_CARD.md",
    "docs/LIMITATIONS.md",
    "docs/REPRODUCIBILITY.md",
]

REQUIRED_SCRIPTS = [
    "scripts/verify_reproducibility.py",
    "scripts/export_thesis_assets.py",
]


@pytest.mark.parametrize("rel", REQUIRED_DOCS)
def test_step10_required_docs_exist_and_nonempty(rel):
    """Moi tai lieu bat buoc phai ton tai va khong rong."""
    p = REPO / rel
    assert p.is_file(), f"Missing required doc: {rel}"
    text = p.read_text(encoding="utf-8-sig")
    assert text.strip(), f"Required doc is empty: {rel}"
    assert len(text.strip().splitlines()) >= 5, f"Required doc too short: {rel}"


@pytest.mark.parametrize("rel", REQUIRED_SCRIPTS)
def test_step10_required_scripts_exist(rel):
    """Moi script bat buoc phai ton tai va co noi dung python."""
    p = REPO / rel
    assert p.is_file(), f"Missing required script: {rel}"
    text = p.read_text(encoding="utf-8-sig")
    assert text.strip(), f"Required script is empty: {rel}"
    assert "def " in text or "import " in text, f"Not a python script: {rel}"


def _limitations_text():
    p = REPO / "docs/LIMITATIONS.md"
    assert p.is_file(), "Missing docs/LIMITATIONS.md"
    return p.read_text(encoding="utf-8-sig")


def test_step10_limitations_mentions_n2_groups():
    """LIMITATIONS phai ghi ro gioi han N=2 test groups."""
    text = _limitations_text()
    assert "N=2" in text or "N = 2" in text, "LIMITATIONS missing N=2 groups"


def test_step10_limitations_mentions_bare_pcb():
    """LIMITATIONS phai ghi ro scope bare PCB (khong linh kien)."""
    low = _limitations_text().lower()
    assert "bare" in low and "pcb" in low, "LIMITATIONS missing bare PCB scope"


def test_step10_limitations_mentions_domain_shift():
    """LIMITATIONS phai ghi ro domain shift / chuyen mien."""
    low = _limitations_text().lower()
    assert "domain shift" in low or "chuyen mien" in low or "chuyển miền" in low, \
        "LIMITATIONS missing domain shift"


def test_step10_limitations_mentions_ad_locked_on_yolo_positive():
    """LIMITATIONS phai ghi ro AD bi khoa khi YOLO duong."""
    text = _limitations_text()
    low = text.lower()
    assert "yolo" in low, "LIMITATIONS missing YOLO reference"
    assert "lock" in low or "khóa" in low or "khoa" in low, \
        "LIMITATIONS missing AD locked statement"
    assert "ad" in low, "LIMITATIONS missing AD reference"


def test_step10_thesis_export_script_has_table_entry():
    """Script xuat thesis assets phai co diem mo xuat bang (latex/markdown)."""
    p = REPO / "scripts/export_thesis_assets.py"
    assert p.is_file(), "Missing scripts/export_thesis_assets.py"
    text = p.read_text(encoding="utf-8-sig").lower()
    assert "latex" in text or "markdown" in text or ".tex" in text or ".md" in text, \
        "Thesis export script has no table export entry point"


def test_step10_reproducibility_script_has_integrity_entry():
    """Script tai lap phai co diem mo kiem tra toan ven (hash/fingerprint)."""
    p = REPO / "scripts/verify_reproducibility.py"
    assert p.is_file(), "Missing scripts/verify_reproducibility.py"
    text = p.read_text(encoding="utf-8-sig").lower()
    assert "sha256" in text or "hash" in text or "fingerprint" in text, \
        "Reproducibility script has no integrity check entry point"
