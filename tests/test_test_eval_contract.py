"""Step 7 Spec-first Contract Tests: Verification of Frozen Evaluation & Test Invariants."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
from pcb_lab.data.manifest import resolve_dataset_root

DATASET = resolve_dataset_root()

HOLD_OUT_PATH = DATASET / "benchmarks/deeppcb/splits/holdout.json"
MANIFEST_PATH = DATASET / "benchmarks/deeppcb/manifests/samples.jsonl"
SHORTLIST_PATH = Path("artifacts/benchmark/shortlist.json")
FROZEN_CONFIG_PATH = Path("artifacts/test/frozen_evaluation.json")


def test_canonical_test_dataset_structure():
    """Verify exact 440 canonical test images (220 good, 220 defect) in 2 source groups."""
    assert HOLD_OUT_PATH.is_file(), f"holdout.json missing at {HOLD_OUT_PATH}"
    with open(HOLD_OUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_ids = data.get("samples", {}).get("test", [])
    assert len(test_ids) == 440, f"Expected exactly 440 test samples, got {len(test_ids)}"
    assert len(set(test_ids)) == 440, "Duplicate sample IDs found in test split"

    good_ids = [sid for sid in test_ids if sid.endswith("_good")]
    defect_ids = [sid for sid in test_ids if sid.endswith("_defect")]
    assert len(good_ids) == 220, f"Expected 220 good samples, got {len(good_ids)}"
    assert len(defect_ids) == 220, f"Expected 220 defect samples, got {len(defect_ids)}"

    # Check source groups from manifest
    assert MANIFEST_PATH.is_file(), f"samples.jsonl missing at {MANIFEST_PATH}"
    test_id_set = set(test_ids)
    group_counts = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in test_id_set:
                grp = row["source_group"]
                is_def = sid.endswith("_defect")
                key = (grp, is_def)
                group_counts[key] = group_counts.get(key, 0) + 1

    assert group_counts == {
        ("group44000", True): 100,
        ("group44000", False): 100,
        ("group92000", True): 120,
        ("group92000", False): 120,
    }, f"Unexpected source group breakdown: {group_counts}"


def test_legacy_extractors_block_test_partition():
    """Verify legacy Step 3/4 extractors strictly reject test partition with PermissionError."""
    from pcb_lab.models.yolo.extract import extract_predictions as yolo_extract
    from pcb_lab.models.efficientad.extract import extract_efficientad_predictions as ad_extract

    with pytest.raises(PermissionError, match="test"):
        yolo_extract(
            artifact_dir=Path("artifacts/yolo/B01_yolo11n/seed42"),
            dataset_root=DATASET,
            partitions=["test"],
        )

    with pytest.raises(PermissionError, match="calibration/fusion"):
        ad_extract(
            artifact_dir=Path("artifacts/efficientad/B03_efficientad_s/seed42"),
            dataset_root=DATASET,
            partitions=["test"],
        )


def test_fusion_rules_block_test_partition_by_default():
    """Verify pcb_lab.fusion.rules blocks test split by default (unless authorized)."""
    from pcb_lab.fusion.rules import predict_cascade, predict_or, validate_sample_alignment

    y_row = {
        "sample_id": "test_sample_01_defect",
        "split": "test",
        "yolo_image_score": 0.5,
        "is_defect": True,
    }
    ad_row = {
        "sample_id": "test_sample_01_defect",
        "split": "test",
        "anomaly_score": 1.5,
        "is_defect": True,
    }

    with pytest.raises(PermissionError):
        validate_sample_alignment([y_row], [ad_row])

    with pytest.raises(PermissionError):
        predict_cascade(y_row, ad_row, tau_yolo=0.14, tau_ad=1.4598)

    with pytest.raises(PermissionError):
        predict_or(y_row, ad_row, tau_yolo=0.14, tau_ad=1.4598)


def test_shortlist_thresholds_match_contract_specification():
    """Ensure locked thresholds match the contract for Step 7."""
    assert SHORTLIST_PATH.is_file(), "Step 6 shortlist missing"
    with open(SHORTLIST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = {m["model_id"]: m for m in data["shortlist"]}
    assert "B01_yolo11n" in models
    assert models["B01_yolo11n"]["tau_yolo"] == 0.14

    assert "B03_efficientad_s" in models
    assert models["B03_efficientad_s"]["tau_ad"] == 1.4598

    assert "H00_yolo11n_or_efficientad_s" in models
    assert models["H00_yolo11n_or_efficientad_s"]["tau_yolo"] == 0.14
    assert models["H00_yolo11n_or_efficientad_s"]["tau_ad"] == 1.4598

    assert "H01_cascade_yolo11n_efficientad_s" in models
    assert models["H01_cascade_yolo11n_efficientad_s"]["tau_yolo"] == 0.14
    assert models["H01_cascade_yolo11n_efficientad_s"]["tau_ad"] == 1.4598

    assert "H02_cascade_yolo11s_efficientad_s" in models
    assert models["H02_cascade_yolo11s_efficientad_s"]["tau_yolo"] == 0.38
    assert models["H02_cascade_yolo11s_efficientad_s"]["tau_ad"] == 1.4598


def test_test_eval_module_signature():
    """Verify pcb_lab.evaluation.test_eval exports the required interface."""
    try:
        import pcb_lab.evaluation.test_eval as te
    except ImportError:
        pytest.fail("pcb_lab.evaluation.test_eval not yet implemented")

    assert hasattr(te, "freeze_evaluation_config"), "freeze_evaluation_config missing"
    assert hasattr(te, "run_test_evaluation"), "run_test_evaluation missing"
