#!/usr/bin/env python3
"""Independent Stdlib Oracle Verification for Step 7 Test Evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

TOLERANCE = 1e-6


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("[Oracle] Starting independent stdlib verification...")
    repo_root = Path.cwd()

    # 1. Verify frozen_evaluation.json
    frozen_path = repo_root / "artifacts/test/frozen_evaluation.json"
    assert frozen_path.is_file(), f"frozen_evaluation.json missing at {frozen_path}"
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)

    assert frozen_cfg["schema_version"] == 1
    assert frozen_cfg["status"] == "FROZEN"
    assert frozen_cfg["test_access_authorized"] is True
    assert frozen_cfg["sample_count"] == 440

    # Verify checkpoint hashes
    for m_id, ckpt_info in frozen_cfg["checkpoints"].items():
        ckpt_path = repo_root / ckpt_info["path"]
        assert ckpt_path.is_file(), f"Checkpoint {ckpt_path} missing"
        actual_sha = sha256_file(ckpt_path)
        assert actual_sha == ckpt_info["sha256"], f"SHA256 mismatch for {m_id}: {actual_sha} vs {ckpt_info['sha256']}"
    print("[Oracle] Checkpoints and frozen config verified.")

    # 2. Verify predictions_test.jsonl
    pred_path = repo_root / "artifacts/test/predictions_test.jsonl"
    assert pred_path.is_file(), f"predictions_test.jsonl missing at {pred_path}"

    records_by_model = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            m_id = row["model_id"]
            records_by_model.setdefault(m_id, []).append(row)

    expected_models = ["B01", "B02", "B03", "H00", "H01", "H02", "H01_sec"]
    assert set(records_by_model.keys()) == set(expected_models), f"Unexpected models in predictions: {set(records_by_model.keys())}"

    # For each model, verify 440 unique samples (220 good, 220 defect)
    for m_id, rows in records_by_model.items():
        assert len(rows) == 440, f"Model {m_id} has {len(rows)} predictions, expected 440"
        sids = [r["sample_id"] for r in rows]
        assert len(set(sids)) == 440, f"Duplicate sample IDs for model {m_id}"
        goods = [r for r in rows if not r["is_defect"]]
        defects = [r for r in rows if r["is_defect"]]
        assert len(goods) == 220, f"Model {m_id} has {len(goods)} good samples, expected 220"
        assert len(defects) == 220, f"Model {m_id} has {len(defects)} defect samples, expected 220"

    print("[Oracle] predictions_test.jsonl records structure verified.")

    # 3. Independent metric calculations
    # Load summary CSV
    summary_csv = repo_root / "artifacts/test/benchmark_summary.csv"
    assert summary_csv.is_file(), "benchmark_summary.csv missing"
    with open(summary_csv, "r", encoding="utf-8") as f:
        summary_rows = {r["model_id"]: r for r in csv.DictReader(f)}

    checked_cells = 0
    for m_id, rows in records_by_model.items():
        tp = sum(1 for r in rows if r["is_defect"] and r["decision"] == "DEFECT")
        tn = sum(1 for r in rows if not r["is_defect"] and r["decision"] == "GOOD")
        fp = sum(1 for r in rows if not r["is_defect"] and r["decision"] == "DEFECT")
        fn = sum(1 for r in rows if r["is_defect"] and r["decision"] == "GOOD")

        recall = tp / (tp + fn)
        miss_rate = fn / (tp + fn)
        frr = fp / (fp + tn)
        bal_acc = (recall + (1.0 - frr)) / 2.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        mcc_num = float(tp * tn - fp * fn)
        mcc_den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

        fn_sim = 100.0 * miss_rate
        fp_sim = 9900.0 * frr
        ecpb = (100.0 * fn_sim + 1.0 * fp_sim) / 10000.0

        csv_row = summary_rows[m_id]

        def _check(name, oracle_val, csv_val_str):
            nonlocal checked_cells
            csv_val = float(csv_val_str)
            assert abs(oracle_val - csv_val) < TOLERANCE, f"Mismatch in {m_id} {name}: oracle={oracle_val}, csv={csv_val}"
            checked_cells += 1

        _check("balanced_accuracy", bal_acc, csv_row["balanced_accuracy"])
        _check("defect_recall", recall, csv_row["defect_recall"])
        _check("defect_precision", prec, csv_row["defect_precision"])
        _check("frr", frr, csv_row["frr"])
        _check("miss_rate", miss_rate, csv_row["miss_rate"])
        _check("f1", f1, csv_row["f1"])
        _check("mcc", mcc, csv_row["mcc"])
        _check("ecpb", ecpb, csv_row["ecpb"])

    print(f"[Oracle] benchmark_summary.csv passed! Checked {checked_cells} metric cells with 0 mismatch.")

    # 4. Independent per-group calculations
    group_csv = repo_root / "artifacts/test/benchmark_per_group.csv"
    assert group_csv.is_file(), "benchmark_per_group.csv missing"
    with open(group_csv, "r", encoding="utf-8") as f:
        group_rows = list(csv.DictReader(f))

    group_map = {(r["model_id"], r["source_group"]): r for r in group_rows}

    group_cells_checked = 0
    for m_id, rows in records_by_model.items():
        grp_metrics = {}
        for grp in ("group44000", "group92000"):
            g_rows = [r for r in rows if r["source_group"] == grp]
            expected_total = 200 if grp == "group44000" else 240
            assert len(g_rows) == expected_total, f"Expected {expected_total} for {grp}, got {len(g_rows)}"

            tp = sum(1 for r in g_rows if r["is_defect"] and r["decision"] == "DEFECT")
            tn = sum(1 for r in g_rows if not r["is_defect"] and r["decision"] == "GOOD")
            fp = sum(1 for r in g_rows if not r["is_defect"] and r["decision"] == "DEFECT")
            fn = sum(1 for r in g_rows if r["is_defect"] and r["decision"] == "GOOD")

            rec = tp / (tp + fn)
            frr = fp / (fp + tn)
            bal = (rec + (1.0 - frr)) / 2.0
            grp_metrics[grp] = {"recall": rec, "frr": frr, "bal_acc": bal}

        worst_rec = min(grp_metrics["group44000"]["recall"], grp_metrics["group92000"]["recall"])
        worst_frr = max(grp_metrics["group44000"]["frr"], grp_metrics["group92000"]["frr"])

        for grp in ("group44000", "group92000"):
            csv_entry = group_map[(m_id, grp)]
            g_rec = grp_metrics[grp]["recall"]
            g_frr = grp_metrics[grp]["frr"]
            g_bal = grp_metrics[grp]["bal_acc"]

            assert abs(g_rec - float(csv_entry["defect_recall"])) < TOLERANCE
            assert abs(g_frr - float(csv_entry["frr"])) < TOLERANCE
            assert abs(g_bal - float(csv_entry["balanced_accuracy"])) < TOLERANCE
            assert abs(worst_rec - float(csv_entry["worst_group_recall"])) < TOLERANCE
            assert abs(worst_frr - float(csv_entry["worst_group_frr"])) < TOLERANCE
            group_cells_checked += 5

    print(f"[Oracle] benchmark_per_group.csv passed! Checked {group_cells_checked} cells with 0 mismatch.")
    print(f"[Oracle] SUCCESS: All {checked_cells + group_cells_checked} oracle checks passed perfectly!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
