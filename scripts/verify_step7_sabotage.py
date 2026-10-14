"""Step 7 Adversarial / Sabotage Audit Suite (20 cases)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from pcb_lab.data.manifest import resolve_dataset_root

from pcb_lab.evaluation.test_eval import freeze_evaluation_config, run_test_evaluation
from pcb_lab.models.yolo.extract import extract_predictions as yolo_extract
from pcb_lab.models.efficientad.extract import extract_efficientad_predictions as ad_extract
from pcb_lab.fusion.rules import predict_cascade, predict_or, validate_sample_alignment


def main():
    cases = []
    print("[Sabotage] Running 20 adversarial cases on Step 7 contracts...")

    def run_case(case_name, fn, expected_exc):
        try:
            fn()
            cases.append({
                "case": case_name,
                "expected": expected_exc.__name__,
                "observed": "NO_EXCEPTION_RAISED",
                "passed": False,
            })
        except Exception as exc:
            passed = isinstance(exc, expected_exc)
            cases.append({
                "case": case_name,
                "expected": expected_exc.__name__,
                "observed": type(exc).__name__,
                "message": str(exc),
                "passed": passed,
            })

    # Case 1: Missing frozen configuration
    def case1():
        run_test_evaluation(frozen_config_path=Path("non_existent_config.json"))
    run_case("S01: Missing frozen config file", case1, FileNotFoundError)

    # Case 2: Unauthorized test access (flag=False)
    def case2():
        tmp_cfg = ROOT / ".cache/sabotage_cfg_unauthorized.json"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        with open("artifacts/test/frozen_evaluation.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        d["test_access_authorized"] = False
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(d, f)
        run_test_evaluation(frozen_config_path=tmp_cfg)
    run_case("S02: Unauthorized test access flag", case2, PermissionError)

    # Case 3: Missing checkpoint B01
    def case3():
        tmp_cfg = ROOT / ".cache/sabotage_cfg_missing_b01.json"
        with open("artifacts/test/frozen_evaluation.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        d["checkpoints"]["B01_yolo11n"]["path"] = "non_existent_b01.pt"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(d, f)
        run_test_evaluation(frozen_config_path=tmp_cfg)
    run_case("S03: Missing checkpoint B01", case3, FileNotFoundError)

    # Case 4: Missing checkpoint B02
    def case4():
        tmp_cfg = ROOT / ".cache/sabotage_cfg_missing_b02.json"
        with open("artifacts/test/frozen_evaluation.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        d["checkpoints"]["B02_yolo11s"]["path"] = "non_existent_b02.pt"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(d, f)
        run_test_evaluation(frozen_config_path=tmp_cfg)
    run_case("S04: Missing checkpoint B02", case4, FileNotFoundError)

    # Case 5: Missing checkpoint B03
    def case5():
        tmp_cfg = ROOT / ".cache/sabotage_cfg_missing_b03.json"
        with open("artifacts/test/frozen_evaluation.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        d["checkpoints"]["B03_efficientad_s"]["path"] = "non_existent_b03.pt"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(d, f)
        run_test_evaluation(frozen_config_path=tmp_cfg)
    run_case("S05: Missing checkpoint B03", case5, FileNotFoundError)

    # Case 6: Legacy yolo_extract refuses test partition
    def case6():
        yolo_extract(
            artifact_dir=Path("artifacts/yolo/B01_yolo11n/seed42"),
            dataset_root=resolve_dataset_root(),
            partitions=["test"],
        )
    run_case("S06: Legacy YOLO extractor rejects test", case6, PermissionError)

    # Case 7: Legacy ad_extract refuses test partition
    def case7():
        ad_extract(
            artifact_dir=Path("artifacts/efficientad/B03_efficientad_s/seed42"),
            dataset_root=resolve_dataset_root(),
            partitions=["test"],
        )
    run_case("S07: Legacy EfficientAD extractor rejects test", case7, PermissionError)

    # Case 8: Fusion rules validate_sample_alignment without allow_test refuses test
    def case8():
        validate_sample_alignment([{"sample_id": "s1", "split": "test"}], [{"sample_id": "s1", "split": "test"}])
    run_case("S08: validate_sample_alignment rejects test by default", case8, PermissionError)

    # Case 9: Fusion rules predict_cascade without allow_test refuses test
    def case9():
        predict_cascade({"sample_id": "s1", "split": "test"}, {"sample_id": "s1", "split": "test"}, 0.14, 1.4598)
    run_case("S09: predict_cascade rejects test by default", case9, PermissionError)

    # Case 10: Fusion rules predict_or without allow_test refuses test
    def case10():
        predict_or({"sample_id": "s1", "split": "test"}, {"sample_id": "s1", "split": "test"}, 0.14, 1.4598)
    run_case("S10: predict_or rejects test by default", case10, PermissionError)

    # Case 11: Sample ID mismatch in predict_cascade
    def case11():
        predict_cascade({"sample_id": "s1", "split": "fusion"}, {"sample_id": "s2", "split": "fusion"}, 0.14, 1.4598)
    run_case("S11: Sample ID mismatch in predict_cascade", case11, ValueError)

    # Case 12: Sample ID mismatch in predict_or
    def case12():
        predict_or({"sample_id": "s1", "split": "fusion"}, {"sample_id": "s2", "split": "fusion"}, 0.14, 1.4598)
    run_case("S12: Sample ID mismatch in predict_or", case12, ValueError)

    # Case 13: Duplicate sample_id in validate_sample_alignment
    def case13():
        validate_sample_alignment(
            [{"sample_id": "s1", "split": "fusion"}, {"sample_id": "s1", "split": "fusion"}],
            [{"sample_id": "s1", "split": "fusion"}],
        )
    run_case("S13: Duplicate sample_id in alignment check", case13, ValueError)

    # Case 14: Non-bijective sample alignment
    def case14():
        validate_sample_alignment(
            [{"sample_id": "s1", "split": "fusion"}],
            [{"sample_id": "s2", "split": "fusion"}],
        )
    run_case("S14: Non-bijective sample IDs", case14, ValueError)

    # Case 15: Tampered test sample list (< 440)
    def case15():
        with tempfile.TemporaryDirectory() as td:
            tmp_holdout = Path(td) / "benchmarks/deeppcb/splits/holdout.json"
            tmp_holdout.parent.mkdir(parents=True, exist_ok=True)
            with open(resolve_dataset_root() / "benchmarks/deeppcb/splits/holdout.json", "r") as f:
                d = json.load(f)
            d["samples"]["test"] = d["samples"]["test"][:400]  # Only 400
            with open(tmp_holdout, "w") as f:
                json.dump(d, f)
            freeze_evaluation_config(dataset_root=Path(td))
    run_case("S15: Tampered test sample count (< 440)", case15, ValueError)

    # Case 16: Tampered test sample list (> 440)
    def case16():
        with tempfile.TemporaryDirectory() as td:
            tmp_holdout = Path(td) / "benchmarks/deeppcb/splits/holdout.json"
            tmp_holdout.parent.mkdir(parents=True, exist_ok=True)
            with open(resolve_dataset_root() / "benchmarks/deeppcb/splits/holdout.json", "r") as f:
                d = json.load(f)
            d["samples"]["test"].append("extra_test_sample")  # 441
            with open(tmp_holdout, "w") as f:
                json.dump(d, f)
            freeze_evaluation_config(dataset_root=Path(td))
    run_case("S16: Tampered test sample count (> 440)", case16, ValueError)

    # Case 17: Missing holdout.json during freeze
    def case17():
        with tempfile.TemporaryDirectory() as td:
            freeze_evaluation_config(dataset_root=Path(td))
    run_case("S17: Missing holdout.json during freeze", case17, FileNotFoundError)

    # Case 18: YOLO inference error propagation in predict_cascade
    def case18():
        predict_cascade({"sample_id": "s1", "split": "fusion", "error_reason": "CUDA out of memory"}, {"sample_id": "s1", "split": "fusion"}, 0.14, 1.4598)
    run_case("S18: YOLO inference error propagation", case18, ValueError)

    # Case 19: AD inference error propagation in predict_cascade
    def case19():
        predict_cascade({"sample_id": "s1", "split": "fusion", "yolo_image_score": 0.01}, {"sample_id": "s1", "split": "fusion", "error_reason": "Tensor NaN"}, 0.14, 1.4598)
    run_case("S19: AD inference error propagation", case19, ValueError)

    # Case 20: AD inference error propagation in predict_or
    def case20():
        predict_or({"sample_id": "s1", "split": "fusion", "yolo_image_score": 0.01}, {"sample_id": "s1", "split": "fusion", "error_reason": "Input corrupt"}, 0.14, 1.4598)
    run_case("S20: AD error propagation in predict_or", case20, ValueError)

    passed_count = sum(1 for c in cases if c["passed"])
    log_output = {
        "step": 7,
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "failed_cases": len(cases) - passed_count,
        "cases": cases,
    }

    log_path = ROOT / "docs/handoff/step7-sabotage-log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_output, f, indent=2)

    print(f"[Sabotage] {passed_count}/{len(cases)} cases passed successfully!")
    print(f"[Sabotage] Log written to {log_path}")
    return 0 if passed_count == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
