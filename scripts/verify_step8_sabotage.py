"""Step 8 Adversarial / Sabotage Audit Suite (20 cases)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from pcb_lab.inference.sources import (
    BaseImageSource,
    FileImageSource,
    UploadImageSource,
    URLImageSource,
    decode_image,
    validate_pixels,
)
from pcb_lab.inference.registry import ModelRegistry, ModelAsset
from pcb_lab.inference.service import InferenceService, load_model
from pcb_lab.inference.reports import load_benchmark


def main():
    cases = []
    print("[Sabotage] Running 20 adversarial cases on Step 8 UI & InferenceService...")

    def run_case(case_name, fn, expected_exc):
        try:
            fn()
            cases.append({
                "case": case_name,
                "expected": (
                    [e.__name__ for e in expected_exc]
                    if isinstance(expected_exc, tuple)
                    else expected_exc.__name__
                ),
                "observed": "NO_EXCEPTION_RAISED",
                "passed": False,
            })
        except Exception as exc:
            passed = isinstance(exc, expected_exc)
            cases.append({
                "case": case_name,
                "expected": (
                    [e.__name__ for e in expected_exc]
                    if isinstance(expected_exc, tuple)
                    else expected_exc.__name__
                ),
                "observed": type(exc).__name__,
                "message": str(exc),
                "passed": passed,
            })

    # Case 1: validate_pixels rejects 1D array
    def case1():
        validate_pixels(np.zeros((100,), dtype=np.uint8))
    run_case("S01: validate_pixels rejects 1D array", case1, ValueError)

    # Case 2: validate_pixels rejects non-uint8 array
    def case2():
        validate_pixels(np.zeros((640, 640, 3), dtype=np.float32))
    run_case("S02: validate_pixels rejects float32 array", case2, ValueError)

    # Case 3: validate_pixels rejects 5 channels
    def case3():
        validate_pixels(np.zeros((640, 640, 5), dtype=np.uint8))
    run_case("S03: validate_pixels rejects 5-channel array", case3, ValueError)

    # Case 4: validate_pixels rejects empty array
    def case4():
        validate_pixels(np.zeros((0, 0, 3), dtype=np.uint8))
    run_case("S04: validate_pixels rejects empty array", case4, ValueError)

    # Case 5: validate_pixels rejects NaN float array
    def case5():
        validate_pixels(np.full((640, 640, 3), float("nan")))
    run_case("S05: validate_pixels rejects NaN float array", case5, ValueError)

    # Case 6: validate_pixels rejects 5D array
    def case6():
        validate_pixels(np.zeros((1, 1, 640, 640, 3), dtype=np.uint8))
    run_case("S06: validate_pixels rejects 5D array", case6, ValueError)

    # Case 7: validate_pixels rejects oversized dimension (> 4096)
    def case7():
        validate_pixels(np.zeros((5000, 100, 3), dtype=np.uint8))
    run_case("S07: validate_pixels rejects dimension > 4096 px", case7, ValueError)

    # Case 8: validate_pixels rejects total pixels > 4 million
    def case8():
        validate_pixels(np.zeros((2500, 2500, 3), dtype=np.uint8))
    run_case("S08: validate_pixels rejects total pixels > 4_000_000", case8, ValueError)

    # Case 9: ModelRegistry rejects hallucinatory model 'YOLO26'
    def case9():
        reg = ModelRegistry()
        reg.get("YOLO26")
    run_case("S09: ModelRegistry rejects hallucinatory model 'YOLO26'", case9, ValueError)

    # Case 10: ModelRegistry rejects unknown model ID
    def case10():
        reg = ModelRegistry()
        reg.get("UNKNOWN_MODEL_99")
    run_case("S10: ModelRegistry rejects unknown model ID", case10, ValueError)

    # Case 11: load_model rejects checkpoint with mismatched SHA-256
    def case11():
        asset = ModelAsset(
            model_id="B01",
            checkpoint=ROOT / "artifacts/yolo/B01_yolo11n/seed42/best.pt",
            sha256="0" * 64,
        )
        load_model(asset)
    run_case("S11: load_model rejects checkpoint with mismatched SHA-256", case11, ValueError)

    # Case 12: load_model rejects non-existent checkpoint path
    def case12():
        asset = ModelAsset(
            model_id="B01",
            checkpoint=ROOT / "artifacts/non_existent.pt",
            sha256="0" * 64,
        )
        load_model(asset)
    run_case("S12: load_model rejects non-existent checkpoint path", case12, (FileNotFoundError, ValueError))

    # Case 13: FileImageSource rejects non-existent image
    def case13():
        src = FileImageSource("non_existent_image.png")
        src.get_frame()
    run_case("S13: FileImageSource rejects missing file", case13, (FileNotFoundError, PermissionError))

    # Case 14: decode_image rejects invalid image bytes
    def case14():
        decode_image(b"NOT_A_VALID_IMAGE_DATA")
    run_case("S14: decode_image rejects invalid byte stream", case14, (ValueError, OSError))

    # Case 15: decode_image rejects empty byte stream
    def case15():
        decode_image(b"")
    run_case("S15: decode_image rejects empty bytes", case15, ValueError)

    # Case 16: InferenceService catches failing source without raising
    def case16():
        svc = InferenceService()
        class FailingSource(BaseImageSource):
            def get_frame(self):
                raise RuntimeError("Simulated sensor failure")
        res = svc.inspect(FailingSource(), model_id="B01")
        assert res.decision == "ERROR"
        assert "Simulated sensor failure" in res.error_reason
    try:
        case16()
        cases.append({"case": "S16: InferenceService handles sensor failure gracefully", "expected": "Handled", "observed": "Handled", "passed": True})
    except Exception as exc:
        cases.append({"case": "S16: InferenceService handles sensor failure gracefully", "expected": "Handled", "observed": str(exc), "passed": False})

    # Case 17: load_benchmark rejects missing repo_root / artifacts
    def case17():
        load_benchmark(repo_root=Path("non_existent_repo_dir"))
    run_case("S17: load_benchmark rejects missing artifacts", case17, (FileNotFoundError, OSError))

    # Case 18: load_benchmark rejects tampered CSV with missing columns
    def case18():
        with tempfile.TemporaryDirectory() as td:
            fake_repo = Path(td)
            art = fake_repo / "artifacts/test"
            art.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(ROOT / "artifacts/test/frozen_evaluation.json", art / "frozen_evaluation.json")
            (art / "benchmark_summary.csv").write_text("model_id,wrong_column\nB01,1.0\n", encoding="utf-8")
            (art / "benchmark_per_group.csv").write_text("model_id\nB01\n", encoding="utf-8")
            (art / "benchmark_per_class.csv").write_text("model_id\nB01\n", encoding="utf-8")
            load_benchmark(repo_root=fake_repo)
    run_case("S18: load_benchmark rejects missing mandatory columns", case18, ValueError)

    # Case 19: ModelRegistry gallery blocks test split when report_mode=False
    def case19():
        reg = ModelRegistry()
        reg.gallery("test", report_mode=False)
    run_case("S19: ModelRegistry.gallery blocks test split when report_mode=False", case19, PermissionError)

    # Case 20: URLImageSource rejects non-HTTP URL
    def case20():
        src = URLImageSource("ftp://malicious.com/image.png")
        src.get_frame()
    run_case("S20: URLImageSource rejects non-HTTP(S) protocol", case20, ValueError)

    passed_count = sum(1 for c in cases if c["passed"])
    log_output = {
        "step": 8,
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "failed_cases": len(cases) - passed_count,
        "cases": cases,
    }

    log_path = ROOT / "docs/handoff/step8-sabotage-log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_output, f, indent=2)

    print(f"[Sabotage] {passed_count}/{len(cases)} cases passed successfully!")
    print(f"[Sabotage] Log written to {log_path}")
    return 0 if passed_count == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
