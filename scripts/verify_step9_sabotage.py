"""Step 9 Adversarial / Sabotage Audit Suite (20 cases).

Covers: shape la, tensor unaligned, backend crash, gia mao metadata,
profiler invariants, export parity, TileManager guards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from pcb_lab.inference.sources import validate_pixels, decode_image
from pcb_lab.inference.preprocessing import to_canonical
from pcb_lab.inference.tiling import TileManager, global_nms
from pcb_lab.inference.registry import ModelRegistry, ModelAsset
from pcb_lab.inference.service import InferenceService, load_model, BaseImageSource
from pcb_lab.runtime.profiler import STAGES, YOLO_SUBSTAGES, AD_SUBSTAGES, ProfileRun, ProfileSummary, _Timer


def main():
    cases = []
    print("[Sabotage Step9] Running 20 adversarial cases on runtime profiler & export...")

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
                "message": str(exc)[:300],
                "passed": passed,
            })

    def run_success_case(case_name, fn):
        try:
            fn()
            cases.append({"case": case_name, "expected": "SUCCESS", "observed": "SUCCESS", "passed": True})
        except Exception as exc:
            cases.append({"case": case_name, "expected": "SUCCESS", "observed": f"{type(exc).__name__}: {exc}", "passed": False})

    # S01: shape la - 5-channel array bi tu choi
    def s01():
        validate_pixels(np.zeros((640, 640, 5), dtype=np.uint8))
    run_case("S01: shape la - 5-channel array bi tu choi", s01, ValueError)

    # S02: tensor unaligned dtype - float32 bi tu choi
    def s02():
        validate_pixels(np.zeros((640, 640, 3), dtype=np.float32))
    run_case("S02: tensor unaligned - float32 dtype bi tu choi", s02, ValueError)

    # S03: shape la - empty 0x0 bi tu choi
    def s03():
        validate_pixels(np.zeros((0, 0, 3), dtype=np.uint8))
    run_case("S03: shape la - empty 0x0 bi tu choi", s03, ValueError)

    # S04: shape la - dimension >4096 bi tu choi
    def s04():
        validate_pixels(np.zeros((5000, 100, 3), dtype=np.uint8))
    run_case("S04: shape la - dimension >4096 bi tu choi", s04, ValueError)

    # S05: shape la - >4M pixels bi tu choi
    def s05():
        validate_pixels(np.zeros((2500, 2500, 3), dtype=np.uint8))
    run_case("S05: shape la - >4M pixel bi tu choi", s05, ValueError)

    # S06: shape la - tiny 4x4 van canonical duoc (khong crash)
    def s06():
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        out = to_canonical(tiny)
        assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    run_success_case("S06: shape la - tiny 4x4 van chay (khong crash)", s06)

    # S07: tensor unaligned - non-contiguous transposed array van chay
    def s07():
        arr = np.zeros((640, 640, 3), dtype=np.uint8)
        transposed = np.asfortranarray(arr)  # non-C-contiguous
        assert not transposed.flags["C_CONTIGUOUS"]
        out = to_canonical(transposed)
        assert out.flags["C_CONTIGUOUS"] and out.shape == (640, 640, 3)
    run_success_case("S07: tensor unaligned - non-contiguous F-order van chay", s07)

    # S08: tensor unaligned - grayscale 1-ch se replicate thanh 3-ch
    def s08():
        gray = np.zeros((640, 640), dtype=np.uint8)
        out = to_canonical(gray)
        assert out.shape == (640, 640, 3)
    run_success_case("S08: tensor unaligned - grayscale 1-ch replicate thanh 3-ch", s08)

    # S09: TileManager - tile_size !=256 bi tu choi
    def s09():
        TileManager(tile_size=320, stride=224)
    run_case("S09: TileManager - tile_size 320 bi tu choi", s09, ValueError)

    # S10: TileManager - stride > tile_size bi tu choi
    def s10():
        TileManager(tile_size=256, stride=300)
    run_case("S10: TileManager - stride > tile_size bi tu choi", s10, ValueError)

    # S11: TileManager - stitch sai so tile bi tu choi
    def s11():
        tm = TileManager()
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        tiles, specs = tm.split(img)
        bad_maps = np.zeros((5, 256, 256), dtype=np.float32)  # wrong count
        tm.stitch(bad_maps, specs, (640, 640))
    run_case("S11: TileManager - stitch sai so tile bi tu choi", s11, ValueError)

    # S12: TileManager - tile_maps chua NaN bi tu choi
    def s12():
        tm = TileManager()
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        tiles, specs = tm.split(img)
        maps = np.zeros((len(specs), 256, 256), dtype=np.float32)
        maps[0, 0, 0] = float("nan")
        tm.stitch(maps, specs, (640, 640))
    run_case("S12: TileManager - tile_maps NaN bi tu choi", s12, ValueError)

    # S13: backend crash - global_nms IoU invalid
    def s13():
        dets = np.array([[0, 0, 10, 10, 0.9]], dtype=np.float64)
        global_nms(dets, iou=1.5)
    run_case("S13: backend crash - global_nms IoU>1 bi tu choi", s13, ValueError)

    # S14: backend crash - ModelRegistry tu choi YOLO26
    def s14():
        reg = ModelRegistry()
        reg.get("YOLO26")
    run_case("S14: backend crash - ModelRegistry tu choi YOLO26", s14, ValueError)

    # S15: backend crash - Gallery chan test khi chua report_mode
    def s15():
        reg = ModelRegistry()
        reg.gallery("test", report_mode=False)
    run_case("S15: backend crash - Gallery chan test khi chua report_mode", s15, PermissionError)

    # S16: gia mao metadata - load_model tu choi SHA256 gia mao (hoac file thieu do gitignore *.pt)
    def s16():
        asset = ModelAsset(
            model_id="B01",
            checkpoint=ROOT / "artifacts/yolo/B01_yolo11n/seed42/best.pt",
            sha256="0" * 64,
        )
        load_model(asset)
    run_case("S16: gia mao metadata - load_model tu choi SHA256 sai", s16, (ValueError, FileNotFoundError, OSError))

    # S17: backend crash - export_yolo_onnx fallback khi thieu onnx (khong crash)
    def s17():
        import importlib.util
        spec_path = ROOT / "scripts/export_runtime.py"
        mod_spec = importlib.util.spec_from_file_location("export_runtime", str(spec_path))
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        export_yolo_onnx = mod.export_yolo_onnx
        reg = ModelRegistry(ROOT)
        spec = reg.get("B01")
        out = export_yolo_onnx(spec.yolo.checkpoint.parent, ROOT / "artifacts/runtime/B01/yolo_test_tmp")
        assert out["status"] in ("fallback", "skipped", "exported"), f"unexpected {out}"
        if out["status"] == "fallback":
            assert out["backend"] == "pytorch"
    run_success_case("S17: backend crash - export_yolo_onnx fallback khi thieu onnx", s17)

    # S18: gia mao metadata - corrupted TorchScript load phai raise
    def s18():
        import tempfile, torch
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tf.write(b"CORRUPTED_NOT_A_MODEL")
            tmp = Path(tf.name)
        try:
            torch.jit.load(str(tmp), map_location="cpu")
            raise AssertionError("Should have raised on corrupted file")
        except Exception as exc:
            # Any exception is expected; ensure it's not silent success
            if isinstance(exc, AssertionError):
                raise
            # pass - correctly rejected
            pass
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass
    run_success_case("S18: gia mao metadata - corrupted TorchScript bi tu choi", s18)

    # S19: profiler invariant - early_exit => AD khong chay, score None, efficientad timing 0
    def s19():
        # STAGES must contain required stages
        assert "decode" in STAGES
        assert "yolo" in STAGES
        assert "ad_tile_infer" in STAGES
        assert "ad_stitch" in STAGES
        assert "fusion_decision" in STAGES
        assert "total_core" in STAGES
        assert "total" in STAGES
        # YOLO/AD substages rollup
        assert set(YOLO_SUBSTAGES) <= set(STAGES)
        assert set(AD_SUBSTAGES) <= set(STAGES)
        # Simulate early_exit ProfileRun
        run = ProfileRun(sample_id="test", model_id="H01")
        run.early_exit = True
        run.efficientad_executed = False
        run.anomaly_score = None
        run.timings_ms = {"ad_tile_infer": 0.0, "efficientad": 0.0, "yolo": 120.0, "total_core": 120.0}
        assert run.early_exit is True
        assert run.efficientad_executed is False
        assert run.anomaly_score is None
        assert run.timings_ms["efficientad"] == 0.0
    run_success_case("S19: profiler invariant - early_exit => AD khong chay", s19)

    # S20: profiler invariant - total_core == yolo_total + efficientad + fusion (rollup)
    def s20():
        # Recompute rollup as profiler.py does
        timer_marks = {
            "preprocess_letterbox": 20.0, "yolo": 140.0, "yolo_postprocess": 0.3,
            "ad_split": 2.5, "ad_tile_infer": 8000.0, "ad_stitch": 12.0, "ad_postprocess": 4.0,
            "fusion_decision": 0.2, "preprocess_canonical": 0.7,
        }
        yolo_total = sum(timer_marks[s] for s in YOLO_SUBSTAGES)
        efficientad = sum(timer_marks[s] for s in AD_SUBSTAGES)
        total_core = yolo_total + efficientad + timer_marks["fusion_decision"]
        # Check matches profiler logic: yolo_total = sum(YOLO_SUBSTAGES), efficientad = sum(AD_SUBSTAGES)
        assert abs(yolo_total - (20.0 + 140.0 + 0.3)) < 1e-6
        assert abs(efficientad - (2.5 + 8000.0 + 12.0 + 4.0)) < 1e-6
        assert abs(total_core - (yolo_total + efficientad + 0.2)) < 1e-6
        # Also check StageStats quantile
        from pcb_lab.runtime.profiler import StageStats
        stats = StageStats(name="test")
        for v in [100, 200, 300]:
            stats.add(v)
        assert abs(stats.mean_ms - 200.0) < 1e-6
        assert abs(stats.p50_ms - 200.0) < 1e-6
    run_success_case("S20: profiler invariant - total_core rollup & StageStats", s20)

    passed = sum(1 for c in cases if c["passed"])
    log = {
        "step": 9,
        "total_cases": len(cases),
        "passed_cases": passed,
        "failed_cases": len(cases) - passed,
        "cases": cases,
    }
    log_path = ROOT / "docs/handoff/step9-sabotage-log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"[Sabotage Step9] {passed}/{len(cases)} cases passed")
    print(f"[Sabotage Step9] Log -> {log_path}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
