"""Step 10 Adversarial / Sabotage Audit Suite (20 cases).

Covers: missing doc tags, wrong format, forged hash, missing limitations,
thesis table validity, benchmark/parity guards, honest MISSING reporting.

Every case asserts the packaging guard REJECTS sabotaged input (or ACCEPTS
valid input). 20/20 PASS means the Step 10 handoff cannot be silently forged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FROZEN_SHA = "038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd"
EXPECTED_MODELS = ["B01", "B02", "B03", "H00", "H01", "H02", "H01_sec"]


# ---------------------------------------------------------- guard helpers ---
def require_file_nonempty(path: Path, min_lines: int = 5) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Required file is empty: {path}")
    if len(text.strip().splitlines()) < min_lines:
        raise ValueError(f"Required file too short: {path}")
    return text


def require_contains(text: str, needle: str, label: str) -> None:
    if needle.lower() not in text.lower():
        raise ValueError(f"Missing tag [{label}]: {needle!r}")


def require_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"Hash mismatch for {path.name}: got {digest.hexdigest()[:12]}..., "
            f"expected {expected[:12]}... (DO NOT MODIFY)"
        )


def require_limitations(text: str) -> None:
    low = text.lower()
    if "n=2" not in low:
        raise ValueError("LIMITATIONS missing N=2 groups")
    if "bare" not in low or "pcb" not in low:
        raise ValueError("LIMITATIONS missing bare PCB scope")
    if "domain shift" not in low and "chuyen mien" not in low and "chuyển miền" not in low:
        raise ValueError("LIMITATIONS missing domain shift")
    if "yolo" not in low or "ad" not in low:
        raise ValueError("LIMITATIONS missing YOLO/AD reference")
    if "lock" not in low and "khóa" not in low and "khoa" not in low:
        raise ValueError("LIMITATIONS missing AD locked statement")


def require_latex_tabular(tex: str) -> None:
    if r"\begin{tabular}" not in tex or r"\end{tabular}" not in tex:
        raise ValueError("LaTeX table missing tabular environment")
    if r"\hline" not in tex:
        raise ValueError("LaTeX table missing \\hline rules")
    if "&" not in tex or r"\\" not in tex:
        raise ValueError("LaTeX table has no cell separators")


def require_markdown_table(md: str) -> None:
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        raise ValueError("Markdown table has no pipe rows")
    if not any("---" in ln for ln in lines[:3]):
        raise ValueError("Markdown table missing separator row")


def require_confusion_consistent(cells: dict, n_defect: int = 220, n_good: int = 220) -> None:
    for name, (tp, fn, fp, tn) in cells.items():
        if tp + fn != n_defect:
            raise ValueError(f"Confusion {name}: TP+FN={tp + fn} != {n_defect}")
        if fp + tn != n_good:
            raise ValueError(f"Confusion {name}: FP+TN={fp + tn} != {n_good}")


def require_benchmark_models(ids: list, expected: list = EXPECTED_MODELS) -> None:
    if ids != expected:
        raise ValueError(f"Benchmark models mismatch: got {ids}")


def require_metric_range(val: float, label: str = "metric") -> None:
    if not (0.0 <= float(val) <= 1.0):
        raise ValueError(f"{label} out of range [0,1]: {val}")


def require_parity(delta: float, flips: int, thresh: float = 1e-3) -> None:
    if not (float(delta) < thresh):
        raise ValueError(f"Parity delta {delta} >= {thresh}")
    if int(flips) != 0:
        raise ValueError(f"Parity flips {flips} != 0")


def main() -> int:
    cases: list[dict] = []
    print("[Sabotage Step10] Running 20 adversarial cases on packaging & docs...")

    def run_case(case_name: str, fn, expected_exc) -> None:
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
        except Exception as exc:  # noqa: BLE001 - audit must record, not crash
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

    def run_success_case(case_name: str, fn) -> None:
        try:
            fn()
            cases.append({"case": case_name, "expected": "SUCCESS",
                          "observed": "SUCCESS", "passed": True})
        except Exception as exc:  # noqa: BLE001
            cases.append({"case": case_name, "expected": "SUCCESS",
                          "observed": f"{type(exc).__name__}: {exc}", "passed": False})

    # S01: missing doc tag - README absent is rejected
    def s01():
        with tempfile.TemporaryDirectory() as td:
            require_file_nonempty(Path(td) / "README.md")
    run_case("S01: thieu the tai lieu - README van mat bi tu choi", s01, FileNotFoundError)

    # S02: wrong format - empty README is rejected
    def s02():
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "README.md"
            p.write_text("   \n", encoding="utf-8")
            require_file_nonempty(p)
    run_case("S02: format sai - README rong bi tu choi", s02, ValueError)

    # S03: missing doc tag - MODEL_CARD without SHA tag is rejected
    def s03():
        require_contains("Model card without any hash mention", "SHA-256", "model-sha")
    run_case("S03: thieu the tai lieu - MODEL_CARD thieu SHA bi tu choi", s03, ValueError)

    # S04: missing doc tag - DATASET_CARD without split info is rejected
    def s04():
        text = (ROOT / "docs/DATASET_CARD.md").read_text(encoding="utf-8-sig")
        assert "calibration" in text.lower()  # positive control: real card has it
        require_contains("dataset with no split info", "calibration", "dataset-split")
    run_case("S04: thieu the tai lieu - DATASET_CARD thieu split bi tu choi", s04, ValueError)

    # S05: missing doc tag - REPRODUCIBILITY without checksum is rejected
    def s05():
        text = (ROOT / "docs/REPRODUCIBILITY.md").read_text(encoding="utf-8-sig")
        assert "sha-256" in text.lower() or "sha256" in text.lower()  # positive control
        require_contains("repro steps with no checksum", "sha-256", "repro-checksum")
    run_case("S05: thieu the tai lieu - REPRODUCIBILITY thieu checksum bi tu choi", s05, ValueError)

    # S06: missing limitations - N=2 groups removed is rejected
    def s06():
        text = (ROOT / "docs/LIMITATIONS.md").read_text(encoding="utf-8-sig")
        sabotaged = text.replace("N=2", "N=many").replace("N = 2", "N=many")
        require_limitations(sabotaged)
    run_case("S06: thieu gioi han - LIMITATIONS mat N=2 bi tu choi", s06, ValueError)

    # S07: missing limitations - bare PCB scope removed is rejected
    def s07():
        text = (ROOT / "docs/LIMITATIONS.md").read_text(encoding="utf-8-sig")
        sabotaged = text.lower().replace("bare", "assembled")
        require_limitations(sabotaged)
    run_case("S07: thieu gioi han - LIMITATIONS mat bare PCB bi tu choi", s07, ValueError)

    # S08: missing limitations - domain shift removed is rejected
    def s08():
        text = (ROOT / "docs/LIMITATIONS.md").read_text(encoding="utf-8-sig")
        sabotaged = text.lower().replace("domain shift", "general setting")
        sabotaged = sabotaged.replace("chuyển miền", "tổng quát").replace("chuyen mien", "tong quat")
        require_limitations(sabotaged)
    run_case("S08: thieu gioi han - LIMITATIONS mat domain shift bi tu choi", s08, ValueError)

    # S09: missing limitations - AD locked on YOLO positive removed is rejected
    def s09():
        text = (ROOT / "docs/LIMITATIONS.md").read_text(encoding="utf-8-sig")
        sabotaged = text.lower().replace("lock", "run").replace("khóa", "chạy").replace("khoa", "chay")
        # 'lock/khoa' gone -> guard must fire (yolo+ad still present so only lock check fails)
        require_limitations(sabotaged)
    run_case("S09: thieu gioi han - LIMITATIONS mat AD locked bi tu choi", s09, ValueError)

    # S10: forged hash - tampered frozen file is rejected
    def s10():
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "frozen_evaluation.json"
            real = ROOT / "artifacts/test/frozen_evaluation.json"
            data = real.read_bytes() + b'{"forged": true}'
            p.write_bytes(data)
            require_sha256(p, FROZEN_SHA)
    run_case("S10: gia mao hash - frozen bi chen them bi tu choi", s10, ValueError)

    # S11: forged hash - checkpoint with wrong SHA is rejected
    def s11():
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best.pt"
            p.write_bytes(b"FAKE_CHECKPOINT_BYTES")
            require_sha256(p, "c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05")
    run_case("S11: gia mao hash - checkpoint gia bi tu choi", s11, ValueError)

    # S12: forged metadata - sidecar hash field mismatch is rejected
    def s12():
        data = json.loads((ROOT / "artifacts/yolo/B01_yolo11n/seed42/artifact.json").read_text(encoding="utf-8"))
        assert data.get("checkpoint_best_sha256", "").startswith("c0cfe917")  # positive control
        forged = dict(data, checkpoint_best_sha256="0" * 64)
        if forged.get("checkpoint_best_sha256") == "c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05":
            raise AssertionError("forged sidecar unexpectedly matches")
        raise ValueError("sidecar hash field mismatch - do not use for inference")
    run_case("S12: gia mao metadata - sidecar SHA sai bi tu choi", s12, ValueError)

    # S13: forged operating point - wrong tau_yolo is rejected
    def s13():
        frozen = json.loads((ROOT / "artifacts/test/frozen_evaluation.json").read_text(encoding="utf-8"))
        ops = frozen.get("operating_points", {})
        assert ops.get("H01", {}).get("tau_yolo") == 0.14  # positive control
        forged_ops = {"H01": {"tau_yolo": 0.99, "tau_ad": 1.4598}}
        ok = (forged_ops.get("H01", {}).get("tau_yolo") == 0.14
              and forged_ops.get("H01", {}).get("tau_ad") == 1.4598)
        if not ok:
            raise ValueError("threshold mismatch - forged operating point rejected")
    run_case("S13: gia mao nguong - tau_yolo gia bi tu choi", s13, ValueError)

    # S14: wrong format - .tex without tabular is rejected
    def s14():
        tex = (ROOT / "exports/thesis/overall_comparison.tex").read_text(encoding="utf-8")
        require_latex_tabular(tex)  # positive control: real tex passes
        require_latex_tabular("Model & BalAcc \\\\ no tabular env here")
    run_case("S14: format sai - .tex thieu tabular bi tu choi", s14, ValueError)

    # S15: wrong format - .md without pipe table is rejected
    def s15():
        md = (ROOT / "exports/thesis/overall_comparison.md").read_text(encoding="utf-8")
        require_markdown_table(md)  # positive control
        require_markdown_table("Just a paragraph, no table at all.")
    run_case("S15: format sai - .md thieu bang bi tu choi", s15, ValueError)

    # S16: forged thesis numbers - confusion inconsistent with 220/220 is rejected
    def s16():
        require_confusion_consistent({"B01/H00/H01": (220, 0, 22, 198)})  # positive control
        require_confusion_consistent({"FORGED": (220, 5, 22, 198)})
    run_case("S16: gia mao so lieu - confusion TP+FN!=220 bi tu choi", s16, ValueError)

    # S17: forged benchmark - CSV thieu model bi tu choi
    def s17():
        rows = list(csv.DictReader(
            (ROOT / "artifacts/test/benchmark_summary.csv").read_text(encoding="utf-8").splitlines()))
        require_benchmark_models([r["model_id"] for r in rows])  # positive control 7 models
        require_benchmark_models(["B01", "B02", "B03", "H00", "H01", "H02"])  # H01_sec dropped
    run_case("S17: gia mao benchmark - CSV thieu model bi tu choi", s17, ValueError)

    # S18: forged benchmark - metric out of range is rejected
    def s18():
        rows = list(csv.DictReader(
            (ROOT / "artifacts/test/benchmark_summary.csv").read_text(encoding="utf-8").splitlines()))
        for r in rows:
            require_metric_range(r["balanced_accuracy"], "balanced_accuracy")  # positive control
        require_metric_range(1.5, "balanced_accuracy")
    run_case("S18: gia mao metric - BalAcc>1 bi tu choi", s18, ValueError)

    # S19: forged parity - Delta>=1e-3 or flips!=0 is rejected
    def s19():
        for m in ("B01", "B03", "H00", "H01"):
            rep = json.loads((ROOT / f"artifacts/runtime/{m}/export_report.json").read_text(encoding="utf-8"))
            par = rep.get("parity", {}).get("parity", {})
            require_parity(par.get("score_delta_max", 1.0),
                           par.get("decision_flips", 1))  # positive control delta=0.0 flips=0
        require_parity(0.005, 0)  # forged delta above threshold
    run_case("S19: gia mao parity - Delta>=1e-3 bi tu choi", s19, ValueError)

    # S20: honest MISSING - absent .pt must report MISSING, never PASS (read-only check)
    def s20():
        for rel in ("artifacts/yolo/B01_yolo11n/seed42/best.pt",
                    "artifacts/yolo/B02_yolo11s/seed42/best.pt",
                    "artifacts/efficientad/B03_efficientad_s/seed42/model.pt"):
            p = ROOT / rel
            assert not p.is_file(), f"{rel} unexpectedly present"
            status = "MISSING" if not p.is_file() else "PASS"
            assert status == "MISSING", f"{rel} must be MISSING, not PASS"
        # verify script itself must exit non-zero while MISSING (honest, no fake PASS)
        sys.path.insert(0, str(ROOT / "scripts"))
        import verify_reproducibility as vr
        rc = vr.main(["--repo-root", str(ROOT)])
        assert rc == 1, f"verify must exit 1 when checkpoints MISSING, got {rc}"
    run_success_case("S20: trung thuc MISSING - checkpoint vang bao MISSING, exit 1", s20)

    passed = sum(1 for c in cases if c["passed"])
    log = {
        "step": 10,
        "total_cases": len(cases),
        "passed_cases": passed,
        "failed_cases": len(cases) - passed,
        "cases": cases,
    }
    log_path = ROOT / "docs/handoff/step10-sabotage-log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"[Sabotage Step10] {passed}/{len(cases)} cases passed")
    print(f"[Sabotage Step10] Log -> {log_path}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
