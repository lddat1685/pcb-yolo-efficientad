"""Export thesis assets for Step 10 (read-only; writes only exports/thesis/).

Reads locked benchmark/runtime CSVs and emits LaTeX (.tex) + Markdown (.md)
tables ready to paste into the graduation report or slides:
  1. overall comparison (test, 7 models)
  2. source-group breakdown (worst-group)
  3. per-class detection (6 defect classes)
  4. latency-memory profiling (Step 9, CPU float32)

Never touches Steps 1-9 source, artifacts or test suites.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "exports" / "thesis"

PCT = lambda x: f"{100 * float(x):.2f}\\%"


def tex_escape(value):
    return str(value).replace("_", r"\_")


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  -> {path}")


# ---------------------------------------------------------------- tables ---

def overall_table(rows: list[dict]) -> tuple[str, str]:
    header_md = ("| Model | BalAcc | Recall | Precision | F1 | MCC | FRR | ECPB | "
                 "Latency mean (ms) | p95 (ms) | Route |")
    lines_md = [header_md, "|" + "---|" * 11]
    lines_tex = [
        r"\begin{tabular}{lcccccccccc}",
        r"\hline",
        r"Model & BalAcc & Recall & Prec. & F1 & MCC & FRR & ECPB & Lat.\,mean (ms) & p95 (ms) & Route \\",
        r"\hline",
    ]
    for r in rows:
        mid = r["model_id"]
        route = r.get("route_rate_total", "N/A")
        route_s = "N/A" if route in ("N/A", "", None) else f"{100 * float(route):.1f}\\%"
        lines_md.append(
            f"| {mid} | {100 * float(r['balanced_accuracy']):.2f}% | "
            f"{100 * float(r['defect_recall']):.2f}% | {100 * float(r['defect_precision']):.2f}% | "
            f"{float(r['f1']):.4f} | {float(r['mcc']):.4f} | {100 * float(r['frr']):.2f}% | "
            f"{float(r['ecpb']):.4f} | {float(r['latency_mean_ms']):.1f} | "
            f"{float(r['latency_p95_ms']):.1f} | {route_s.replace(chr(92), '')} |")
        lines_tex.append(
            f"{tex_escape(mid)} & {PCT(r['balanced_accuracy'])} & {PCT(r['defect_recall'])} & "
            f"{PCT(r['defect_precision'])} & {float(r['f1']):.4f} & {float(r['mcc']):.4f} & "
            f"{PCT(r['frr'])} & {float(r['ecpb']):.4f} & {float(r['latency_mean_ms']):.1f} & "
            f"{float(r['latency_p95_ms']):.1f} & {route_s} \\\\")
    lines_tex += [r"\hline", r"\end{tabular}"]
    caption = ("% Test split: 440 images (220 good / 220 defect), frozen thresholds "
               "(H01 tau_yolo=0.14, tau_ad=1.4598). Source: artifacts/test/benchmark_summary.csv. Latencies are historical branch-sum estimates, not measured E2E.")
    return "\n".join(lines_md) + "\n\n*" + caption + "*\n", "\n".join(lines_tex) + "\n% " + caption + "\n"


def group_table(rows: list[dict]) -> tuple[str, str]:
    lines_md = ["| Model | Group | n | Recall | FRR | BalAcc |",
                "|---|---|---:|---:|---:|---:|"]
    lines_tex = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Model & Group & n & Recall & FRR & BalAcc \\",
        r"\hline",
    ]
    for r in rows:
        lines_md.append(
            f"| {r['model_id']} | {r['source_group']} | {r['sample_count']} | "
            f"{100 * float(r['defect_recall']):.2f}% | {100 * float(r['frr']):.2f}% | "
            f"{100 * float(r['balanced_accuracy']):.2f}% |")
        lines_tex.append(
            f"{tex_escape(r['model_id'])} & {r['source_group']} & {r['sample_count']} & "
            f"{PCT(r['defect_recall'])} & {PCT(r['frr'])} & {PCT(r['balanced_accuracy'])} \\\\")
    lines_tex += [r"\hline", r"\end{tabular}"]
    caption = ("% Per-group breakdown on the 2 test source groups (N=2 at group level). "
               "Source: artifacts/test/benchmark_per_group.csv.")
    return "\n".join(lines_md) + "\n\n*" + caption + "*\n", "\n".join(lines_tex) + "\n% " + caption + "\n"


def class_table(rows: list[dict]) -> tuple[str, str]:
    keep = [r for r in rows if r["model_id"] in ("B01", "B02", "H01")]
    lines_md = ["| Model | Class | GT boxes | Detected | AP@0.5 | Recall |",
                "|---|---|---:|---:|---:|---:|"]
    lines_tex = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Model & Class & GT & Det. & AP@0.5 & Recall \\",
        r"\hline",
    ]
    for r in keep:
        cls = r["class_name"].replace("_", r"\_")
        lines_md.append(
            f"| {r['model_id']} | `{r['class_name']}` | {r['gt_box_count']} | "
            f"{r['detected_box_count']} | {float(r['ap_50']):.4f} | "
            f"{100 * float(r['class_recall']):.2f}% |")
        lines_tex.append(
            f"{tex_escape(r['model_id'])} & {cls} & {r['gt_box_count']} & {r['detected_box_count']} & "
            f"{float(r['ap_50']):.4f} & {PCT(r['class_recall'])} \\\\")
    lines_tex += [r"\hline", r"\end{tabular}"]
    caption = ("% Per-class detection on test (box-level; H01 inherits B01 boxes). "
               "Source: artifacts/test/benchmark_per_class.csv.")
    return "\n".join(lines_md) + "\n\n*" + caption + "*\n", "\n".join(lines_tex) + "\n% " + caption + "\n"


def latency_table() -> tuple[str, str]:
    specs = [
        ("B01", "artifacts/runtime/B01/profile.json", "calibration 460"),
        ("B03", "artifacts/runtime/B03/profile.json", "calibration 460"),
        ("H01", "artifacts/runtime/H01/profile.json", "calibration 460"),
        ("H00", "artifacts/runtime/H00/profile_20.json", "calibration 20 (ref)"),
    ]
    lines_md = ["| Model | Split | Total mean (ms) | Total p50 (ms) | Total p95 (ms) | Peak RSS (MB) |",
                "|---|---|---:|---:|---:|---:|"]
    lines_tex = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Model & Split & Mean (ms) & p50 (ms) & p95 (ms) & Peak RSS (MB) \\",
        r"\hline",
    ]
    for mid, rel, label in specs:
        path = REPO / rel
        if not path.is_file():
            lines_md.append(f"| {mid} | {label} | — | — | — | — |")
            lines_tex.append(f"{tex_escape(mid)} & {label} & -- & -- & -- & -- \\\\")
            continue
        data = read_json(path)
        stages = data.get("summary", {}).get("stages", {})
        total = stages.get("total", {})
        lines_md.append(
            f"| {mid} | {label} | {total.get('mean_ms', float('nan')):.1f} | "
            f"{total.get('p50_ms', float('nan')):.1f} | {total.get('p95_ms', float('nan')):.1f} | "
            f"{total.get('peak_memory_mb', float('nan')):.1f} |")
        lines_tex.append(
            f"{tex_escape(mid)} & {label} & {total.get('mean_ms', float('nan')):.1f} & "
            f"{total.get('p50_ms', float('nan')):.1f} & {total.get('p95_ms', float('nan')):.1f} & "
            f"{total.get('peak_memory_mb', float('nan')):.1f} \\\\")
    lines_tex += [r"\hline", r"\end{tabular}"]
    caption = ("% CPU float32, batch 1, PyTorch backend (Step 9 profiler, 14 stages). "
               "H01 and H00 use different sample counts; no paired speedup claim. H01 bimodal: p50 early-exit vs p95 AD-routed. Source: artifacts/runtime/*/profile.json.")
    return "\n".join(lines_md) + "\n\n*" + caption + "*\n", "\n".join(lines_tex) + "\n% " + caption + "\n"


def confusion_block() -> tuple[str, str]:
    # Derived from locked test metrics: B01/H00/H01 share TP=220 FN=0 at tau 0.14
    # (recall 100%); FP from FRR=10% on 220 good -> 22 FP, TN=198.
    # H01_sec: recall 99.55% -> 219/220 (1 FN); FRR 5% -> 11 FP, TN=209.
    # B03: recall 36.36% -> 80/220 (140 FN); FRR 0% -> TN=220, FP=0.
    cells = {
        "B01/H00/H01": (220, 0, 22, 198),
        "H01_sec": (219, 1, 11, 209),
        "B03": (80, 140, 0, 220),
    }
    lines_md = ["| Model | TP | FN | FP | TN |", "|---|---:|---:|---:|---:|"]
    lines_tex = [r"\begin{tabular}{lcccc}", r"\hline",
                 r"Model & TP & FN & FP & TN \\", r"\hline"]
    for mid, (tp, fn, fp, tn) in cells.items():
        lines_md.append(f"| {mid} | {tp} | {fn} | {fp} | {tn} |")
        lines_tex.append(f"{tex_escape(mid)} & {tp} & {fn} & {fp} & {tn} \\\\")
    lines_tex += [r"\hline", r"\end{tabular}"]
    caption = ("% Confusion counts derived from locked test rates (220 defect / 220 good). "
               "B01/H00/H01 identical at the frozen point. Source: benchmark_summary.csv.")
    return "\n".join(lines_md) + "\n\n*" + caption + "*\n", "\n".join(lines_tex) + "\n% " + caption + "\n"


# ------------------------------------------------------------------ main ---

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    print(f"[Step 10] Exporting thesis assets to {out} ...")

    summary = read_csv(REPO / "artifacts/test/benchmark_summary.csv")
    groups = read_csv(REPO / "artifacts/test/benchmark_per_group.csv")
    per_class = read_csv(REPO / "artifacts/test/benchmark_per_class.csv")

    tables = {
        "overall_comparison": overall_table(summary),
        "per_group_breakdown": group_table(groups),
        "per_class_detection": class_table(per_class),
        "latency_memory": latency_table(),
        "confusion_counts": confusion_block(),
    }
    for name, (md, tex) in tables.items():
        write(out / f"{name}.md", f"# {name}\n\n" + md)
        write(out / f"{name}.tex", f"% {name} — generated by scripts/export_thesis_assets.py\n" + tex)

    index = ["# Thesis assets index",
             "",
             "Generated by `scripts/export_thesis_assets.py` from locked artifacts.",
             "LaTeX model IDs are escaped; compilation still requires a TeX engine.",
             "Step 7 latency columns preserve historical branch-sum estimates; they are not measured E2E.",
             "",
             "| Asset | Markdown | LaTeX | Source |",
             "|---|---|---|---|"]
    sources = {
        "overall_comparison": "artifacts/test/benchmark_summary.csv",
        "per_group_breakdown": "artifacts/test/benchmark_per_group.csv",
        "per_class_detection": "artifacts/test/benchmark_per_class.csv",
        "latency_memory": "artifacts/runtime/*/profile.json",
        "confusion_counts": "benchmark_summary.csv (derived)",
    }
    for name in tables:
        index.append(f"| {name} | `{name}.md` | `{name}.tex` | `{sources[name]}` |")
    index.append("")
    write(out / "README.md", "\n".join(index))
    print("[Step 10] Done: 5 tables x (md + tex) + README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
