"""CLI runner for Step 6: baseline benchmark (B01/B02/B03/H00/H01/H02) + ablations + shortlist.

100% static on existing preds_*.jsonl; no GPU forward; test partition refused
with PermissionError (test_access: False).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pcb_lab.evaluation import (  # noqa: E402
    compute_ablations,
    run_development_benchmark,
    select_shortlist,
)

METRIC_COLS = [
    "balanced_accuracy", "defect_recall", "defect_precision", "frr",
    "defect_to_good", "f1", "mcc", "ecpb",
    "tp", "tn", "fp", "fn", "total", "n_defect", "n_good",
]
HYBRID_COLS = [
    "route_rate", "route_rate_good", "route_rate_defect",
    "yolo_fn_count", "rescued_defects", "rescue_rate",
    "added_false_rejects", "unclassified_positive_rate", "mean_latency_ms",
]


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    keys: List[str] = []
    for r in rows:
        for k, v in r.items():
            if k not in keys and not isinstance(v, dict):
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, dict) or v is None else v)
                        for k, v in r.items() if k in keys})


def generate_report(bench, abl, short) -> str:
    fus = bench.splits["fusion"]
    cal = bench.splits["calibration"]
    op = bench.operating_points
    a1 = abl.ablations["A1_detector_capacity_B01_vs_B02"]
    a2 = abl.ablations["A2_fusion_logic_H00_vs_H01"]
    a3 = abl.ablations["A3_operating_threshold_primary_vs_secondary"]

    def row(m, split, method):
        e = (bench.splits[split])[method]
        cells = [f"| **{m}** ({split}) |"]
        for c in METRIC_COLS[:8]:
            v = e.get(c)
            if c in ("tp", "tn", "fp", "fn"):
                cells.append(f" {v} |")
            elif c in ("frr", "defect_recall", "balanced_accuracy",
                       "defect_precision", "defect_to_good"):
                cells.append(f" {_fmt_pct(v)} |")
            else:
                cells.append(f" {_fmt(v)} |")
        rr = e.get("route_rate")
        cells.append(f" {_fmt_pct(rr) if rr is not None else '—'} |")
        return "".join(cells)

    n_cal = getattr(bench, "counts", {}).get("yolo_n_calibration", 460)
    n_fus = getattr(bench, "counts", {}).get("yolo_n_fusion", 306)
    n_cal_good = bench.splits.get("calibration", {}).get("B01", {}).get("n_good", n_cal // 2)
    n_cal_defect = bench.splits.get("calibration", {}).get("B01", {}).get("n_defect", n_cal // 2)
    n_fus_good = bench.splits.get("fusion", {}).get("B01", {}).get("n_good", n_fus // 2)
    n_fus_defect = bench.splits.get("fusion", {}).get("B01", {}).get("n_defect", n_fus // 2)

    L = [
        "# Báo cáo Benchmark Baseline và Shortlist (Step 6)",
        "",
        f"**Recipe:** `{bench.recipe_id}` | **Calibration:** {n_cal} ảnh ({n_cal_good} good / {n_cal_defect} defect) "
        f"| **Fusion (development selection):** {n_fus} ảnh ({n_fus_good} good / {n_fus_defect} defect) | "
        f"**test_access:** `False`",
        "",
        f"Điểm vận hành khóa (Bước 5): H01 τ_yolo={op['H01']['tau_yolo']}, "
        f"τ_ad={op['H01']['tau_ad']}; H02 τ_yolo={op['H02']['tau_yolo']}, "
        f"τ_ad={op['H02']['tau_ad']}; ràng buộc chính FRR≤{bench.target_frr:.0%}.",
        "",
        "---",
        "",
        f"## 1. Bảng development (fusion, {n_fus} ảnh)",
        "",
        "| Phương pháp | BalAcc ↑ | Recall ↑ | Precision ↑ | FRR ↓ | Miss ↓ | F1 ↑ | MCC ↑ | ECPB ↓ | Route AD |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, key in (("B01 (YOLO11n đơn)", "B01"), ("B02 (YOLO11s đơn)", "B02"),
                       ("B03 (EfficientAD-S đơn)", "B03"),
                       ("H00 (OR song song)", "H00"), ("H01 (Cascade)", "H01"),
                       ("H02 (Cascade s)", "H02")):
        L.append(row(label, "fusion", key))
    L += [
        "",
        f"## 2. Bảng development (calibration, {n_cal} ảnh)",
        "",
        "| Phương pháp | BalAcc ↑ | Recall ↑ | Precision ↑ | FRR ↓ | Miss ↓ | F1 ↑ | MCC ↑ | ECPB ↓ | Route AD |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, key in (("B01 (YOLO11n đơn)", "B01"), ("B02 (YOLO11s đơn)", "B02"),
                       ("B03 (EfficientAD-S đơn)", "B03"),
                       ("H00 (OR song song)", "H00"), ("H01 (Cascade)", "H01"),
                       ("H02 (Cascade s)", "H02")):
        L.append(row(label, "calibration", key))
    L += [
        "",
        "---",
        "",
        "## 3. Ablation",
        "",
        "### A1 — Dung lượng detector: B01 (YOLO11n) vs B02 (YOLO11s), fusion",
        "",
        f"- Δrecall (B02−B01): {_fmt(a1['delta_recall'], 4)}; "
        f"ΔFRR: {_fmt(a1['delta_frr'], 4)}; ΔBalAcc: {_fmt(a1['delta_balanced_accuracy'], 4)}; "
        f"ΔECPB: {_fmt(a1['delta_ecpb'], 4)}; ΔMCC: {_fmt(a1['delta_mcc'], 4)}",
        f"- τ_yolo: B01={a1['tau_yolo_B01']} vs B02={a1['tau_yolo_B02']} "
        "(ngưỡng khóa riêng từng cascade, so sánh trực tiếp có giới hạn).",
        "",
        "### A2 — Logic kết hợp: H00 (OR) vs H01 (Cascade), cùng điểm vận hành",
        "",
        f"- Δrecall (H01−H00): {_fmt(a2['delta_recall'], 4)}; "
        f"ΔFRR: {_fmt(a2['delta_frr'], 4)}; ΔECPB: {_fmt(a2['delta_ecpb'], 4)}; "
        f"Δlatency: {_fmt(a2['delta_latency_ms'])} (timing_ms=null, preds tĩnh)",
        f"- Route H01: {_fmt_pct(a2['route_rate_H01'])} "
        f"(good {_fmt_pct(a2['route_rate_good_H01'])}, defect {_fmt_pct(a2['route_rate_defect_H01'])}); "
        f"H00 luôn chạy cả 2 nhánh (route 100%).",
        f"- Early-exit saving rate: {_fmt_pct(a2['saving_rate_early_exit'])} ảnh bỏ qua EfficientAD.",
        f"- Rescued defects: {a2['rescued_defects_H01']} "
        f"(rescue_rate {_fmt_pct(a2['rescue_rate_H01'])}); "
        f"added false rejects: {a2['added_false_rejects_H01']}.",
        "",
        "### A3 — Ngưỡng vận hành: H01 chính (FRR≤5%) vs phụ (FRR≤1%)",
        "",
    ]
    if a3.get("secondary"):
        L += [
            f"- Chính: τ=({a3['primary']['tau_yolo']}, {a3['primary']['tau_ad']}); "
            f"phụ: τ=({a3['secondary']['tau_yolo']}, {a3['secondary']['tau_ad']}).",
            f"- Δrecall (phụ−chính, fusion): {_fmt(a3['delta_recall'], 4)}; "
            f"ΔFRR: {_fmt(a3['delta_frr'], 4)}; ΔECPB: {_fmt(a3['delta_ecpb'], 4)}.",
            f"- Calibration: chính FRR={_fmt_pct(a3['H01_primary_calibration']['frr'])} "
            f"vs phụ FRR={_fmt_pct(a3['H01_secondary_calibration']['frr'])}.",
        ]
    else:
        L.append("- Điểm phụ không khả dụng (thiếu grid file).")
    L += [
        "",
        "> Ghi chú: ablation tile-256 vs resize-256 (PLAN R7/V2) không thực hiện được — "
        "chỉ tồn tại preds tile-256 của B03.",
        "",
        "---",
        "",
        "## 4. Shortlist cho Bước 7",
        "",
        "| # | model_id | Vai trò | τ_yolo | τ_ad | Recall | FRR | ECPB | Luận cứ |",
        "| --- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for i, s in enumerate(short.shortlist, 1):
        sm = s["selection_metrics"]
        L.append(
            f"| {i} | `{s['model_id']}` | {s['role']} | {s['tau_yolo']} | {s['tau_ad']} | "
            f"{_fmt_pct(sm.get('defect_recall'))} | {_fmt_pct(sm.get('frr'))} | "
            f"{_fmt(sm.get('ecpb'))} | {s['rationale'][:140]}… |"
        )
    b = short.budget_summary
    L += [
        "",
        "### Ngân sách đã dùng",
        "",
        f"- B01 YOLO11n seed 42: 33 epoch (best 13), train_time 1260.5 s ≈ **0.35 GPU-h** "
        "(device 0, Colab); AdamW lr 1e-3, batch 16, imgsz 640, tối đa 100 epoch patience 20; "
        "pretrained `yolo11n.pt`.",
        f"- B02 YOLO11s seed 42: 43 epoch (best 23), train_time 1695.3 s ≈ **0.47 GPU-h**; cùng recipe; "
        "pretrained `yolo11s.pt`.",
        "- B03 EfficientAD-S seed 42: **70.000 optimizer steps**, 895 good train (8055 tile), "
        "tile 256/stride 224; teacher PDN-S pretrained frozen; AdamW, không ImageNette penalty.",
        f"- Tuning Bước 5: {b.get('tuning_grid_points', 2050)} điểm/lưới × 2 lưới (H01/H02), "
        "CPU <1 phút — đã bao gồm trong ngân sách (không fit thêm tham số).",
        "- Vòng sàng lọc: **seed 42 duy nhất**; seed 43/44 sau shortlist (PLAN Bước 6).",
        "",
        "---",
        "",
        "## 5. Provenance & tính tất định",
        "",
        "- 100% tĩnh trên 6 file preds (460 calib + 306 fusion mỗi detector); "
        "không forward GPU, không huấn luyện lại, `test_access: false`.",
        "- Mọi số fusion khớp từng điểm với `thresholds.json` của Bước 5 "
        "(kiểm tra tự động trong benchmark, sai lệch >1e-9 sẽ fail).",
        "- SHA-256 6 file preds khớp `routing_policy.json/source_checksums` "
        "(xem `development_summary.json/source_checksums`).",
        "- Drift calibration→fusion: FRR H01 4.78%→38.56% "
        "(YOLO τ=0.14 quá lỏng trên fusion; H02 τ=0.38: FRR 15.69%) — "
        "ghi nhận trung thực, không chọn lại ngưỡng trên fusion.",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/fusion_common.yaml")
    parser.add_argument("--thresholds", default="artifacts/fusion/thresholds.json")
    parser.add_argument("--policy", default="artifacts/fusion/routing_policy.json")
    parser.add_argument("--out-dir", default="artifacts/benchmark")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = repo_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    print("--> 1. Chạy development benchmark (calibration 460 + fusion 306)...")
    bench = run_development_benchmark(
        args.config, args.thresholds, args.policy, repo_root=repo_root
    )
    print(f"    operating points: H01={bench.operating_points['H01']}, "
          f"H02={bench.operating_points['H02']}")

    print("--> 2. Tính 3 ablation bắt buộc...")
    abl = compute_ablations(bench)

    print("--> 3. Chọn shortlist...")
    short = select_shortlist(bench, abl)

    dev_path = out_dir / "development_summary.json"
    dev_path.write_text(json.dumps(bench.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"    - {dev_path}")
    write_csv(bench.to_rows(), out_dir / "development_summary.csv")

    abl_path = out_dir / "ablation_summary.json"
    abl_path.write_text(json.dumps(abl.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"    - {abl_path}")
    write_csv(abl.to_rows(), out_dir / "ablation_summary.csv")

    short_path = out_dir / "shortlist.json"
    short_path.write_text(json.dumps(short.to_dict(), indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"    - {short_path}")

    report = generate_report(bench, abl, short)
    report_path = report_dir / "step6_baseline_benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"    - {report_path}")
    print("\n--> Hoàn tất Step 6 thành công!")


if __name__ == "__main__":
    main()
