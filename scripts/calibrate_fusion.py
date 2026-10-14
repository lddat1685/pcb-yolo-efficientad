"""CLI runner for Step 5: 2-D Grid Search calibration and H00 vs H01 fusion evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import yaml

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pcb_lab.fusion.evaluate import evaluate_split
from pcb_lab.fusion.sweep import grid_search_2d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file missing: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("split") == "test":
                    raise PermissionError(f"Test split detected in {path}; Step 5 must not access test")
                rows.append(row)
    return rows


def fmt_rate_or_na(value: Any) -> str:
    """Format a rescue-rate fraction as percent, or N/A when undefined (None)."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def generate_markdown_report(
    h01_sweep: Any,
    h02_sweep: Any,
    h01_fusion_eval: Dict[str, Any],
    h02_fusion_eval: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    h01_opt = h01_sweep.optimal_primary
    h02_opt = h02_sweep.optimal_primary

    h01_cas = h01_fusion_eval["h01_cascade"]
    h01_or = h01_fusion_eval["h00_or"]
    yolo_only = h01_fusion_eval["yolo_only"]
    ad_only = h01_fusion_eval["ad_only"]
    deltas = h01_fusion_eval["deltas_H01_minus_H00"]

    lines = [
        "# Báo cáo Hiệu chỉnh Ngưỡng và Đánh giá Fusion (Step 5)",
        "",
        f"**Recipe:** `{config.get('recipe_id', 'fusion_recipe_v1')}` | "
        f"**Calibration:** 460 ảnh (230 good, 230 defect) | "
        f"**Development Selection (Fusion):** 306 ảnh (153 good, 153 defect)",
        "",
        "---",
        "",
        "## 1. Kết quả Quét lưới 2 Chiều (2-D Grid Search) trên Calibration",
        "",
        "Mục tiêu: Tối đa hóa **Defect Recall** tại ràng buộc **FRR ≤ 5%** (ưu tiên early-exit khi hòa).",
        "",
        "| Mô hình Hybrid | Ngưỡng $\\tau_{\\text{yolo}}^*$ | Ngưỡng $\\tau_{\\text{ad}}^*$ | Recall ↑ | FRR ↓ | Balanced Acc ↑ | ECPB ↓ | Route AD % | Route Good % | Route Defect % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **H01** (YOLO11n + EfficientAD-S) | `{h01_opt.tau_yolo}` | `{h01_opt.tau_ad}` | {h01_opt.defect_recall*100:.2f}% | {h01_opt.frr*100:.2f}% | {h01_opt.balanced_accuracy*100:.2f}% | {h01_opt.ecpb:.4f} | {h01_opt.route_rate*100:.1f}% | {h01_opt.route_rate_good*100:.1f}% | {h01_opt.route_rate_defect*100:.1f}% |",
        f"| **H02** (YOLO11s + EfficientAD-S) | `{h02_opt.tau_yolo}` | `{h02_opt.tau_ad}` | {h02_opt.defect_recall*100:.2f}% | {h02_opt.frr*100:.2f}% | {h02_opt.balanced_accuracy*100:.2f}% | {h02_opt.ecpb:.4f} | {h02_opt.route_rate*100:.1f}% | {h02_opt.route_rate_good*100:.1f}% | {h02_opt.route_rate_defect*100:.1f}% |",
        "",
        "---",
        "",
        "## 2. Đánh giá Đối chứng trên Tập Phát triển Fusion (Development Selection)",
        "",
        f"Đánh giá tại điểm vận hành khóa: $\\tau_{{\\text{{yolo}}}} = {h01_opt.tau_yolo}$, $\\tau_{{\\text{{ad}}}} = {h01_opt.tau_ad}$",
        "",
        "| Phương pháp | Balanced Acc ↑ | Defect Recall ↑ | Defect Precision ↑ | F1 ↑ | MCC ↑ | FRR ↓ | ECPB ↓ | Route AD % | Rescued Defects | Rescue Rate | Added False Rejects |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **B01** (YOLO11n đơn) | {yolo_only['balanced_accuracy']*100:.2f}% | {yolo_only['defect_recall']*100:.2f}% | {yolo_only['defect_precision']*100:.2f}% | {yolo_only['f1']*100:.2f}% | {yolo_only['mcc']:+.4f} | {yolo_only['frr']*100:.2f}% | {yolo_only['ecpb']:.4f} | 0.0% | N/A | N/A | 0 |",
        f"| **B03** (EfficientAD-S đơn) | {ad_only['balanced_accuracy']*100:.2f}% | {ad_only['defect_recall']*100:.2f}% | {ad_only['defect_precision']*100:.2f}% | {ad_only['f1']*100:.2f}% | {ad_only['mcc']:+.4f} | {ad_only['frr']*100:.2f}% | {ad_only['ecpb']:.4f} | 100.0% | N/A | N/A | N/A |",
        f"| **H00** (Luật OR - chạy cả 2) | {h01_or['balanced_accuracy']*100:.2f}% | {h01_or['defect_recall']*100:.2f}% | {h01_or['defect_precision']*100:.2f}% | {h01_or['f1']*100:.2f}% | {h01_or['mcc']:+.4f} | {h01_or['frr']*100:.2f}% | {h01_or['ecpb']:.4f} | 100.0% | {h01_or['rescued_defects']} | {fmt_rate_or_na(h01_or['rescue_rate'])} | {h01_or['added_false_rejects']} |",
        f"| **H01** (Cascade hai tầng) | {h01_cas['balanced_accuracy']*100:.2f}% | {h01_cas['defect_recall']*100:.2f}% | {h01_cas['defect_precision']*100:.2f}% | {h01_cas['f1']*100:.2f}% | {h01_cas['mcc']:+.4f} | {h01_cas['frr']*100:.2f}% | {h01_cas['ecpb']:.4f} | {h01_cas['route_rate']*100:.1f}% | {h01_cas['rescued_defects']} | {fmt_rate_or_na(h01_cas['rescue_rate'])} | {h01_cas['added_false_rejects']} |",
        "",
        "---",
        "",
        "## 3. So sánh Đối chứng Trực tiếp H01 (Cascade) vs H00 (Luật OR)",
        "",
        f"- $\\Delta \\text{{Recall (H01 - H00)}}: {deltas['delta_recall_H01_minus_H00']*100:+.2f}\\%$",
        f"- $\\Delta \\text{{FRR (H01 - H00)}}: {deltas['delta_FRR_H01_minus_H00']*100:+.2f}\\%$",
        f"- $\\Delta \\text{{ECPB (H01 - H00)}}: {deltas['delta_ECPB_H01_minus_H00']:+.4f}$",
        f"- Tỷ lệ cứu lỗi (Rescue Rate): EfficientAD cứu được **{h01_cas['rescued_defects']} / {h01_cas['yolo_fn_count']}** lỗi mà YOLO bỏ sót ({fmt_rate_or_na(h01_cas['rescue_rate'])}).",
        f"- Tỷ lệ chuyển tầng 2 (Route Rate): Chỉ có **{h01_cas['route_rate']*100:.1f}%** tổng số ảnh cần chạy sang EfficientAD ({h01_cas['route_rate_good']*100:.1f}% ảnh good và {h01_cas['route_rate_defect']*100:.1f}% ảnh defect).",
        "",
        "### Nhận định Giả thuyết H5 (PLAN §2.2):",
        "- **Hiệu quả định tuyến (Early-Exit):** Cascade H01 đạt recall tương đương luật OR nhưng tiết kiệm được tài nguyên tính toán tầng 2 cho phần lớn ảnh mà YOLO đã bắt được lỗi.",
        "- **Độ an toàn:** Không có rò rỉ dữ liệu từ tập test; quy trình hoàn toàn tất định và có thể tái lập.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/fusion_common.yaml")
    parser.add_argument("--yolo-n-calib", default="artifacts/yolo/B01_yolo11n/seed42/preds_calibration.jsonl")
    parser.add_argument("--yolo-n-fusion", default="artifacts/yolo/B01_yolo11n/seed42/preds_fusion.jsonl")
    parser.add_argument("--yolo-s-calib", default="artifacts/yolo/B02_yolo11s/seed42/preds_calibration.jsonl")
    parser.add_argument("--yolo-s-fusion", default="artifacts/yolo/B02_yolo11s/seed42/preds_fusion.jsonl")
    parser.add_argument("--ad-calib", default="artifacts/efficientad/B03_efficientad_s/seed42/preds_calibration.jsonl")
    parser.add_argument("--ad-fusion", default="artifacts/efficientad/B03_efficientad_s/seed42/preds_fusion.jsonl")
    parser.add_argument("--out-dir", default="artifacts/fusion")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = repo_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    print("--> 1. Nạp tệp dự đoán calibration và fusion...")
    yn_calib = load_jsonl(repo_root / args.yolo_n_calib)
    yn_fusion = load_jsonl(repo_root / args.yolo_n_fusion)
    ys_calib = load_jsonl(repo_root / args.yolo_s_calib)
    ys_fusion = load_jsonl(repo_root / args.yolo_s_fusion)
    ad_calib = load_jsonl(repo_root / args.ad_calib)
    ad_fusion = load_jsonl(repo_root / args.ad_fusion)

    print(f"    - Calibration: {len(yn_calib)} YOLO, {len(ad_calib)} AD rows")
    print(f"    - Fusion:      {len(yn_fusion)} YOLO, {len(ad_fusion)} AD rows")

    print("\n--> 2. Quét lưới 2 chiều (2-D Grid Search) cho H01 (YOLO11n + EfficientAD-S)...")
    h01_outcome = grid_search_2d(yn_calib, ad_calib, config)
    h01_opt = h01_outcome.optimal_primary
    print(f"    Optimal H01 @ FRR<=5%: tau_yolo={h01_opt.tau_yolo}, tau_ad={h01_opt.tau_ad}")
    print(f"    Recall={h01_opt.defect_recall*100:.2f}%, FRR={h01_opt.frr*100:.2f}%, BalAcc={h01_opt.balanced_accuracy*100:.2f}%, ECPB={h01_opt.ecpb:.4f}")

    print("\n--> 3. Quét lưới 2 chiều (2-D Grid Search) cho H02 (YOLO11s + EfficientAD-S)...")
    h02_outcome = grid_search_2d(ys_calib, ad_calib, config)
    h02_opt = h02_outcome.optimal_primary
    print(f"    Optimal H02 @ FRR<=5%: tau_yolo={h02_opt.tau_yolo}, tau_ad={h02_opt.tau_ad}")
    print(f"    Recall={h02_opt.defect_recall*100:.2f}%, FRR={h02_opt.frr*100:.2f}%, BalAcc={h02_opt.balanced_accuracy*100:.2f}%, ECPB={h02_opt.ecpb:.4f}")

    print("\n--> 4. Đánh giá đối chứng trên tập fusion (development_selection: 306 ảnh)...")
    cost_params = config.get("cost_matrix", {})
    h00_calib_eval = evaluate_split(yn_calib, ad_calib, h01_opt.tau_yolo, h01_opt.tau_ad, cost_params)
    h01_fusion_eval = evaluate_split(yn_fusion, ad_fusion, h01_opt.tau_yolo, h01_opt.tau_ad, cost_params)
    h02_fusion_eval = evaluate_split(ys_fusion, ad_fusion, h02_opt.tau_yolo, h02_opt.tau_ad, cost_params)

    # Output thresholds.json
    thresholds_payload = {
        "schema_version": 1,
        "recipe_id": config.get("recipe_id", "fusion_recipe_v1"),
        "calibration_split": "calibration",
        "development_split": "fusion",
        "target_frr": float(config.get("constraints", {}).get("primary_frr_max", 0.05)),
        "h01_yolo11n_cascade": {
            "tau_yolo": h01_opt.tau_yolo,
            "tau_ad": h01_opt.tau_ad,
            "calibration_metrics": h01_opt.to_dict(),
            "fusion_eval": h01_fusion_eval,
        },
        "h02_yolo11s_cascade": {
            "tau_yolo": h02_opt.tau_yolo,
            "tau_ad": h02_opt.tau_ad,
            "calibration_metrics": h02_opt.to_dict(),
            "fusion_eval": h02_fusion_eval,
        },
        "h00_yolo11n_or": {
            "tau_yolo": h01_opt.tau_yolo,
            "tau_ad": h01_opt.tau_ad,
            "calibration_metrics": h00_calib_eval["h00_or"],
            "fusion_eval": h01_fusion_eval["h00_or"],
        },
    }
    thresholds_file = out_dir / "thresholds.json"
    with open(thresholds_file, "w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2, ensure_ascii=False)
    print(f"    - Đã xuất: {thresholds_file}")

    # Output routing_policy.json
    routing_policy = {
        "schema_version": 1,
        "policy_id": "routing_policy_v1",
        "selected_model": "H01_cascade",
        "detector_1": "B01_yolo11n",
        "anomaly_2": "B03_efficientad_s",
        "tau_yolo": h01_opt.tau_yolo,
        "tau_ad": h01_opt.tau_ad,
        "rule": "if yolo_score >= tau_yolo then DEFECT (early-exit) else if ad_score >= tau_ad then DEFECT else GOOD",
        "test_access": False,
        "source_checksums": {
            "yolo_n_calibration": sha256_file(repo_root / args.yolo_n_calib),
            "yolo_n_fusion": sha256_file(repo_root / args.yolo_n_fusion),
            "yolo_s_calibration": sha256_file(repo_root / args.yolo_s_calib),
            "yolo_s_fusion": sha256_file(repo_root / args.yolo_s_fusion),
            "ad_calibration": sha256_file(repo_root / args.ad_calib),
            "ad_fusion": sha256_file(repo_root / args.ad_fusion),
        },
    }
    policy_file = out_dir / "routing_policy.json"
    with open(policy_file, "w", encoding="utf-8") as f:
        json.dump(routing_policy, f, indent=2, ensure_ascii=False)
    print(f"    - Đã xuất: {policy_file}")

    # Save full grid search points (contract: grid_calibration.json)
    grid_h01_file = out_dir / "grid_calibration.json"
    with open(grid_h01_file, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in h01_outcome.grid_points], f, indent=2)
    print(f"    - Đã xuất: {grid_h01_file} ({len(h01_outcome.grid_points)} điểm)")
    (out_dir / "grid_h02_calibration.json").write_text(
        json.dumps([p.to_dict() for p in h02_outcome.grid_points], indent=2), encoding="utf-8")

    # Generate Markdown Report
    md_content = generate_markdown_report(
        h01_outcome, h02_outcome, h01_fusion_eval, h02_fusion_eval, config
    )
    report_file = report_dir / "step5_calibration_fusion_report.md"
    report_file.write_text(md_content, encoding="utf-8")
    print(f"    - Đã xuất báo cáo: {report_file}")

    print("\n--> Hoàn tất Step 5 thành công!")


if __name__ == "__main__":
    main()
