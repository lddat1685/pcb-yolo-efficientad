# Step 5 IMPLEMENTER — 2-D Grid Search và Fusion H00 vs H01

## Phase 0 — Diễn giải yêu cầu (2026-09-28)

Đã đọc `WORKFLOW.md`, `step5-contract.md`, `PLAN.md` (§5 Bước 5, §7.1–§7.6).
Worktree `../PCB_Lab_impl5`, nhánh `step5/impl`;
venv `./.cache/step3-verifier-b/.venv/Scripts/python.exe`.
Không có AGENTS.md/CLAUDE.md áp dụng.

### Diễn giải

- Nạp dự đoán **tĩnh** B01 (YOLO11n), B02 (YOLO11s), B03 (EfficientAD-S) trên 2 split
  cho phép: `calibration` (460 ảnh: 230 good / 230 defect) và `fusion`
  (`development_selection`, 306 ảnh: 153 good / 153 defect). Tuyệt đối không chạm `test`.
- Quét lưới 2-D đồng thời `(τ_yolo, τ_ad)`: τ_yolo ∈ [0.10, 0.90] bước 0.02 (41 điểm),
  τ_ad 50 điểm trên `[min(p10, μ−σ), max(p99.5, μ+3σ)]` từ phân phối defect calibration
  → tổng **2050 điểm lưới** (`grid_calibration.json`).
- Chọn `(τ_yolo*, τ_ad*)` tối đa **Defect Recall** tại ràng buộc **FRR ≤ 5%** (chính)
  và FRR ≤ 1% (phụ); hòa → τ_yolo thấp hơn (early-exit), rồi ECPB thấp hơn.
- H01 cascade: YOLO dương → DEFECT early-exit (`route_ad=False`, `ad_score=None`,
  `reason="yolo_hit"`); ngược lại route AD. H00 OR: cả 2 nhánh luôn chạy
  (`route_ad=True`). Cùng cặp ngưỡng, đánh giá cả hai trên 306 ảnh fusion +
  baseline B01/B03, tính Δrecall/ΔFRR/ΔECPB/Δlatency.
- ECPB chuẩn v3-P0.3: `N_sim=10.000`, prevalence 1%, `C_FN=100`, `C_FP=1`.
- H02 (cascade với YOLO11s) là đối chứng dung lượng detector, cùng phương pháp.

### Danh sách file chạm vào

- `configs/models/fusion_common.yaml` (mới): `fusion_recipe_v1`, lưới, ràng buộc, cost matrix.
- `src/pcb_lab/fusion/__init__.py`, `rules.py`, `metrics.py`, `sweep.py`, `evaluate.py` (mới).
- `scripts/calibrate_fusion.py` (mới): nạp preds tĩnh → sweep H01/H02 → evaluate →
  `thresholds.json` + `routing_policy.json` + `grid_calibration.json` +
  `reports/step5_calibration_fusion_report.md`.
- `tests/test_fusion_{rules,metrics,sweep,contract}.py` (mới).
- `artifacts/fusion/{thresholds.json,routing_policy.json,grid_calibration.json}` (mới, JSON nhỏ được commit theo tiền lệ Step 4).
- `docs/handoff/step5-impl.md` (file này).

### Quyết định mở / lệch hợp đồng đã xử lý (không sửa file hợp đồng)

1. `rules.py`: detections thực tế của Step 3 là **dict** (`{"confidence": …}`), không phải
   list `[x1..conf]` — đã thêm `_filter_yolo_boxes` hỗ trợ cả hai định dạng (trước đó filter
   list-format luôn rỗng với dữ liệu thật, nhưng do fallback `or raw` nên kết quả không sai).
2. `compute_hybrid_metrics(decisions, yolo_only_decisions=None)` theo contract thiếu
   `tau_yolo` cần cho rescue/added-FR — giữ tương thích cả hai cách gọi: keyword
   `tau_yolo=` (chính), list vị trí thứ 2 = `yolo_only_decisions`, form contract không
   ngưỡng → rescue fields về `None`/0.
3. `compute_delta_metrics` trả thêm alias ngắn `delta_recall/delta_frr/delta_ecpb`
   đúng tên contract, giữ nguyên khóa dài `*_H01_minus_H00` đang dùng.
4. `sweep.py`: thêm alias `GridSearchResult = SweepOutcome` đúng tên contract.
5. `sweep.py`: bound dưới τ_ad cũ dùng `μ−1.5σ`, đã sửa đúng contract `min(p10, μ−σ)`.
   Cả hai bound đều âm (contract −0.1257, cũ −0.9184) → cùng clamp 0.0 → **lưới và
   ngưỡng tối ưu không đổi** (xác minh: cùng `(0.14, 0.7879)` trước/sau sửa).
6. `evaluate.py`: thêm wrapper contract `evaluate_fusion(yolo_fusion_rows, ad_fusion_rows, policy)`
   lấy ngưỡng từ policy; `evaluate_split` giữ làm engine.
7. `thresholds.json`: thêm `target_frr: 0.05` và `calibration_metrics` cho `h00_yolo11n_or`
   (OR dùng chung ngưỡng H01 nên metrics calibration đồng nhất).
8. `routing_policy.json`: thêm `source_checksums` (SHA-256 của 6 file preds B01/B02/B03).
9. Tên file lưới: `grid_calibration_h01.json` → **`grid_calibration.json`** đúng contract
   (đã xóa file tên cũ, cập nhật test).
10. Báo cáo: thêm cột MCC (contract §3 yêu cầu), hiển thị `N/A` (không phải `0.0%`) khi
    `rescue_rate is None`.

### Câu hỏi chặn: không có — mọi mơ hồ đều giải được từ contract/PLAN/dữ liệu thật.

## Phase 1 & 2 — Triển khai và xác minh thực tế

### Lệnh đã chạy thật (venv pinned, tại `../PCB_Lab_impl5`)

```powershell
<venv>/python.exe scripts/calibrate_fusion.py
<venv>/python.exe -m pytest tests/test_fusion_rules.py tests/test_fusion_metrics.py tests/test_fusion_sweep.py tests/test_fusion_contract.py -q -p no:cacheprovider
```

### Kết quả quét lưới (calibration 460 ảnh, 2050 điểm, eligible FRR≤5%: 1245/2050)

Phân phối AD defect: μ=1.4598, σ=1.5855, p10=0.4092, p99.5=9.6522.

| Model | τ_yolo* | τ_ad* | Recall | FRR | BalAcc | Prec | F1 | MCC | ECPB | Route AD% |
|---|---|---|---|---|---|---|---|---|---|---|
| **H01** chính (FRR≤5%) | 0.14 | 0.7879 | 100.00% | 4.78% (TP230/TN219/FP11/FN0) | 97.61% | 95.44% | 0.9766 | 0.9533 | 0.0473 | 47.6% (good 95.2%, defect 0.0%) |
| H01 phụ (FRR≤1%) | 0.48 | 0.7879 | 100.00% | 0.87% | — | — | — | — | 0.0086 | — |
| **H02** chính (FRR≤5%) | 0.38 | 0.7879 | 100.00% | 4.78% | 97.61% | 95.44% | 0.9766 | 0.9533 | 0.0473 | 47.6% |
| H02 phụ (FRR≤1%) | 0.88 | 0.7879 | 90.00% | 0.43% | — | — | — | — | 0.1043 | — |

### Đánh giá fusion / development_selection (306 ảnh)

| Phương pháp (ngưỡng H01) | BalAcc | Recall | Prec | F1 | MCC | FRR | ECPB | Route% | Rescued | AddedFR |
|---|---|---|---|---|---|---|---|---|---|---|
| B01 YOLO11n đơn | 80.72% | 100.00% | 72.17% | 0.8384 | 0.6659 | 38.56% | 0.3818 | 0.0% | N/A | 0 |
| B03 AD-S đơn | 75.82% | 52.94% | 97.59% | 0.6864 | 0.5807 | 1.31% | 0.4835 | 100% | N/A | N/A |
| H00 OR | 80.39% | 100.00% | 71.83% | 0.8361 | 0.6608 | 39.22% | 0.3882 | 100% | 0 | 1 |
| **H01 cascade** | 80.39% | 100.00% | 71.83% | 0.8361 | 0.6608 | 39.22% | 0.3882 | **30.7%** | 0 | 1 |

| Phương pháp (ngưỡng H02) | BalAcc | Recall | Prec | F1 | MCC | FRR | ECPB | Route% |
|---|---|---|---|---|---|---|---|---|
| H00/H01 (đồng nhất) | 92.16% | 100.00% | 86.44% | 0.9273 | 0.8537 | 15.69% | 0.1553 | H01: 42.2% |

- Δ(H01−H00): recall +0.00%, FRR +0.00%, ECPB +0.0000 (đồng nhất tuyệt đối ở cả hai
  ngưỡng). Δlatency = null — preds tĩnh không có `timing_ms` (null toàn bộ), không bịa latency.
- `rescue_rate = N/A` (không phải 0%): YOLO tại ngưỡng khóa bắt 100% defect fusion
  (`yolo_fn_count=0`) nên không có mẫu để cứu — báo cáo hiển thị N/A.

### Đọc trung thực (không chỉnh code/kỳ vọng cho khớp)

1. **FRR calibration→fusion lệch lớn ở ngưỡng H01**: 4.78% → 39.22% (H02: 4.78% → 15.69%).
   Ngưỡng τ_yolo=0.14 quá thấp để giữ trên fusion; H02 (YOLO11s, τ=0.38) tổng quát hóa tốt hơn.
2. **Tại điểm khóa, fusion không cộng recall**: AD chỉ thêm đúng 1 FP trên good
   (59→60 FP ở H01; H02 thêm 0 FP). Lợi ích đo được của cascade ở đây là **early-exit**
   (69.3% ảnh H01 / 57.8% ảnh H02 không cần chạy AD), không phải rescue — đúng phạm vi
   kiểm chứng H5, chờ seed 43/44 và test cuối (Bước 6/7) kết luận.
3. Giới hạn cascade [S1] giữ nguyên: defect nào YOLO đã dương thì AD bị khóa hoàn toàn.

### Bất biến đã kiểm

- **Zero Test Leak**: `grep '"split": "test"'` trên 6 file preds = 0; mọi entrypoint
  (`predict_cascade`, `predict_or`, `grid_search_2d`, `evaluate_split`, `load_jsonl`)
  raise `PermissionError` với split test (có test bao phủ).
- **Tĩnh 100%**: `src/pcb_lab/fusion/` + runner không import torch/ultralytics/CUDA,
  không forward GPU (grep xác nhận).
- **Deterministic**: chạy `grid_search_2d` 2 lần → cùng ngưỡng, cùng 2050 điểm.
- **Test**: **10 passed** (`rules` 3, `metrics` 3, `sweep` 1, `contract` 3) bằng venv pinned.
- Checksum SHA-256 6 file preds nguồn đã khóa trong `routing_policy.json`
  (`source_checksums`).

### Handoff cho VERIFIER

Artifact: `artifacts/fusion/thresholds.json` (`target_frr: 0.05`), `routing_policy.json`
(`test_access: false` + checksums), `grid_calibration.json` (2050 điểm),
`reports/step5_calibration_fusion_report.md`. Tái chạy: một lệnh `calibrate_fusion.py`
bằng venv pinned (CPU, <1 phút). Chưa chạy benchmark test (dành Bước 7).
