# Báo cáo Benchmark Baseline và Shortlist (Step 6)

**Recipe:** `fusion_recipe_v1` | **Calibration:** 460 ảnh (230 good / 230 defect) | **Fusion (development selection):** 306 ảnh (153 good / 153 defect) | **test_access:** `False`

Điểm vận hành khóa (Bước 5): H01 τ_yolo=0.14, τ_ad=1.4598; H02 τ_yolo=0.38, τ_ad=1.4598; ràng buộc chính FRR≤5%.

---

## 1. Bảng development (fusion, 306 ảnh)

| Phương pháp | BalAcc ↑ | Recall ↑ | Precision ↑ | FRR ↓ | Miss ↓ | F1 ↑ | MCC ↑ | ECPB ↓ | Route AD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B01 (YOLO11n đơn)** (fusion) | 80.72% | 100.00% | 72.17% | 38.56% | 0.00% | 0.8384 | 0.6659 | 0.3818 | 0.00% |
| **B02 (YOLO11s đơn)** (fusion) | 92.16% | 100.00% | 86.44% | 15.69% | 0.00% | 0.9273 | 0.8537 | 0.1553 | 0.00% |
| **B03 (EfficientAD-S đơn)** (fusion) | 64.38% | 28.76% | 100.00% | 0.00% | 71.24% | 0.4467 | 0.4098 | 0.7124 | 100.00% |
| **H00 (OR song song)** (fusion) | 80.72% | 100.00% | 72.17% | 38.56% | 0.00% | 0.8384 | 0.6659 | 0.3818 | 100.00% |
| **H01 (Cascade)** (fusion) | 80.72% | 100.00% | 72.17% | 38.56% | 0.00% | 0.8384 | 0.6659 | 0.3818 | 30.72% |
| **H02 (Cascade s)** (fusion) | 92.16% | 100.00% | 86.44% | 15.69% | 0.00% | 0.9273 | 0.8537 | 0.1553 | 42.16% |

## 2. Bảng development (calibration, 460 ảnh)

| Phương pháp | BalAcc ↑ | Recall ↑ | Precision ↑ | FRR ↓ | Miss ↓ | F1 ↑ | MCC ↑ | ECPB ↓ | Route AD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B01 (YOLO11n đơn)** (calibration) | 97.61% | 100.00% | 95.44% | 4.78% | 0.00% | 0.9766 | 0.9533 | 0.0473 | 0.00% |
| **B02 (YOLO11s đơn)** (calibration) | 97.61% | 100.00% | 95.44% | 4.78% | 0.00% | 0.9766 | 0.9533 | 0.0473 | 0.00% |
| **B03 (EfficientAD-S đơn)** (calibration) | 61.52% | 23.04% | 100.00% | 0.00% | 76.96% | 0.3746 | 0.3609 | 0.7696 | 100.00% |
| **H00 (OR song song)** (calibration) | 97.61% | 100.00% | 95.44% | 4.78% | 0.00% | 0.9766 | 0.9533 | 0.0473 | 100.00% |
| **H01 (Cascade)** (calibration) | 97.61% | 100.00% | 95.44% | 4.78% | 0.00% | 0.9766 | 0.9533 | 0.0473 | 47.61% |
| **H02 (Cascade s)** (calibration) | 97.61% | 100.00% | 95.44% | 4.78% | 0.00% | 0.9766 | 0.9533 | 0.0473 | 47.61% |

---

## 3. Ablation

### A1 — Dung lượng detector: B01 (YOLO11n) vs B02 (YOLO11s), fusion

- Δrecall (B02−B01): 0.0000; ΔFRR: -0.2288; ΔBalAcc: 0.1144; ΔECPB: -0.2265; ΔMCC: 0.1878
- τ_yolo: B01=0.14 vs B02=0.38 (ngưỡng khóa riêng từng cascade, so sánh trực tiếp có giới hạn).

### A2 — Logic kết hợp: H00 (OR) vs H01 (Cascade), cùng điểm vận hành

- Δrecall (H01−H00): 0.0000; ΔFRR: 0.0000; ΔECPB: 0.0000; Δlatency: N/A (timing_ms=null, preds tĩnh)
- Route H01: 30.72% (good 61.44%, defect 0.00%); H00 luôn chạy cả 2 nhánh (route 100%).
- Early-exit saving rate: 69.28% ảnh bỏ qua EfficientAD.
- Rescued defects: 0 (rescue_rate N/A); added false rejects: 0.

### A3 — Ngưỡng vận hành: H01 chính (FRR≤5%) vs phụ (FRR≤1%)

- Chính: τ=(0.14, 1.4598); phụ: τ=(0.48, 1.4598).
- Δrecall (phụ−chính, fusion): 0.0000; ΔFRR: -0.2157; ΔECPB: -0.2135.
- Calibration: chính FRR=4.78% vs phụ FRR=0.87%.

> Ghi chú: ablation tile-256 vs resize-256 (PLAN R7/V2) không thực hiện được — chỉ tồn tại preds tile-256 của B03.

---

## 4. Shortlist cho Bước 7

| # | model_id | Vai trò | τ_yolo | τ_ad | Recall | FRR | ECPB | Luận cứ |
| --- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `B01_yolo11n` | Primary Baseline Detector | 0.14 | None | 100.00% | 38.56% | 0.3818 | YOLO11n baseline: 100.00% recall, FRR=38.56%, ECPB=0.3818. Calibration 460 samples; fusion 306 samples. Fastest single-stage detector in bas… |
| 2 | `B03_efficientad_s` | Primary Baseline Anomaly Detector | None | 1.4598 | 28.76% | 0.00% | 0.7124 | EfficientAD-S baseline: 28.76% recall, FRR=0.00%, precision=100.00%. Tile 256x256, stride 224, detector-only anomaly score. 70k optimizer st… |
| 3 | `H00_yolo11n_or_efficientad_s` | Hybrid OR Rule | 0.14 | 1.4598 | 100.00% | 38.56% | 0.3818 | OR-rule hybrid: 100.00% recall at H01 operating point, ECPB=0.3818. Full parallel evaluation (route_rate=1.0), no early-exit benefit. Serves… |
| 4 | `H01_cascade_yolo11n_efficientad_s` | Primary Hybrid Cascade | 0.14 | 1.4598 | 100.00% | 38.56% | 0.3818 | Cascade hybrid (YOLO11n -> EfficientAD-S): 100.00% defect recall at FRR=38.56%, ECPB=0.3818. Early-exit saves 69.3% of AD compute (route_rat… |
| 5 | `H02_cascade_yolo11s_efficientad_s` | Secondary Hybrid Cascade | 0.38 | 1.4598 | 100.00% | 15.69% | 0.1553 | Cascade hybrid (YOLO11s -> EfficientAD-S): 100.00% defect recall at FRR=15.69% on fusion, BalAcc=92.16%, ECPB=0.1553. Higher tau_yolo (0.38)… |

### Ngân sách đã dùng

- B01 YOLO11n seed 42: 33 epoch (best 13), train_time 1260.5 s ≈ **0.35 GPU-h** (device 0, Colab); AdamW lr 1e-3, batch 16, imgsz 640, tối đa 100 epoch patience 20; pretrained `yolo11n.pt`.
- B02 YOLO11s seed 42: 43 epoch (best 23), train_time 1695.3 s ≈ **0.47 GPU-h**; cùng recipe; pretrained `yolo11s.pt`.
- B03 EfficientAD-S seed 42: **70.000 optimizer steps**, 895 good train (8055 tile), tile 256/stride 224; teacher PDN-S pretrained frozen; AdamW, không ImageNette penalty.
- Tuning Bước 5: 2050 điểm/lưới × 2 lưới (H01/H02), CPU <1 phút — đã bao gồm trong ngân sách (không fit thêm tham số).
- Vòng sàng lọc: **seed 42 duy nhất**; seed 43/44 sau shortlist (PLAN Bước 6).

---

## 5. Provenance & tính tất định

- 100% tĩnh trên 6 file preds (460 calib + 306 fusion mỗi detector); không forward GPU, không huấn luyện lại, `test_access: false`.
- Mọi số fusion khớp từng điểm với `thresholds.json` của Bước 5 (kiểm tra tự động trong benchmark, sai lệch >1e-9 sẽ fail).
- SHA-256 6 file preds khớp `routing_policy.json/source_checksums` (xem `development_summary.json/source_checksums`).
- Drift calibration→fusion: FRR H01 4.78%→38.56% (YOLO τ=0.14 quá lỏng trên fusion; H02 τ=0.38: FRR 15.69%) — ghi nhận trung thực, không chọn lại ngưỡng trên fusion.
