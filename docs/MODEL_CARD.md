# Thẻ mô hình (Model Card) — PCB Hybrid YOLO + EfficientAD

Protocol: `deeppcb_cascade_holdout_v1` · Dataset: DeepPCB 640×640 · Split khóa test 440 ảnh.
Mọi checkpoint và ngưỡng dưới đây là bản **đã khóa (frozen)** phục vụ demo và tái lập;
không tune thêm trên test. Nguồn chân lý: `artifacts/test/frozen_evaluation.json`.

> SHA-256 tập test: `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24`
> (440 ảnh: 220 good / 220 defect).

---

## 1. B01 — YOLO11n (baseline detector, tầng một mặc định)

- **Kiến trúc:** YOLO11n (Ultralytics), detector 6 lớp lỗi PCB, input `imgsz=640`.
- **Số tham số:** ~2.6M (YOLO11n chuẩn; checkpoint `best.pt` ~5.3 MB trên đĩa).
- **Checkpoint:**
  - Đường dẫn: `artifacts/yolo/B01_yolo11n/seed42/best.pt`
  - SHA-256 (best): `c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05`
  - SHA-256 (last): `4de28ac1fbb70c3243c33e97aec0f2fb3b99f09784cd257338ade41c4dfbbfad`
  - Pretrained khởi tạo: `artifacts/pretrained/yolo11n.pt`
    SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
    (nguồn `ultralytics-assets(github)`).
- **Dữ liệu train:** 1.792 ảnh train (895 good + 897 defect); chọn checkpoint trên
  calibration 460 ảnh; không dùng test để chọn epoch/ngưỡng.
- **Recipe:** `yolo_aug_v1` — hình học thuần (xoay 90°×k + lật ngang),
  tắt photometric/mosaic; optimizer AdamW, `lr0=0.001`, batch 16, epochs 100,
  patience 20, deterministic, seed 42. Thực chạy: 33 epoch (best epoch 13),
  `train_time_s=1260.5` (~0.35 GPU-h trên Colab CUDA).
- **Môi trường train:** ultralytics 8.4.161, torch 2.14.0+cu130, Python 3.13.15.
- **Preprocessing / inference đã khóa:** `prep_v1`
  (hash `69e65565a2402077b41380b2d742ec989d939bf2cf6f3c7dbd80182fe177f770`),
  `conf_floor=0.001`, `iou=0.7`, `max_det=300`, `agnostic_nms=false`, `half=false`.
  Class order: `open_circuit, short, mouse_bite, spur, spurious_copper, pin_hole`.
- **Ngưỡng vận hành đã khóa:** `τ_yolo = 0.14` (primary, FRR ≤ 5% trên calibration).
- **Validation calibration (F1-optimal, không phải ngưỡng vận hành):**
  mAP50 = 0.9341, mAP50-95 = 0.7241, precision = 0.9200, recall = 0.8945.
- **Kết quả test (τ=0.14):** BalAcc 95.00%, recall 100.00%, precision 90.91%,
  FRR 10.00%, ECPB 0.0990, latency mean ~120 ms (CPU).
- **Giới hạn riêng:** FRR test (10%) cao hơn mục tiêu calibration (4.78%) — domain shift
  giữa các nhóm nguồn; xem `docs/LIMITATIONS.md`.

## 2. B02 — YOLO11s (đối chứng dung lượng detector)

- **Kiến trúc:** YOLO11s (Ultralytics), cùng schema 6 lớp, `imgsz=640`.
- **Số tham số:** ~9.4M (YOLO11s chuẩn; checkpoint `best.pt` ~19 MB).
- **Checkpoint:**
  - Đường dẫn: `artifacts/yolo/B02_yolo11s/seed42/best.pt`
  - SHA-256 (best): `aa079248bd566b3adf8863e3b43dde93d3392307a3ef17107e4c126bf410936c`
  - SHA-256 (last): `1f1ce96cf729af3f998bf6f9d4b47a5b94782088b8569ccf1627f6ea656d5850`
  - Cùng pretrained họ YOLO11, cùng recipe `yolo_aug_v1`, seed 42.
  - Thực chạy: 43 epoch (best epoch 23), `train_time_s=1695.3` (~0.47 GPU-h).
- **Preprocessing / inference:** giống B01 (`prep_v1`, cùng hash, cùng
  `conf_floor/iou/max_det`). `config_hash=6070bffbb4b71c71929fce974acfa0e81148dc12d42882a40f226e45df3afa7b`.
- **Ngưỡng vận hành đã khóa:** `τ_yolo = 0.38`.
- **Kết quả test (τ=0.38):** BalAcc 82.50%, recall 100.00%, precision 74.07%,
  FRR 35.00%, ECPB 0.3465, latency mean ~166 ms.
- **Vai trò:** đối chứng dung lượng — detector lớn hơn nhưng FRR tệ hơn B01 tại điểm
  vận hành đã khóa; không chọn làm tầng một chính.

## 3. B03 — EfficientAD-S (baseline anomaly, tầng hai mặc định)

- **Kiến trúc:** EfficientAD-S — teacher PDN-S pretrained frozen, student 768 kênh +
  autoencoder; học chỉ từ good train (normal-only).
- **Số tham số:** ~8.06M (checkpoint `model.pt` ~31 MB; TorchScript export cùng cỡ).
- **Checkpoint:**
  - Đường dẫn: `artifacts/efficientad/B03_efficientad_s/seed42/model.pt`
  - SHA-256: `9453bc2c740e81dfb8949df929ed001d60ebf8e5b0f6a18c95eddd9b3a2814a0`
  - Teacher pretrained: `artifacts/pretrained/efficientad_teacher_s.pt` (frozen).
- **Dữ liệu train:** 895 good train → 8.055 tile; 70.000 optimizer steps (~1.30 GPU-h
  Colab); AdamW (`lr=1e-4`, `weight_decay=1e-5`); không ImageNette penalty, không
  augmentation, không auto-threshold.
- **TileManager đã khóa:** tile 256×256, stride 224 (overlap 32 px), pad reflect,
  stitch linear blending (feathering), Global NMS IoU 0.45. Ảnh 640×640 → 9 tile;
  mọi tile cùng ảnh thuộc cùng partition.
- **Chuẩn hóa map:** thống kê teacher từ train good; chuẩn hóa map từ 230 calibration
  good (2.070 tile): `quantile_min=0.1174 (p90)`, `quantile_max=0.3574 (p99.5)`,
  `mean=0.0468`, `std=0.0711`; công thức `mean(0.1·(branch−q_min)/max(q_max−q_min,1e-12))`
  trên raw 56×56 tile-map trước padding. Defect calibration chỉ dùng báo cáo mô tả,
  không fit normalization.
- **Image score:** `max` của anomaly map đã stitch 640×640.
  Phân phối calibration: good mean 0.191 (p95 0.526, max 0.724);
  defect mean 1.460 (p50 0.991, p95 4.778, max 9.916).
- **Heatmap → box (khóa):** `τ_pixel=0.5` → Morph Open k=3 → Connected Components →
  `min_area=20` → Bounding Rect → Global NMS 0.45; nhãn `anomaly_unclassified`
  (không ép thành 1 trong 6 lớp).
- **Ngưỡng vận hành đã khóa:** `τ_ad = 1.4598` (mean defect calibration).
- **Kết quả test:** BalAcc 68.18%, recall 36.36%, precision 100.00%, FRR 0.00%,
  ECPB 0.6364, latency mean ~2733 ms (CPU, batch tile=1 ở UI/CLI).
- **Giới hạn riêng:** recall thấp — AD-only không đủ thay detector 6 lớp; chỉ có giá trị
  trong hybrid khi YOLO bỏ sót.

## 4. H00 — YOLO11n + EfficientAD-S, luật OR (đối chứng)

- **Kiến trúc:** kế thừa đúng checkpoint B01 + B03, không train thêm.
- **Luật:** chạy cả hai nhánh mọi ảnh; `DEFECT` nếu
  `(yolo_score ≥ 0.14) OR (ad_score ≥ 1.4598)`. Route 100%.
- **Ngưỡng:** `τ_yolo=0.14`, `τ_ad=1.4598` (cùng H01 — so sánh trực tiếp giá trị định tuyến).
- **Kết quả test:** BalAcc 95.00%, recall 100.00%, precision 90.91%, FRR 10.00%,
  ECPB 0.0990, latency mean ~2853 ms (≈ T_yolo + T_ad, không early-exit).
- **Vai trò:** upper-bound kết hợp không điều kiện; mọi delta H01−H00 về recall/FRR
  đều bằng 0 tại điểm khóa — lợi ích duy nhất của cascade là latency.

## 5. H01 — YOLO11n → EfficientAD-S, cascade (hybrid chính, demo mặc định)

- **Kiến trúc:** kế thừa đúng checkpoint B01 + B03, không train thêm.
- **Luật:** `if yolo_score ≥ 0.14 → DEFECT (early-exit, không chạy AD)`;
  ngược lại chạy AD tile-256 và so `τ_ad=1.4598`.
- **Ngưỡng:** `τ_yolo=0.14`, `τ_ad=1.4598` (chọn bằng 2-D grid 2050 điểm trên
  calibration, tiêu chí tối đa recall tại ràng buộc FRR ≤ 5%; calibration:
  recall 100%, FRR 4.78%, route 47.6%).
- **Kết quả test:** BalAcc 95.00%, recall 100.00%, precision 90.91%, FRR 10.00%,
  ECPB 0.0990; route 45.0% (good 90.0%, defect 0.0%); rescued 0; added FR 0;
  latency mean 1349 ms, p50 ~132 ms (early-exit) / p95 ~2938 ms (routed).
  Δ(H01−H00): recall +0, FRR +0, latency mean −1505 ms.
- **Runtime export:** TorchScript EfficientAD tại
  `artifacts/runtime/H01/efficientad/efficientad.pt`; parity 20/20, max Δ = 0.0
  (Δ < 1e-3, decision parity 100%). YOLO ONNX/OpenVINO fallback PyTorch float32
  trong venv CPU hiện tại (thiếu module `onnx`/`openvino`).
- **Khuyến nghị:** mô hình demo mặc định (`recommended_demo_model` trong frozen file).

## 6. H01_sec — điểm vận hành phụ (FRR ≤ 1%)

- **Kiến trúc/luật:** giống H01, chỉ đổi ngưỡng YOLO.
- **Ngưỡng:** `τ_yolo=0.48`, `τ_ad=1.4598`.
- **Kết quả test:** BalAcc 97.27%, recall 99.55% (1 FN), precision 95.22%,
  FRR 5.00%, ECPB 0.0540; route 47.7%; latency mean ~1423 ms.
- **Vai trò:** operating point phụ cho scenario chi phí báo nhầm cao; 1 miss duy nhất
  phải review thủ công trước khi dùng.

## 7. H02 — YOLO11s → EfficientAD-S, cascade (phụ)

- **Ngưỡng:** `τ_yolo=0.38`, `τ_ad=1.4598`.
- **Kết quả test:** BalAcc 82.50%, recall 100.00%, precision 74.07%, FRR 35.00%,
  ECPB 0.3465; route 32.5%; latency mean ~1054 ms.
- **Vai trò:** đổi detector trong hybrid; thua H01 toàn diện tại điểm khóa.

---

## 8. Bảng tổng hợp ngưỡng và checkpoint

| ID | Checkpoint SHA-256 (rút gọn) | τ_yolo | τ_ad | Test BalAcc | Test recall | Test FRR |
|----|------------------------------|-------:|-----:|------------:|------------:|---------:|
| B01 | `c0cfe917…` | 0.14 | — | 95.00% | 100.00% | 10.00% |
| B02 | `aa079248…` | 0.38 | — | 82.50% | 100.00% | 35.00% |
| B03 | `9453bc2c…` | — | 1.4598 | 68.18% | 36.36% | 0.00% |
| H00 | B01 + B03 | 0.14 | 1.4598 | 95.00% | 100.00% | 10.00% |
| H01 | B01 + B03 | 0.14 | 1.4598 | 95.00% | 100.00% | 10.00% |
| H02 | B02 + B03 | 0.38 | 1.4598 | 82.50% | 100.00% | 35.00% |
| H01_sec | B01 + B03 | 0.48 | 1.4598 | 97.27% | 99.55% | 5.00% |

Nguồn số liệu: `artifacts/test/benchmark_summary.csv`,
`artifacts/test/frozen_evaluation.json`, `artifacts/fusion/thresholds.json`.
