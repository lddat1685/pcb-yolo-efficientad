# Báo cáo Đánh giá Test Cuối cùng (Step 7: Final Test Evaluation Report)

Ngày lập: 29/09/2026 12:24:23. Trạng thái: **Đã hoàn tất kiểm chứng test**.

> Đính chính kiểm toán 30/09/2026: bảng số liệu gốc được giữ để truy xuất lịch sử.
> Latency dưới đây là **ước tính từ số đo từng nhánh**, chưa gồm decode và chưa đo cascade E2E thực.
> Raw boxes của lần chạy gốc không được lưu; AP@0.5 dùng nội suy 11 điểm, không tương đương AP101 của bản sửa.
> Evaluator sửa lỗi xuất lượt chạy mới vào thư mục riêng và lưu pipeline thực, raw detections, AP50/AP50:95.

## 1. Xác thực Khóa Cấu hình (Frozen Evaluation Verification)
- **Split:** `test` (440 ảnh: 220 good, 220 defect).
- **SHA-256 tập test:** `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24`
- **Checkpoint B01 (YOLO11n):** `c0cfe917a685a9d2...`
- **Checkpoint B02 (YOLO11s):** `aa079248bd566b3a...`
- **Checkpoint B03 (EfficientAD-S):** `9453bc2c740e81df...`
- **Mô hình đề xuất cho demo:** `H01_cascade_yolo11n_efficientad_s`

## 2. Bảng Tổng hợp Kết quả Test ở Mức Ảnh (PLAN §7.2)

| Model | Bal. Acc | Recall | Precision | FRR | Miss Rate | F1 | MCC | ECPB | Mean Latency (ms) | p95 Latency (ms) | Route % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B01** | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 120.2 | 140.4 | N/A |
| **B02** | 82.50% | 100.00% | 74.07% | 35.00% | 0.00% | 0.8511 | 0.6939 | 0.3465 | 165.6 | 194.3 | N/A |
| **B03** | 68.18% | 36.36% | 100.00% | 0.00% | 63.64% | 0.5333 | 0.4714 | 0.6364 | 2733.2 | 2850.6 | 100.0% |
| **H00** | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 2853.4 | 2978.0 | 100.0% |
| **H01** | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 1348.8 | 2938.4 | 45.0% |
| **H02** | 82.50% | 100.00% | 74.07% | 35.00% | 0.00% | 0.8511 | 0.6939 | 0.3465 | 1054.1 | 2970.4 | 32.5% |
| **H01_sec** | 97.27% | 99.55% | 95.22% | 5.00% | 0.45% | 0.9733 | 0.9464 | 0.0540 | 1423.3 | 2952.8 | 47.7% |

## 3. Phân tích Routing và Hiệu quả Định tuyến (H01 vs H00)
- **Δ Recall (H01 - H00):** +0.00%
- **Δ FRR (H01 - H00):** +0.00%
- **Δ Latency Mean (H01 - H00):** -1504.6 ms
- **Số lượng mẫu Early-Exit:** 242 / 440 ảnh (55.0%)
- **Khuyết tật được cứu (Rescued Defects):** 0
- **Số Good bị báo nhầm tăng thêm (Added False Rejects):** 0

## 4. Phân tích Phân rã theo Nhóm Nguồn (Source Group Breakdown)

| Model | Source Group | Samples | Good | Defect | Recall | FRR | Bal. Acc | Worst-Group Recall | Worst-Group FRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B01 | group44000 | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 16.00% |
| B01 | group92000 | 240 | 120 | 120 | 100.00% | 5.00% | 97.50% | 100.00% | 16.00% |
| B02 | group44000 | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 50.83% |
| B02 | group92000 | 240 | 120 | 120 | 100.00% | 50.83% | 74.58% | 100.00% | 50.83% |
| B03 | group44000 | 200 | 100 | 100 | 40.00% | 0.00% | 70.00% | 33.33% | 0.00% |
| B03 | group92000 | 240 | 120 | 120 | 33.33% | 0.00% | 66.67% | 33.33% | 0.00% |
| H00 | group44000 | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 16.00% |
| H00 | group92000 | 240 | 120 | 120 | 100.00% | 5.00% | 97.50% | 100.00% | 16.00% |
| H01 | group44000 | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 16.00% |
| H01 | group92000 | 240 | 120 | 120 | 100.00% | 5.00% | 97.50% | 100.00% | 16.00% |
| H02 | group44000 | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 50.83% |
| H02 | group92000 | 240 | 120 | 120 | 100.00% | 50.83% | 74.58% | 100.00% | 50.83% |
| H01_sec | group44000 | 200 | 100 | 100 | 100.00% | 9.00% | 95.50% | 99.17% | 9.00% |
| H01_sec | group92000 | 240 | 120 | 120 | 99.17% | 1.67% | 98.75% | 99.17% | 9.00% |

## 5. Phân tích Phát hiện 6 Lớp Khuyết tật (Per-Class Detection)

| Model | Class Name | GT Boxes | Detected Boxes | AP@0.5 | Recall |
|---|---|---:|---:|---:|---:|
| B01 | `open_circuit` | 298 | 406 | 0.8175 | 87.58% |
| B01 | `short` | 215 | 237 | 0.8029 | 79.07% |
| B01 | `mouse_bite` | 250 | 287 | 0.7529 | 76.80% |
| B01 | `spur` | 228 | 217 | 0.7358 | 71.49% |
| B01 | `spurious_copper` | 194 | 225 | 0.8806 | 90.72% |
| B01 | `pin_hole` | 222 | 415 | 0.7680 | 90.54% |
| B02 | `open_circuit` | 298 | 345 | 0.8764 | 92.28% |
| B02 | `short` | 215 | 195 | 0.8740 | 81.86% |
| B02 | `mouse_bite` | 250 | 242 | 0.8465 | 82.80% |
| B02 | `spur` | 228 | 190 | 0.7578 | 71.49% |
| B02 | `spurious_copper` | 194 | 169 | 0.8974 | 84.54% |
| B02 | `pin_hole` | 222 | 647 | 0.7825 | 97.30% |
| H01 | `open_circuit` | 298 | 406 | 0.8175 | 87.58% |
| H01 | `short` | 215 | 237 | 0.8029 | 79.07% |
| H01 | `mouse_bite` | 250 | 287 | 0.7529 | 76.80% |
| H01 | `spur` | 228 | 217 | 0.7358 | 71.49% |
| H01 | `spurious_copper` | 194 | 225 | 0.8806 | 90.72% |
| H01 | `pin_hole` | 222 | 415 | 0.7680 | 90.54% |

## 6. Kết luận và Khuyến nghị Triển khai

- Trên test canonical, **H01 không cứu thêm ảnh lỗi nào so với YOLO11n**: YOLO11n đã đạt 100% recall tại τ_yolo=0.14 (220/220 lỗi), nên `rescued_defects=0`; rescue rate là N/A.
- B01, H00 và H01 có cùng quyết định trên 440 ảnh, FRR 10% và ECPB 0.099. H01 không cải thiện khả năng phát hiện ở điểm khóa; chưa có cơ sở gọi H01 là cấu hình tối ưu hoặc khẳng định AD sẽ cứu lỗi tương lai.
- Giá trị của cơ chế H01 so với **H00 OR chạy cả hai nhánh** là giảm tính toán nhờ ngắt sớm trên 242/440 ảnh. Số liệu lịch sử cho **tiết kiệm 52.7% độ trễ mean và p50 nhanh hơn 21.5 lần**. Đây là tỷ lệ từ latency cộng nhánh cũ, **chưa phải mức tiết kiệm E2E đã xác minh**; cần đo lại cùng 440 ảnh, warmup và 3 lượt timing.
- So với YOLO11n đơn, H01 thêm chi phí AD mà không thêm TP trong thí nghiệm này. H01_sec tại τ=0.48 cũng trùng YOLO-only cùng ngưỡng, không phải bằng chứng AD cứu lỗi.
- H01 là lựa chọn demo đã đăng ký, không phải lựa chọn mới dựa trên test. Mọi thí nghiệm bổ sung sau khi mở test được ghi là khám phá trên dữ liệu phát triển và không đổi ngưỡng frozen.
