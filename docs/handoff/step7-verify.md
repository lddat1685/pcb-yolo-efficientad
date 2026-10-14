# Báo cáo Kiểm định Step 7 = Bước 7 của PLAN (Khóa cấu hình và đánh giá test)

Quy ước làm việc: `docs/handoff/WORKFLOW.md`. Tài liệu căn cứ: `PLAN.md` §5 (Bước 7), §6, §7 và `docs/handoff/step7-contract.md`.

## 1. Kết luận Kiểm định (Final Verdict)

**KẾT LUẬN: PASS**
- **Unit & Contract Suite:** 10/10 test PASS (100%).
- **Independent Stdlib Oracle:** 126/126 ô số liệu đối chiếu khớp hoàn hảo (sai số < $10^{-6}$).
- **Adversarial / Sabotage Suite:** 20/20 ca phá hoại bị chặn thành công (100%).
- **Bảng phát hiện:** 0 Critical, 0 Major, 0 Minor.

---

## 2. Kết quả Xác minh Độc lập (Independent Verification Findings)

### 2.1. Đóng băng Cấu hình (Frozen Evaluation Integrity)
- Tệp `artifacts/test/frozen_evaluation.json` đã được sinh ra trước khi chạy dữ liệu test, lưu trữ đầy đủ checksum SHA-256 của các checkpoint:
  - B01 (YOLO11n): `c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05`
  - B02 (YOLO11s): `aa079248bd566b3adf8863e3b43dde93d3392307a3ef17107e4c126bf410936c`
  - B03 (EfficientAD-S): `9453bc2c740e81dfb8949df929ed001d60ebf8e5b0f6a18c95eddd9b3a2814a0`
  - Split SHA-256 (440 sample IDs test sorted): `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24`
- Cơ chế bảo vệ: Toàn bộ module cũ (`pcb_lab.models.yolo.extract`, `pcb_lab.models.efficientad.extract`) và các hàm routing mặc định (`predict_cascade`, `predict_or`, `validate_sample_alignment`) đều ném `PermissionError` khi phát hiện split `test` nếu không có cờ `allow_test=True` được ủy quyền qua cấu hình đóng băng.

### 2.2. Kiểm tra Phân bổ Dữ liệu Test Canonical
Đã kiểm tra độc lập trên `../DatasetVer4_Public/benchmarks/deeppcb/splits/holdout.json` và `samples.jsonl`:
- Tổng số mẫu: đúng 440 ảnh (220 good, 220 defect).
- Không có mẫu trùng lặp ID, không có rò rỉ metadata.
- Phân bổ theo 2 nhóm nguồn:
  - `group44000`: 100 good, 100 defect (tổng 200 ảnh).
  - `group92000`: 120 good, 120 defect (tổng 240 ảnh).

### 2.3. Bảng Đối chứng Số liệu Thực tế qua Oracle Độc lập (Python stdlib)

| Model | Balanced Acc | Defect Recall | Defect Precision | FRR | Miss Rate | F1 | MCC | ECPB | Mean Latency | p50 Latency | Route Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B01** | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 120.2 ms | 120.5 ms | N/A |
| **B02** | 82.50% | 100.00% | 74.07% | 35.00% | 0.00% | 0.8511 | 0.6939 | 0.3465 | 165.6 ms | 161.9 ms | N/A |
| **B03** | 68.18% | 36.36% | 100.00% | 0.00% | 63.64% | 0.5333 | 0.4714 | 0.6364 | 2733.2 ms | 2723.1 ms | 100.0% |
| **H00** | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 2853.4 ms | 2841.2 ms | 100.0% |
| **H01** | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 1348.8 ms | 132.3 ms | 45.0% |
| **H02** | 82.50% | 100.00% | 74.07% | 35.00% | 0.00% | 0.8511 | 0.6939 | 0.3465 | 1054.1 ms | 167.8 ms | 32.5% |
| **H01_sec** | 97.27% | 99.55% | 95.22% | 5.00% | 0.45% | 0.9733 | 0.9464 | 0.0540 | 1423.3 ms | 144.2 ms | 47.7% |

*Xác nhận: Tất cả 56 ô số liệu trong bảng summary và 70 ô trong bảng per-group khớp tuyệt đối với tính toán độc lập từ script `scripts/verify_step7_oracle.py`.*

### 2.4. Phân tích Giá trị Định tuyến H01 vs H00
- **Chất lượng phát hiện:** $\Delta\text{recall} = 0.00\%$, $\Delta\text{FRR} = 0.00\%$ tại operating point chính ($\tau_{\text{yolo}}=0.14$).
- **Lợi ích Early-Exit:** Tiết kiệm **1504.6 ms** thời gian suy luận trung bình mỗi ảnh (giảm từ 2853.4 ms xuống 1348.8 ms, tương đương tiết kiệm 52.7% tài nguyên tính toán). Độ trễ p50 của Cascade là 132.3 ms (giảm 21.5 lần so với OR rule).
- **Điểm vận hành phụ H01_sec:** Đạt hiệu quả tối ưu chi phí: FRR giảm một nửa xuống 5.00% (chỉ 11 ảnh báo nhầm), Balanced Accuracy 97.27%, ECPB 0.0540.

### 2.5. Phân tích Phân rã Nhóm Nguồn & 6 Lớp Khuyết tật
- **Nhóm nguồn:**
  - `group92000` cho kết quả rất tốt: Recall 100% (H01) và 99.17% (H01_sec), FRR chỉ 5.0% (H01) và 1.67% (H01_sec).
  - `group44000` có FRR cao hơn (16% ở H01, 9% ở H01_sec), phản ánh đúng sự biến thiên phân phối (domain shift) giữa các thiết kế bo mạch trần.
- **6 Lớp lỗi:** Kiểm tra 1.407 GT boxes trên test set:
  - `open_circuit`: 298 boxes (B01 AP50 0.8175)
  - `short`: 215 boxes (B01 AP50 0.8029)
  - `mouse_bite`: 250 boxes (B01 AP50 0.7529)
  - `spur`: 228 boxes (B01 AP50 0.7358)
  - `spurious_copper`: 194 boxes (B01 AP50 0.8806)
  - `pin_hole`: 222 boxes (B01 AP50 0.7680)

---

## 3. Nhật ký Phá hoại (Adversarial / Sabotage Suite Log)

Đã chạy `scripts/verify_step7_sabotage.py` với 20 ca tấn công giả định:
- **S01:** Thiếu file `frozen_evaluation.json` $\to$ Chặn bằng `FileNotFoundError` (PASS).
- **S02:** File cấu hình có `test_access_authorized: False` $\to$ Chặn bằng `PermissionError` (PASS).
- **S03:** Checkpoint B01 bị thiếu $\to$ Chặn bằng `FileNotFoundError` (PASS).
- **S04:** Checkpoint B02 bị thiếu $\to$ Chặn bằng `FileNotFoundError` (PASS).
- **S05:** Checkpoint B03 bị thiếu $\to$ Chặn bằng `FileNotFoundError` (PASS).
- **S06:** Gọi `yolo_extract` trên partition `test` $\to$ Chặn bằng `PermissionError` (PASS).
- **S07:** Gọi `ad_extract` trên partition `test` $\to$ Chặn bằng `PermissionError` (PASS).
- **S08:** Gọi `validate_sample_alignment` với split `test` không ủy quyền $\to$ Chặn bằng `PermissionError` (PASS).
- **S09:** Gọi `predict_cascade` với split `test` không ủy quyền $\to$ Chặn bằng `PermissionError` (PASS).
- **S10:** Gọi `predict_or` với split `test` không ủy quyền $\to$ Chặn bằng `PermissionError` (PASS).
- **S11:** Lệch `sample_id` trong `predict_cascade` $\to$ Chặn bằng `ValueError` (PASS).
- **S12:** Lệch `sample_id` trong `predict_or` $\to$ Chặn bằng `ValueError` (PASS).
- **S13:** Lặp `sample_id` trong `validate_sample_alignment` $\to$ Chặn bằng `ValueError` (PASS).
- **S14:** Tập ID không song ánh giữa 2 mô hình $\to$ Chặn bằng `ValueError` (PASS).
- **S15:** Giả mạo danh sách mẫu test (< 440) $\to$ Chặn bằng `ValueError` (PASS).
- **S16:** Giả mạo danh sách mẫu test (> 440) $\to$ Chặn bằng `ValueError` (PASS).
- **S17:** Thiếu file `holdout.json` $\to$ Chặn bằng `FileNotFoundError` (PASS).
- **S18:** Lỗi suy luận YOLO được ghi nhận $\to$ Chặn bằng `ValueError` (PASS).
- **S19:** Lỗi suy luận AD trong Cascade được ghi nhận $\to$ Chặn bằng `ValueError` (PASS).
- **S20:** Lỗi suy luận AD trong OR được ghi nhận $\to$ Chặn bằng `ValueError` (PASS).

Kết quả: **20/20 ca phá hoại bị chặn thành công** (Chi tiết tại `docs/handoff/step7-sabotage-log.json`).

---

## 4. Bảng Phát hiện (Findings Table)

| Mức độ | Vị trí (File:Dòng) | Hiện tượng | Đề xuất xử lý | Trạng thái |
|---|---|---|---|---|
| - | Không có | Không phát hiện sai sót nào vi phạm hợp đồng | Chuyển sang Bước 8 | **PASS** |

---

## 5. Khuyến nghị Nghiệm thu
Bước 7 đã hoàn thành xuất sắc và đầy đủ mọi yêu cầu khoa học:
1. Đã khóa cấu hình trước khi mở tập test.
2. Đo latency thực tế phản ánh chính xác hiệu quả định tuyến early-exit.
3. Số liệu trung thực, minh bạch, được đối chứng độc lập 100% bằng stdlib.
4. Đủ 8 file artifact chuẩn và báo cáo chi tiết.
Chấp thuận nghiệm thu và sẵn sàng chuyển giao sang **Bước 8 (Xây giao diện kiểm thử Streamlit)**.
