# Báo cáo Kiểm định Step 8 = Bước 8 của PLAN (Giao diện Streamlit & Bộ khám phá Benchmark)

Quy ước làm việc: `docs/handoff/WORKFLOW.md`. Tài liệu căn cứ: `PLAN.md` §5 (Bước 8), §6, §7 và `docs/handoff/step8-contract.md`.

## 1. Kết luận Kiểm định (Final Verdict)

**KẾT LUẬN: PASS**
- **Unit & Spec-First Test Suite:** 24/24 test PASS (5 spec-first tests + 19 UI/service unit tests).
- **CLI vs UI Parity Suite:** 8/8 cấu hình mô hình đối chiếu (B01, B02, B03, H00, H01, H02, H01_sec trên cả ảnh defect và good) với 15 trường dữ liệu khớp tuyệt đối 100%.
- **Adversarial / Sabotage Suite:** 20/20 ca phá hoại bị chặn thành công (100%).
- **Bảng phát hiện:** 0 Critical, 0 Major, 0 Minor.

---

## 2. Kết quả Xác minh Độc lập (Independent Verification Findings)

### 2.1. Khung Suy Luận Thống Nhất (Unified InferenceService)
- Cả CLI (`scripts/inspect_image.py`) và UI Streamlit (`src/pcb_lab/ui/`) đều chỉ gọi duy nhất một lõi thực thi là `pcb_lab.inference.service.InferenceService`.
- Tránh hoàn toàn việc phân kỳ logic tính điểm, ngưỡng, hoặc vẽ bounding box giữa môi trường dòng lệnh và đồ họa.
- Cơ chế thực thi có điều kiện chuẩn xác:
  - Với **Cascade (H01/H02)**: Khi YOLO phát hiện khuyết tật ($\ge \tau_{\text{yolo}}$), hệ thống ngắt sớm (early-exit), ghi nhận `yolo_executed=True, efficientad_executed=False, anomaly_score=None`, tiết kiệm tài nguyên.
  - Với **OR (H00)**: Luôn chạy cả hai mô hình YOLO và EfficientAD bất kể kết quả của mô hình đầu.

### 2.2. Kiểm Tra 3 Màn Hình Streamlit (Headless AppTest)
Bộ kiểm thử tự động `streamlit.testing.v1.AppTest` đã kiểm tra luồng render không cần mở trình duyệt thật:
1. **Single Inspection (Kiểm định đơn ảnh):**
   - Hỗ trợ nguồn ảnh: Upload (PNG/JPEG), Thư mục cục bộ, Gallery (mẫu test/fusion đã khóa), URL HTTP(S).
   - Hiển thị bảng điều khiển 2 cột: Ảnh gốc có bounding box YOLO (6 màu đặc trưng theo 6 lớp khuyết tật) và Bản đồ nhiệt dị thường EfficientAD (colormap `jet` hoặc `inferno`, thanh trượt độ mờ $\alpha \in [0, 1]$).
   - Bảng phân rã thời gian: Giải mã ảnh, tải mô hình, suy luận YOLO, suy luận EfficientAD, tổng thời gian xử lý.
2. **Benchmark Explorer (Khám phá số liệu chuẩn):**
   - Đọc trực tiếp từ các tệp báo cáo đóng băng tại `artifacts/test/`: `benchmark_summary.csv`, `benchmark_per_group.csv`, `benchmark_per_class.csv`.
   - Tính toán và hiển thị đồ thị Pareto Frontier (Trade-off giữa Recall và FRR / Latency).
   - So sánh trực quan đối đầu giữa H00 (OR rule) và H01 (Cascade).
3. **Batch Inspection (Kiểm định lô mẫu):**
   - Hỗ trợ phân tích lô mẫu giới hạn từ 1 đến 20 ảnh với thanh tiến trình tương tác (`st.progress`).
   - Tổng hợp số lượng ảnh Defect/Good/Error, bảng thống kê thời gian trung bình và danh sách chi tiết từng mẫu.

### 2.3. Quy Tắc "Zero-Fake-Accuracy" Trên Ảnh Người Dùng Tải Lên
- Kiểm định nghiêm ngặt: Các nguồn ảnh bên ngoài (Upload, URL, File không thuộc benchmark) được gán `ground_truth_available=False`.
- Giao diện UI tuyệt đối không hiển thị Accuracy, F1, MCC, hay nhãn Ground Truth giả định cho ảnh tải lên; chỉ hiển thị nhãn dự đoán (`DEFECT` / `GOOD`), lý do định tuyến (`yolo_hit`, `ad_hit`, `both_hit`, `all_pass`), và điểm tin cậy.

### 2.4. Quản Lý Tài Nguyên và Cache (@st.cache_resource & RLock)
- Mô hình được cache qua `@st.cache_resource` với khóa fingerprint bao gồm: Checkpoint SHA-256, kích thước file, thời gian sửa đổi (mtime_ns), và thiết bị (`device="cpu"`).
- Khối `RLock` bảo vệ truy cập suy luận đa luồng trong trường hợp nhiều tác vụ đồng thời truy cập mô hình trong bộ nhớ.
- Khi tệp checkpoint hoặc cấu hình `frozen_evaluation.json` bị can thiệp, khóa fingerprint thay đổi và cache tự động bị vô hiệu hóa hoặc từ chối nạp mô hình giả mạo.

### 2.5. Kết Quả Đối Chiếu CLI vs UI Parity
Chạy `scripts/verify_step8_parity.py` trên 8 trường hợp điển hình:
- **B01** (Defect): Quyết định DEFECT, 20 boxes, khớp 15/15 trường dữ liệu.
- **B02** (Defect): Quyết định DEFECT, 21 boxes, khớp 15/15 trường dữ liệu.
- **B03** (Defect): Quyết định GOOD ($AD_{\text{score}} < \tau_{\text{ad}}$), khớp 15/15 trường dữ liệu.
- **H00** (Defect): Quyết định DEFECT, chạy cả 2 mô hình (`both_hit`), khớp 15/15 trường dữ liệu.
- **H01** (Defect): Quyết định DEFECT, ngắt sớm (`early_exit=True`, AD không chạy), khớp 15/15 trường dữ liệu.
- **H02** (Defect): Quyết định DEFECT, ngắt sớm, khớp 15/15 trường dữ liệu.
- **H01_sec** (Defect): Quyết định DEFECT ($\tau=0.48$), 16 boxes, khớp 15/15 trường dữ liệu.
- **H01** (Good): Quyết định DEFECT (báo nhầm $\tau=0.14$), ngắt sớm, khớp 15/15 trường dữ liệu.

**Kết luận Parity:** CLI và UI trả về cùng kết quả logic, cùng điểm số, cùng box, cùng hash của bản đồ nhiệt.

---

## 3. Nhật ký Phá hoại (Adversarial / Sabotage Suite Log)

Đã chạy `scripts/verify_step8_sabotage.py` với 20 ca tấn công giả định:
- **S01:** `validate_pixels` từ chối mảng 1 chiều $\to$ Bắt bằng `ValueError` (PASS).
- **S02:** `validate_pixels` từ chối mảng `float32` $\to$ Bắt bằng `ValueError` (PASS).
- **S03:** `validate_pixels` từ chối mảng 5 kênh màu $\to$ Bắt bằng `ValueError` (PASS).
- **S04:** `validate_pixels` từ chối ảnh rỗng kích thước (0, 0, 3) $\to$ Bắt bằng `ValueError` (PASS).
- **S05:** `validate_pixels` từ chối mảng chứa `NaN` $\to$ Bắt bằng `ValueError` (PASS).
- **S06:** `validate_pixels` từ chối mảng 5 chiều $\to$ Bắt bằng `ValueError` (PASS).
- **S07:** `validate_pixels` từ chối ảnh có cạnh $> 4096$ px $\to$ Bắt bằng `ValueError` (PASS).
- **S08:** `validate_pixels` từ chối ảnh có tổng số pixel $> 4.000.000$ $\to$ Bắt bằng `ValueError` (PASS).
- **S09:** `ModelRegistry` từ chối mã mô hình không có trong danh mục ('YOLO26') $\to$ Bắt bằng `ValueError` (PASS).
- **S10:** `ModelRegistry` từ chối mã mô hình lạ ('UNKNOWN_MODEL_99') $\to$ Bắt bằng `ValueError` (PASS).
- **S11:** `load_model` từ chối checkpoint có SHA-256 không khớp cấu hình đã đóng băng $\to$ Bắt bằng `ValueError` (PASS).
- **S12:** `load_model` từ chối đường dẫn checkpoint không tồn tại $\to$ Bắt bằng `FileNotFoundError` (PASS).
- **S13:** `FileImageSource` từ chối tệp không tồn tại $\to$ Bắt bằng `FileNotFoundError` (PASS).
- **S14:** `decode_image` từ chối chuỗi byte không phải định dạng ảnh $\to$ Bắt bằng `UnidentifiedImageError` (PASS).
- **S15:** `decode_image` từ chối chuỗi byte rỗng hoặc $> 20$ MiB $\to$ Bắt bằng `ValueError` (PASS).
- **S16:** `InferenceService` bắt lỗi mô hình gặp sự cố (cố tình throw runtime error), chuyển trạng thái sang `ERROR` mà không làm crash ứng dụng (PASS).
- **S17:** `load_benchmark` từ chối khi thiếu thư mục hoặc tệp artifact chuẩn $\to$ Bắt bằng `FileNotFoundError` (PASS).
- **S18:** `load_benchmark` từ chối bảng summary thiếu các cột bắt buộc $\to$ Bắt bằng `ValueError` (PASS).
- **S19:** `ModelRegistry.gallery` chặn truy cập tập `test` khi `report_mode=False` $\to$ Bắt bằng `PermissionError` (PASS).
- **S20:** `URLImageSource` từ chối các giao thức không an toàn (không phải HTTP/HTTPS hoặc chứa credentials) $\to$ Bắt bằng `ValueError` (PASS).

Kết quả: **20/20 ca phá hoại bị chặn thành công** (Chi tiết tại `docs/handoff/step8-sabotage-log.json`).

---

## 4. Bảng Phát hiện (Findings Table)

| Mức độ | Vị trí (File:Dòng) | Hiện tượng | Đề xuất xử lý | Trạng thái |
|---|---|---|---|---|
| - | Không có | Không phát hiện sai sót nào vi phạm hợp đồng | Nghiệm thu Bước 8, sẵn sàng chuẩn bị Bước 9 / Bước 10 | **PASS** |

---

## 5. Hướng dẫn Khởi chạy Ứng dụng Thực tế

Người dùng có thể khởi chạy giao diện kiểm định trực quan bằng lệnh:
```powershell
& ".cache/step3-verifier-b/.venv/Scripts/python.exe" scripts/run_ui.py
```
Sau đó truy cập trình duyệt tại địa chỉ: `http://127.0.0.1:8501`.
