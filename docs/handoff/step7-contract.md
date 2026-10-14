# Hợp đồng Step 7 = Bước 7 của PLAN (Khóa cấu hình và đánh giá test)

Quy tắc chung: `docs/handoff/WORKFLOW.md`. Mâu thuẫn với `PLAN.md` thì plan thắng.

## 1. Phạm vi (Scope)
- **Có:**
  - Lưu tệp khóa cấu hình `artifacts/test/frozen_evaluation.json` **TRƯỚC KHI** thực thi đánh giá dữ liệu test:
    - Checkpoint hashes: B01 (`best.pt`), B02 (`best.pt`), B03 (`model.pt`).
    - Ngưỡng quyết định đã khóa từ Step 5 và Step 6 shortlist:
      - B01: $\tau_{\text{yolo}} = 0.14$
      - B02: $\tau_{\text{yolo}} = 0.38$
      - B03: $\tau_{\text{ad}} = 1.4598$
      - H00: $\tau_{\text{yolo}} = 0.14, \tau_{\text{ad}} = 1.4598$ (OR rule)
      - H01: $\tau_{\text{yolo}} = 0.14, \tau_{\text{ad}} = 1.4598$ (Cascade chính, $\text{FRR} \le 5\%$)
      - H02: $\tau_{\text{yolo}} = 0.38, \tau_{\text{ad}} = 1.4598$ (Cascade phụ YOLO11s)
      - Điểm vận hành phụ: H01 tại $\text{FRR} \le 1\%$ ($\tau_{\text{yolo}} = 0.48, \tau_{\text{ad}} = 1.4598$).
    - Checksum tập `test` (440 sample IDs từ `holdout.json`, SHA-256 danh sách).
    - Cấu hình tiền xử lý, runtime/device backend (CPU/CUDA), random seed (42).
    - Cấu hình đề xuất cho demo: `H01_cascade_yolo11n_efficientad_s` (chọn từ development selection ở Step 6 shortlist).
    - Cấp quyền `test_access: true` DUY NHẤT trong pipeline đánh giá test sau khi đã khóa cấu hình.
  - Xây dựng module đánh giá test: `src/pcb_lab/evaluation/test_eval.py` và CLI `scripts/evaluate_test.py`.
  - Thực thi suy luận pipeline thực tế trên toàn bộ **440 ảnh test canonical** (220 good, 220 defect) từ `../DatasetVer4_Public`:
    - Đo độ trễ thực tế E2E (batch size 1, warmup 30 lượt, dùng `time.perf_counter()`):
      - Ghi nhận mean, median (p50), 95th percentile (p95), min, max cho mỗi mô hình.
      - H00 (OR): thực thi đồng thời cả YOLO và EfficientAD trên 100% ảnh.
      - H01/H02 (Cascade): thực thi early-exit (khi YOLO phát hiện lỗi $\ge \tau_{\text{yolo}}$, bỏ qua EfficientAD; khi YOLO âm tính, định tuyến sang EfficientAD).
  - Tính toán và tổng hợp đầy đủ các chỉ số:
    - **Cấp ảnh tổng thể (Overall Image-level):** Balanced Accuracy, Defect Recall (TPR), Defect Precision, FRR, Defect→GOOD (miss rate), F1, MCC, ECPB ($N=10.000, 1\%\text{ defect}, C_{\text{FN}}=100, C_{\text{FP}}=1$).
    - **Routing & Hybrid metrics:** `route_rate_total`, `route_rate_good`, `route_rate_defect`, `rescued_defects`, `rescue_rate`, `added_false_rejects`, `unclassified_positive_rate`.
    - **Đối chứng H01 vs H00:** $\Delta\text{recall}$, $\Delta\text{FRR}$, $\Delta\text{latency}$ để chứng minh giá trị thực tế của early-exit định tuyến.
    - **Phân rã theo nhóm nguồn (Source Group Breakdown):**
      - `group44000`: 200 ảnh (100 good, 100 defect).
      - `group92000`: 240 ảnh (120 good, 120 defect).
      - Báo cáo `worst-group recall` và `worst-group FRR` theo PLAN §1.4 và §7.6.
    - **Phân rã phát hiện 6 lớp khuyết tật (Per-class Detection Breakdown):**
      - 6 lớp: `open_circuit`, `short`, `mouse_bite`, `spur`, `spurious_copper`, `pin_hole`.
      - Đánh giá mAP@0.5, mAP@0.5:0.95 và box recall tại ngưỡng vận hành cho các detector.
  - Xuất bộ artifacts chuẩn tại `artifacts/test/` và báo cáo tại `reports/`:
    - `artifacts/test/frozen_evaluation.json`
    - `artifacts/test/predictions_test.jsonl`
    - `artifacts/test/benchmark_summary.json` (và `.csv`)
    - `artifacts/test/benchmark_per_group.json` (và `.csv`)
    - `artifacts/test/benchmark_per_class.json` (và `.csv`)
    - `reports/step7_final_test_evaluation_report.md`

- **Không:**
  - **TUYỆT ĐỐI CẤM** huấn luyện lại (retraining) hoặc tinh chỉnh lại ngưỡng (re-tuning) sau khi xem tập test. Nếu kết quả không đạt, báo cáo trung thực không đạt (PLAN §5 Bước 7).
  - Không dùng cache score có sẵn thay cho việc chạy pipeline inference thực để đo latency.
  - Không sửa đổi hay ghi đè các artifacts đã khóa của Step 3, Step 4, Step 5, Step 6.
  - Không đánh giá ngoài 440 ảnh test canonical của split `holdout.json`.

## 2. Giao diện Module (`src/pcb_lab/evaluation/test_eval.py`)

### `freeze_evaluation_config(...) -> Path`
- Tạo và lưu `artifacts/test/frozen_evaluation.json`.
- Ghi nhận đầy đủ thông tin:
  - Checkpoint paths & sha256 checksums (B01, B02, B03).
  - Ngưỡng quyết định đã khóa từ Step 5 / Step 6 shortlist.
  - Checksum danh sách mẫu `test` (440 ảnh).
  - Metadata runtime: device, torch version, batch size, seed.
  - Đánh dấu `test_access: true` có chữ ký khóa cấu hình.

### `run_test_evaluation(...) -> TestEvaluationResult`
- Nhận vào `frozen_config_path` (đã được khóa trước đó).
- Nạp 440 ảnh test từ manifest dataset `../DatasetVer4_Public`.
- Thực hiện inference đo đạc thời gian thật (batch size 1, warmup 30 lượt).
- Tính toán đầy đủ:
  - Bảng tổng hợp image-level metrics cho B01, B02, B03, H00, H01, H02 (và secondary H01).
  - Bảng per-source-group metrics (`group44000`, `group92000`) và xác định worst-group.
  - Bảng per-class detection metrics cho các mô hình YOLO.
  - Danh sách bản ghi dự đoán chi tiết từng ảnh `predictions_test.jsonl`.
- Xuất toàn bộ file CSV, JSON, JSONL và trả về đối tượng `TestEvaluationResult`.

## 3. CLI Thực thi (`scripts/evaluate_test.py`)
- Cung cấp CLI chạy độc lập:
  ```powershell
  python scripts/evaluate_test.py --freeze-only
  python scripts/evaluate_test.py --run --frozen-config artifacts/test/frozen_evaluation.json
  ```
- Hoặc chạy cả hai bước tuần tự:
  ```powershell
  python scripts/evaluate_test.py --all
  ```

## 4. Cấu trúc Artifacts Xuất ra (`artifacts/test/`)

### `artifacts/test/frozen_evaluation.json`
```json
{
  "schema_version": 1,
  "status": "FROZEN",
  "created_at": "...",
  "test_access_authorized": true,
  "split_name": "test",
  "sample_count": 440,
  "test_split_sha256": "...",
  "checkpoints": {
    "B01_yolo11n": { "path": "...", "sha256": "..." },
    "B02_yolo11s": { "path": "...", "sha256": "..." },
    "B03_efficientad_s": { "path": "...", "sha256": "..." }
  },
  "operating_points": {
    "B01": { "tau_yolo": 0.14, "tau_ad": null },
    "B02": { "tau_yolo": 0.38, "tau_ad": null },
    "B03": { "tau_yolo": null, "tau_ad": 1.4598 },
    "H00": { "tau_yolo": 0.14, "tau_ad": 1.4598 },
    "H01": { "tau_yolo": 0.14, "tau_ad": 1.4598 },
    "H02": { "tau_yolo": 0.38, "tau_ad": 1.4598 },
    "H01_sec": { "tau_yolo": 0.48, "tau_ad": 1.4598 }
  },
  "recommended_demo_model": "H01_cascade_yolo11n_efficientad_s"
}
```

### `artifacts/test/benchmark_summary.json` (và `.csv`)
Chứa các cột bắt buộc theo PLAN §7.2:
- `model_id`, `balanced_accuracy`, `defect_recall`, `defect_precision`, `f1`, `mcc`, `frr`, `miss_rate`, `ecpb`, `latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`, `route_rate_total`, `route_rate_good`, `route_rate_defect`, `rescued_defects`, `rescue_rate`, `added_false_rejects`, `unclassified_positive_rate`.

### `artifacts/test/benchmark_per_group.json` (và `.csv`)
- Đánh giá phân rã theo `group44000` và `group92000`.
- Báo cáo rõ ràng: group name, sample count, good count, defect count, recall, FRR, balanced acc, worst-group recall, worst-group FRR.

### `artifacts/test/benchmark_per_class.json` (và `.csv`)
- Báo cáo 6 lớp cho B01, B02, H01: `open_circuit`, `short`, `mouse_bite`, `spur`, `spurious_copper`, `pin_hole`.
- Chỉ số: `gt_box_count`, `detected_box_count`, `ap_50`, `class_recall`.

### `artifacts/test/predictions_test.jsonl`
Mỗi dòng đại diện cho một ảnh test x mô hình:
```json
{
  "sample_id": "00044000_good",
  "model_id": "H01_cascade_yolo11n_efficientad_s",
  "split": "test",
  "source_group": "group44000",
  "is_defect": false,
  "decision": "GOOD",
  "yolo_score": 0.02,
  "ad_score": 0.85,
  "route_ad": true,
  "reason": "all_pass",
  "latency_ms": 35.2
}
```

## 5. Tiêu chí Nghiệm thu (Acceptance Criteria)
1. **Khóa cấu hình trước (Zero-P-Hacking):** `frozen_evaluation.json` phải được tạo trước khi bất kỳ suy luận test nào diễn ra; checksum và tham số khớp 100% với Step 6 shortlist.
2. **Đủ 440 ảnh canonical:** Không thiếu, không thừa mẫu nào trong 440 ảnh test (220 good, 220 defect); phân bổ chính xác trên 2 nhóm nguồn `group44000` (200 ảnh) và `group92000` (240 ảnh).
3. **Độ trễ đo thực tế (Real Latency):** Phép đo độ trễ phải thông qua pipeline thực (warmup 30 ảnh, batch size 1, `time.perf_counter()`); chứng minh được sự khác biệt độ trễ giữa H00 (chạy cả hai) và H01 (early-exit).
4. **Báo cáo trung thực:** Kết quả tính toán độc lập, khách quan. Không chỉnh sửa lại ngưỡng để làm đẹp số liệu. Phân tích rõ sự đánh đổi giữa recall, FRR, ECPB và latency.
5. **Đầy đủ artifacts:** Toàn bộ 6 tệp artifact và báo cáo nghiệm thu `reports/step7_final_test_evaluation_report.md` được sinh ra hoàn chỉnh và có thể tái lập.
