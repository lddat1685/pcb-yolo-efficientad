# Hợp đồng Step 6 = Bước 6 của PLAN (Benchmark baseline và chọn shortlist)

Quy tắc chung: `docs/handoff/WORKFLOW.md`. Mâu thuẫn với `PLAN.md` thì plan thắng.

## 1. Phạm vi (Scope)
- **Có:**
  - Hoàn thiện module evaluator/benchmark dùng chung (`src/pcb_lab/evaluation/`).
  - Nạp các dự đoán tĩnh và ngưỡng đã khóa từ Bước 3 (`B01_yolo11n`, `B02_yolo11s`), Bước 4 (`B03_efficientad_s`) và Bước 5 (`thresholds.json`, `routing_policy.json`).
  - Chạy đánh giá đối chứng toàn bộ ma trận baseline bắt buộc trên tập phát triển `fusion` (development selection, 306 ảnh: 153 good, 153 defect) và `calibration` (460 ảnh):
    - **B01:** YOLO11n đơn lẻ.
    - **B02:** YOLO11s đơn lẻ.
    - **B03:** EfficientAD-S đơn lẻ.
    - **H00:** YOLO11n + EfficientAD-S theo luật OR (chạy song song không điều kiện).
    - **H01:** YOLO11n → EfficientAD-S theo cascade hai tầng (định tuyến early-exit).
    - **H02:** YOLO11s → EfficientAD-S theo cascade hai tầng.
  - Tính toán và tổng hợp đầy đủ các chỉ số ảnh theo PLAN §7.2:
    - Balanced Accuracy, Defect Recall (TPR), Defect Precision, FRR, Defect→GOOD (miss rate), F1, MCC, ECPB ($N=10.000, 1\%\text{ defect}, C_{\text{FN}}=100, C_{\text{FP}}=1$).
    - Chỉ số routing: `route_rate_total`, `route_rate_good`, `route_rate_defect`, `rescued_defects`, `rescue_rate`, `added_false_rejects`, `unclassified_positive_rate`.
  - Thực hiện các phân tích Ablation ưu tiên (PLAN §5 Bước 6):
    - **Ablation 1 (Dung lượng detector):** B01 (YOLO11n) vs B02 (YOLO11s).
    - **Ablation 2 (Logic kết hợp):** H00 (Luật OR) vs H01 (Cascade) — đánh giá lợi ích giảm tải tầng 2 (early-exit) so với giữ nguyên recall.
    - **Ablation 3 (Ngưỡng vận hành):** H01 tại ràng buộc chính ($\text{FRR} \le 5\%$) vs ràng buộc phụ ($\text{FRR} \le 1\%$).
  - Lập danh sách Shortlist chính thức cho Bước 7:
    - Ghi nhận `shortlist.json` chứa danh sách cấu hình được chọn (tối thiểu B01, B03, H00, H01, H02), lý do định lượng, metadata và ngân sách tài nguyên đã dùng (GPU-hours, tuning budget).
  - Xuất artifacts:
    - `artifacts/benchmark/development_summary.json` (và `.csv`)
    - `artifacts/benchmark/ablation_summary.json` (và `.csv`)
    - `artifacts/benchmark/shortlist.json`
    - `reports/step6_baseline_benchmark_report.md`

- **Không:**
  - **TUYỆT ĐỐI CẤM** truy cập tập `test` (`test_access: False`). Bất kỳ truy cập `test` nào đều bị từ chối bằng `PermissionError`.
  - Không huấn luyện lại mô hình; không chạy lại forward GPU (toàn bộ chạy tĩnh trên kết quả dự đoán đã xuất).
  - Không đánh giá benchmark test cuối cùng (dành riêng cho Bước 7).

## 2. Giao diện Module (`src/pcb_lab/evaluation/`)

### `benchmark.py`
- `run_development_benchmark(config_path, thresholds_path, policy_path) -> BenchmarkResult`:
  - Nạp toàn bộ dữ liệu dự đoán `calibration` và `fusion`.
  - Tính bảng metric tổng hợp cho B01, B02, B03, H00, H01, H02.
  - Xuất kết quả dạng bảng và cấu trúc dữ liệu chuẩn.

### `ablation.py`
- `compute_ablations(benchmark_result) -> Dict[str, Any]`:
  - Trích xuất $\Delta \text{recall}, \Delta \text{FRR}, \Delta \text{ECPB}$, tỷ lệ tiết kiệm tính toán (saving rate).

### `shortlist.py`
- `select_shortlist(benchmark_result, ablation_result) -> Dict[str, Any]`:
  - Định nghĩa danh sách shortlist kèm luận cứ số liệu và ngân sách.

## 3. Cấu trúc Artifact xuất ra (`artifacts/benchmark/`)

### `artifacts/benchmark/shortlist.json`
```json
{
  "schema_version": 1,
  "selection_split": "fusion",
  "test_access": false,
  "shortlist": [
    {
      "model_id": "H01_cascade_yolo11n_efficientad_s",
      "role": "Primary Hybrid",
      "tau_yolo": 0.14,
      "tau_ad": 1.4598,
      "selection_metrics": { ... },
      "rationale": "Achieves 100% recall with 69.3% early-exit savings"
    }
  ],
  "budget_summary": {
    "gpu_hours": "...",
    "tuning_budget": "..."
  }
}
```

## 4. Tiêu chí nghiệm thu (Acceptance Criteria)
1. **Zero Test Leak:** Không đọc, không nạp, không dùng bất kỳ dữ liệu nào thuộc tập `test`.
2. **Tính nhất quán dữ liệu (Data Provenance):** Mọi con số trong bảng benchmark đối chứng khớp chính xác từng điểm số từ Bước 3, Bước 4 và Bước 5.
3. **Đối chứng H00 vs H01 minh bạch:** Phân tích rõ ràng sự đánh đổi giữa luật OR và Cascade hai tầng.
4. **Shortlist có luận cứ định lượng:** Shortlist đề xuất dựa trên số liệu thực từ tập `fusion` (development selection), không dựa trên cảm tính hay nhìn trước tập `test`.
