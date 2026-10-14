# Hợp đồng Step 5 = Bước 5 của PLAN (Hiệu chỉnh ngưỡng 2-D Grid Search và Đánh giá Fusion H00 vs H01)

Quy tắc chung: `docs/handoff/WORKFLOW.md`. Mâu thuẫn với `PLAN.md` thì plan thắng.

## 1. Phạm vi (Scope)
- **Có:**
  - Nạp các tệp dự đoán thô tĩnh đã trích xuất từ Bước 3 (`artifacts/yolo/B01_yolo11n/seed42/preds_*.jsonl`, `artifacts/yolo/B02_yolo11s/seed42/preds_*.jsonl`) và Bước 4 (`artifacts/efficientad/B03_efficientad_s/seed42/preds_*.jsonl`).
  - Lập phân phối điểm: chạy phân tích trên toàn bộ calibration (460 ảnh: 230 good, 230 defect).
  - Thuật toán quét lưới 2 chiều đồng thời $(\tau_{\text{yolo}}, \tau_{\text{ad}})$ có điều kiện trên pipeline cascade calibration:
    - Không gian $\tau_{\text{yolo}} \in [0.10, 0.90]$.
    - Không gian $\tau_{\text{ad}} \in [\min(p_{10}, \mu - \sigma), \max(p_{99.5}, \mu + 3\sigma)]$ từ phân phối defect calibration.
  - Tiêu chí lựa chọn: cặp $(\tau_{\text{yolo}}^*, \tau_{\text{ad}}^*)$ tối đa hóa **Defect Recall (TPR)** tại ràng buộc $\mathbf{\text{FRR} \le 5\%}$ (mốc chính) và $\mathbf{\text{FRR} \le 1\%}$ (mốc phân tích phụ).
  - Tiêu chí hòa (Tie-breaker): Ưu tiên $\tau_{\text{yolo}}$ thấp hơn (tăng early-exit, giảm tải AD), sau đó ưu tiên ECPB thấp hơn. Lưu toàn bộ điểm lưới (`grid_calibration.json`).
  - Thực thi logic kết hợp:
    - **H01 (Cascade):** Nếu `yolo_score >= tau_yolo` $\rightarrow$ DEFECT (`route_ad = False`, `ad_score = None`, `reason = "yolo_hit"`). Ngược lại chuyển AD: nếu `ad_score >= tau_ad` $\rightarrow$ DEFECT (`reason = "ad_hit"`), ngược lại GOOD (`reason = "all_pass"`).
    - **H00 (Luật OR):** Cả 2 nhánh luôn chạy (`route_ad = True`). Kết luận DEFECT nếu (`yolo_score >= tau_yolo` HOẶC `ad_score >= tau_ad`).
    - **H02 (Cascade với YOLO11s):** Tương tự H01 nhưng dùng B02 làm tầng 1.
  - Đánh giá đối chứng trên tập **`fusion` (`development_selection`, 306 ảnh: 153 good, 153 defect)**:
    - Tính các chỉ số: Balanced Accuracy, Defect Recall, Defect Precision, FRR, F1, MCC.
    - Tính ECPB ($N=10.000$, tỷ lệ $1\%$ Defect / $99\%$ Good, $C_{\text{FN}}=100, C_{\text{FP}}=1$).
    - Đo các chỉ số đặc thù hybrid: `route_rate_good`, `route_rate_defect`, `rescued_defects`, `rescue_rate`, `added_false_rejects`, `unclassified_positive_rate`.
    - Tính độ chênh lệch đối chứng: $\Delta \text{recall}(H01 - H00)$, $\Delta \text{FRR}(H01 - H00)$, $\Delta \text{latency}(H01 - H00)$.
  - Xuất artifact: `artifacts/fusion/thresholds.json`, `artifacts/fusion/routing_policy.json`, `reports/step5_calibration_fusion_report.md`.

- **Không:**
  - **TUYỆT ĐỐI CẤM** truy cập tập `test` (`test_access: False`). Không đọc, không nạp, không dùng metadata của test.
  - Không huấn luyện lại mô hình; không chạy lại forward GPU (toàn bộ chạy tĩnh trên kết quả dự đoán đã xuất).
  - Không fit logistic regression head (giữ cascade không tham số, giải thích minh bạch theo PLAN §5 Bước 5).
  - Không chạy benchmark test cuối cùng (dành cho Bước 7).

## 2. Cấu hình (`configs/models/fusion_common.yaml`)

```yaml
recipe_id: fusion_recipe_v1
calibration_split: calibration
development_split: fusion
grid_search:
  yolo_min: 0.10
  yolo_max: 0.90
  yolo_step: 0.02
  ad_num_points: 50
constraints:
  primary_frr_max: 0.05
  secondary_frr_max: 0.01
cost_matrix:
  c_fn: 100.0
  c_fp: 1.0
  defect_prevalence: 0.01
  n_simulation: 10000
```

## 3. Giao diện Module (`src/pcb_lab/fusion/`)

### `rules.py`
- `@dataclass(frozen=True) class RoutingDecision`:
  - `sample_id: str`
  - `split: str`
  - `is_defect: bool`
  - `decision: str` (`"GOOD"` hoặc `"DEFECT"`)
  - `mode: str` (`"CASCADE"` hoặc `"OR"`)
  - `yolo_score: float`
  - `ad_score: Optional[float]` (bắt buộc `None` khi `route_ad = False` trong cascade)
  - `route_ad: bool`
  - `reason: str` (`"yolo_hit"`, `"ad_hit"`, `"all_pass"`)
  - `yolo_boxes: list`
  - `ad_boxes: list`
  - `timing_ms: Optional[float]`
- `predict_cascade(yolo_row, ad_row, tau_yolo, tau_ad) -> RoutingDecision`
- `predict_or(yolo_row, ad_row, tau_yolo, tau_ad) -> RoutingDecision`

### `metrics.py`
- `compute_confusion(decisions: List[RoutingDecision]) -> dict`
- `compute_binary_metrics(tp, tn, fp, fn, cost_params=None) -> dict`:
  - `balanced_accuracy`, `defect_recall`, `defect_precision`, `frr`, `defect_to_good`, `f1`, `mcc`, `ecpb`
- `compute_hybrid_metrics(decisions, yolo_only_decisions=None) -> dict`:
  - `route_rate`, `route_rate_good`, `route_rate_defect`
  - `rescued_defects`, `rescue_rate`
  - `added_false_rejects`, `unclassified_positive_rate`
- `compute_delta_metrics(cascade_metrics, or_metrics) -> dict`:
  - `delta_recall`, `delta_frr`, `delta_ecpb`

### `sweep.py`
- `grid_search_2d(yolo_calib_rows, ad_calib_rows, config) -> GridSearchResult`:
  - Trả về danh sách tất cả các điểm lưới kèm metrics (`grid_points`).
  - Lựa chọn `optimal_primary` ($\text{FRR} \le 0.05$) và `optimal_secondary` ($\text{FRR} \le 0.01$).

### `evaluate.py`
- `evaluate_fusion(yolo_fusion_rows, ad_fusion_rows, policy) -> dict`:
  - Đánh giá toàn diện trên 306 ảnh `fusion` cho cả H00 và H01.

## 4. Cấu trúc Artifact xuất ra (`artifacts/fusion/`)

### `artifacts/fusion/thresholds.json`
```json
{
  "schema_version": 1,
  "recipe_id": "fusion_recipe_v1",
  "calibration_split": "calibration",
  "target_frr": 0.05,
  "h01_yolo11n_cascade": {
    "tau_yolo": 0.25,
    "tau_ad": 0.45,
    "calibration_metrics": { ... }
  },
  "h02_yolo11s_cascade": {
    "tau_yolo": 0.25,
    "tau_ad": 0.45,
    "calibration_metrics": { ... }
  },
  "h00_yolo11n_or": {
    "tau_yolo": 0.25,
    "tau_ad": 0.45,
    "calibration_metrics": { ... }
  }
}
```

### `artifacts/fusion/routing_policy.json`
- Chứa quy tắc định tuyến, ngưỡng đã khóa, checksum các artifact B01, B02, B03 nguồn, và cờ `test_access: False`.

### `reports/step5_calibration_fusion_report.md`
- Báo cáo so sánh đối chứng chi tiết giữa H00 và H01, bảng chỉ số, tỷ lệ early-exit và phân tích lợi ích hybrid.

## 5. Tiêu chí nghiệm thu (Acceptance Criteria)
1. **Zero Test Leak:** Toàn bộ runner và kiểm thử không import, không nạp `test`. Bất kỳ truy cập `test` nào đều bị từ chối bằng `PermissionError`.
2. **Deterministic:** Cùng danh sách dự đoán đầu vào luôn sinh ra cùng bảng điểm lưới và cùng ngưỡng tối ưu.
3. **Logic Cascade chặt chẽ:** Khi YOLO vượt ngưỡng, tầng AD không được chạy (`route_ad == False`), điểm AD phải là `None`, lý do là `"yolo_hit"`.
4. **H00 vs H01 đối chứng minh bạch:** Cùng một cặp ngưỡng $(\tau_{\text{yolo}}, \tau_{\text{ad}})$, so sánh trực tiếp H00 và H01 trên cùng 306 ảnh tập `fusion` để tính $\Delta \text{recall}$, $\Delta \text{FRR}$, $\Delta \text{latency}$.
