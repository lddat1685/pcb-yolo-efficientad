# Báo cáo Triển khai Step 7 = Bước 7 của PLAN (Khóa cấu hình và đánh giá test)

Quy ước làm việc: `docs/handoff/WORKFLOW.md`. Kế hoạch gốc: `PLAN.md` §5 (Bước 7), §6, §7 và `docs/handoff/step7-contract.md`.

## 1. Phase 0: Phân tích Đặc tả & Kế hoạch Thực thi

### 1.1. Diễn giải yêu cầu
- Chuyển từ giai đoạn phát triển (calibration/fusion) sang đánh giá kiểm chứng cuối cùng trên tập **test** (440 ảnh canonical: 220 good, 220 defect).
- Bảo đảm quy tắc **Zero P-Hacking**: File `artifacts/test/frozen_evaluation.json` phải được sinh ra và chốt trước khi thực hiện suy luận trên bất kỳ ảnh test nào. Cấm tuyệt đối việc điều chỉnh lại ngưỡng hay huấn luyện lại sau khi xem tập test.
- Đánh giá trên toàn bộ ma trận bắt buộc:
  - B01 (YOLO11n, $\tau_{\text{yolo}}=0.14$)
  - B02 (YOLO11s, $\tau_{\text{yolo}}=0.38$)
  - B03 (EfficientAD-S, $\tau_{\text{ad}}=1.4598$)
  - H00 (Luật OR B01+B03: $\tau_{\text{yolo}}=0.14, \tau_{\text{ad}}=1.4598$)
  - H01 (Cascade B01 $\to$ B03: $\tau_{\text{yolo}}=0.14, \tau_{\text{ad}}=1.4598$)
  - H02 (Cascade B02 $\to$ B03: $\tau_{\text{yolo}}=0.38, \tau_{\text{ad}}=1.4598$)
  - H01 phụ ($\text{FRR} \le 1\%$: $\tau_{\text{yolo}}=0.48, \tau_{\text{ad}}=1.4598$).
- Đo độ trễ E2E bằng pipeline thực tế (`time.perf_counter()`), warmup 30 lượt, batch size 1. H00 thực thi cả 2 mô hình; H01/H02 áp dụng early-exit thực tế.
- Báo cáo phân rã đa chiều:
  - Toàn thể 440 ảnh test.
  - Phân rã theo 2 nhóm nguồn `group44000` (200 ảnh) và `group92000` (240 ảnh), chỉ rõ worst-group recall và worst-group FRR (PLAN §1.4, §7.6).
  - Phân rã theo 6 lớp khuyết tật (`open_circuit`, `short`, `mouse_bite`, `spur`, `spurious_copper`, `pin_hole`) cho các mô hình YOLO.

### 1.2. Danh sách tệp triển khai
1. `src/pcb_lab/evaluation/test_eval.py` [NEW]: Module lõi đóng băng cấu hình và thực thi đánh giá test.
2. `scripts/evaluate_test.py` [NEW]: CLI thực thi đánh giá `--freeze-only`, `--run`, `--all`.
3. `src/pcb_lab/fusion/rules.py` [MODIFY]: Bổ sung tham số `allow_test: bool = False` có kiểm soát cho `validate_sample_alignment`, `predict_cascade`, `predict_or`.
4. `src/pcb_lab/fusion/evaluate.py` [MODIFY]: Bổ sung tham số `allow_test: bool = False` cho `evaluate_split`.
5. `artifacts/test/frozen_evaluation.json` [NEW]: Tệp cấu hình đã khóa trước khi test.
6. `artifacts/test/predictions_test.jsonl` [NEW]: Bản ghi dự đoán chi tiết từng ảnh test.
7. `artifacts/test/benchmark_summary.json` & `.csv` [NEW]: Bảng tổng hợp mức ảnh.
8. `artifacts/test/benchmark_per_group.json` & `.csv` [NEW]: Bảng phân rã theo nhóm nguồn.
9. `artifacts/test/benchmark_per_class.json` & `.csv` [NEW]: Bảng phân rã 6 lớp lỗi.
10. `reports/step7_final_test_evaluation_report.md` [NEW]: Báo cáo nghiệm thu Markdown.

---

## 2. Kết quả Thực thi và Đánh giá Thực tế

### 2.1. Đóng băng Cấu hình (Frozen Evaluation Verification)
- Tệp `artifacts/test/frozen_evaluation.json` được tạo thành công trước khi nạp ảnh test:
  - `split_name`: `test` (440 ảnh canonical: 220 good, 220 defect).
  - SHA-256 danh sách 440 mẫu test: `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24`.
  - Checkpoint B01 (YOLO11n): SHA-256 `c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05`.
  - Checkpoint B02 (YOLO11s): SHA-256 `aa079248bd566b3adf8863e3b43dde93d3392307a3ef17107e4c126bf410936c`.
  - Checkpoint B03 (EfficientAD-S): SHA-256 `9453bc2c740e81dfb8949df929ed001d60ebf8e5b0f6a18c95eddd9b3a2814a0`.
  - Quyền truy cập test: `test_access_authorized: True`.

### 2.2. Bảng Tổng hợp Mức Ảnh trên 440 Ảnh Test (PLAN §7.2)

| Model | Balanced Acc | Defect Recall | Defect Precision | FRR | Miss Rate | F1 | MCC | ECPB | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) | Route % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B01** (YOLO11n, $\tau=0.14$) | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 120.2 | 120.5 | 140.4 | N/A |
| **B02** (YOLO11s, $\tau=0.38$) | 82.50% | 100.00% | 74.07% | 35.00% | 0.00% | 0.8511 | 0.6939 | 0.3465 | 165.6 | 161.9 | 194.3 | N/A |
| **B03** (EfficientAD-S, $\tau=1.4598$) | 68.18% | 36.36% | 100.00% | 0.00% | 63.64% | 0.5333 | 0.4714 | 0.6364 | 2733.2 | 2723.1 | 2850.6 | 100.0% |
| **H00** (OR B01+B03) | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 2853.4 | 2841.2 | 2978.0 | 100.0% |
| **H01** (Cascade B01 $\to$ B03) | 95.00% | 100.00% | 90.91% | 10.00% | 0.00% | 0.9524 | 0.9045 | 0.0990 | 1348.8 | 132.3 | 2938.4 | 45.0% |
| **H02** (Cascade B02 $\to$ B03) | 82.50% | 100.00% | 74.07% | 35.00% | 0.00% | 0.8511 | 0.6939 | 0.3465 | 1054.1 | 167.8 | 2970.4 | 32.5% |
| **H01_sec** (Cascade, $\tau=0.48$) | 97.27% | 99.55% | 95.22% | 5.00% | 0.45% | 0.9733 | 0.9464 | 0.0540 | 1423.3 | 144.2 | 2952.8 | 47.7% |

### 2.3. Phân tích Định tuyến & Đối chứng H01 vs H00
1. **Recall & Quyết định lỗi:**
   - Tại điểm vận hành chính ($\tau_{\text{yolo}}=0.14$), YOLO11n phát hiện 100% ảnh lỗi (220/220).
   - Do đó, trong Cascade H01, toàn bộ 220 ảnh defect đều được early-exit ngay tại tầng 1 (route_rate_defect = 0.0%).
   - Tầng EfficientAD được gọi trên 90.0% ảnh good (198/220 ảnh good mà YOLO cho qua). B03 đạt FRR = 0.0% nên không sinh thêm false reject nào (added_false_rejects = 0).
   - Vì vậy, H01 và H00 có chất lượng quyết định tương đương: $\Delta\text{recall} = 0.00\%$, $\Delta\text{FRR} = 0.00\%$.
2. **Tiết kiệm Độ trễ (Early-exit Speedup):**
   - **Tỷ lệ early-exit toàn pipeline:** 55.0% (242 / 440 ảnh thoát ngay sau YOLO).
   - **Độ trễ trung bình:** H01 giảm từ 2853.4 ms (H00) xuống 1348.8 ms ($\Delta\text{latency} = -1504.6\text{ ms}$, giảm hơn 52.7%).
   - **Độ trễ trung vị (p50):** H01 đạt 132.3 ms so với 2841.2 ms của H00 (giảm 21.5 lần, xấp xỉ thời gian chạy YOLO đơn lẻ).
3. **Điểm vận hành phụ H01_sec ($\tau_{\text{yolo}}=0.48$):**
   - Giảm mạnh FRR từ 10.00% xuống **5.00%** (chỉ còn 11 false rejects trên toàn bộ 220 good).
   - Đạt Balanced Accuracy cao nhất: **97.27%**.
   - Chi phí kỳ vọng nhà máy thấp nhất toàn benchmark: **ECPB = 0.0540**.

### 2.4. Phân tích Phân rã theo Nhóm Nguồn (Worst-Group Analysis)

| Model | Nhóm Nguồn | Số mẫu | Good | Defect | Defect Recall | FRR | Balanced Acc | Worst-Group Recall | Worst-Group FRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B01** | `group44000` | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 16.00% |
| **B01** | `group92000` | 240 | 120 | 120 | 100.00% | 5.00% | 97.50% | 100.00% | 16.00% |
| **B03** | `group44000` | 200 | 100 | 100 | 40.00% | 0.00% | 70.00% | 33.33% | 0.00% |
| **B03** | `group92000` | 240 | 120 | 120 | 33.33% | 0.00% | 66.67% | 33.33% | 0.00% |
| **H01** | `group44000` | 200 | 100 | 100 | 100.00% | 16.00% | 92.00% | 100.00% | 16.00% |
| **H01** | `group92000` | 240 | 120 | 120 | 100.00% | 5.00% | 97.50% | 100.00% | 16.00% |
| **H01_sec** | `group44000` | 200 | 100 | 100 | 100.00% | 9.00% | 95.50% | 99.17% | 9.00% |
| **H01_sec** | `group92000` | 240 | 120 | 120 | 99.17% | 1.67% | 98.75% | 99.17% | 9.00% |

- Trên `group92000`, H01_sec đạt chất lượng vượt trội: Recall 99.17%, FRR chỉ 1.67% (2/120 ảnh good), Bal. Acc 98.75%.
- Trên `group44000`, FRR cao hơn (16.0% ở H01 và 9.0% ở H01_sec), phản ánh domain variance giữa 2 thiết kế bo mạch.

### 2.5. Phân tích Phát hiện 6 Lớp Khuyết tật (Per-Class Detection)
Tổng số GT boxes trên tập test: đúng **1.407 boxes** (khớp chính xác 100% PLAN §1.3):
- `open_circuit`: 298 boxes (B01: AP50 = 0.8175, Recall = 87.58%)
- `short`: 215 boxes (B01: AP50 = 0.8029, Recall = 79.07%)
- `mouse_bite`: 250 boxes (B01: AP50 = 0.7529, Recall = 76.80%)
- `spur`: 228 boxes (B01: AP50 = 0.7358, Recall = 71.49%)
- `spurious_copper`: 194 boxes (B01: AP50 = 0.8806, Recall = 90.72%)
- `pin_hole`: 222 boxes (B01: AP50 = 0.7680, Recall = 90.54%)

### 2.6. Kết luận & Đề xuất Bàn giao
1. **Tuân thủ thỏa ước:** Đã hoàn thành 100% tiêu chí nghiệm thu của `docs/handoff/step7-contract.md`.
2. **Khóa cấu hình minh bạch:** Không có hiện tượng tuning hay nhìn trước dữ liệu test.
3. **Mô hình đề xuất cho Bước 8 (Demo & UI):** `H01_cascade_yolo11n_efficientad_s` (chính) và cấu hình vận hành `H01_sec` ($\tau_{\text{yolo}}=0.48$).
4. Toàn bộ 8 tệp artifact và báo cáo tại `artifacts/test/` và `reports/` có thể tái lập hoàn toàn qua CLI `python scripts/evaluate_test.py --run`.

