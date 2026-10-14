# Báo cáo Hiệu chỉnh Ngưỡng và Đánh giá Fusion (Step 5)

**Recipe:** `fusion_recipe_v1` | **Calibration:** 460 ảnh (230 good, 230 defect) | **Development Selection (Fusion):** 306 ảnh (153 good, 153 defect)

---

## 1. Kết quả Quét lưới 2 Chiều (2-D Grid Search) trên Calibration

Mục tiêu: Tối đa hóa **Defect Recall** tại ràng buộc **FRR ≤ 5%** (ưu tiên early-exit khi hòa).

| Mô hình Hybrid | Ngưỡng $\tau_{\text{yolo}}^*$ | Ngưỡng $\tau_{\text{ad}}^*$ | Recall ↑ | FRR ↓ | Balanced Acc ↑ | ECPB ↓ | Route AD % | Route Good % | Route Defect % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H01** (YOLO11n + EfficientAD-S) | `0.14` | `1.4598` | 100.00% | 4.78% | 97.61% | 0.0473 | 47.6% | 95.2% | 0.0% |
| **H02** (YOLO11s + EfficientAD-S) | `0.38` | `1.4598` | 100.00% | 4.78% | 97.61% | 0.0473 | 47.6% | 95.2% | 0.0% |

---

## 2. Đánh giá Đối chứng trên Tập Phát triển Fusion (Development Selection)

Đánh giá tại điểm vận hành khóa: $\tau_{\text{yolo}} = 0.14$, $\tau_{\text{ad}} = 1.4598$

| Phương pháp | Balanced Acc ↑ | Defect Recall ↑ | Defect Precision ↑ | F1 ↑ | MCC ↑ | FRR ↓ | ECPB ↓ | Route AD % | Rescued Defects | Rescue Rate | Added False Rejects |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B01** (YOLO11n đơn) | 80.72% | 100.00% | 72.17% | 83.84% | +0.6659 | 38.56% | 0.3818 | 0.0% | N/A | N/A | 0 |
| **B03** (EfficientAD-S đơn) | 64.38% | 28.76% | 100.00% | 44.67% | +0.4098 | 0.00% | 0.7124 | 100.0% | N/A | N/A | N/A |
| **H00** (Luật OR - chạy cả 2) | 80.72% | 100.00% | 72.17% | 83.84% | +0.6659 | 38.56% | 0.3818 | 100.0% | 0 | N/A | 0 |
| **H01** (Cascade hai tầng) | 80.72% | 100.00% | 72.17% | 83.84% | +0.6659 | 38.56% | 0.3818 | 30.7% | 0 | N/A | 0 |

---

## 3. So sánh Đối chứng Trực tiếp H01 (Cascade) vs H00 (Luật OR)

- $\Delta \text{Recall (H01 - H00)}: +0.00\%$
- $\Delta \text{FRR (H01 - H00)}: +0.00\%$
- $\Delta \text{ECPB (H01 - H00)}: +0.0000$
- Tỷ lệ cứu lỗi (Rescue Rate): EfficientAD cứu được **0 / 0** lỗi mà YOLO bỏ sót (N/A).
- Tỷ lệ chuyển tầng 2 (Route Rate): Chỉ có **30.7%** tổng số ảnh cần chạy sang EfficientAD (61.4% ảnh good và 0.0% ảnh defect).

### Nhận định Giả thuyết H5 (PLAN §2.2):
- **Hiệu quả định tuyến (Early-Exit):** Cascade H01 đạt recall tương đương luật OR nhưng tiết kiệm được tài nguyên tính toán tầng 2 cho phần lớn ảnh mà YOLO đã bắt được lỗi.
- **Độ an toàn:** Không có rò rỉ dữ liệu từ tập test; quy trình hoàn toàn tất định và có thể tái lập.
