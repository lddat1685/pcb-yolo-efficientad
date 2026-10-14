# DECISIONS — Quyết định triage và áp dụng vào plan.v2.md

> Tài liệu đi kèm `docs/TRIAGE.md` và `plan.v2.md`. Mọi mục Critical/Major từ `Review.md` đều có quyết định. Mục REJECT/HUMAN/VERIFY cũng được ghi đầy đủ dòng tại đây (theo quy tắc Giai đoạn 2).

## 1. Bảng quyết định tổng hợp

| ID | Tóm tắt (từ Review / tự phát hiện) | Mức độ | Quyết định | Áp dụng vào plan.v2.md? |
|---|---|:---:|---|---|
| R1 | Test chỉ 2 nhóm nguồn; đề xuất LOGO-CV 11 groups | Critical | ACCEPT-MODIFIED | ✅ Giữ báo giới hạn, worst-group; **KHÔNG** LOGO-CV |
| R2 | Hiệu chuẩn τ_ad trên tập lỗi quá nhỏ | Critical | ACCEPT | ✅ Calibrate τ_ad toàn defect calib |
| R3 | YOLO26 là mô hình không có thật | Critical | ACCEPT | ✅ Loại B05 khỏi ma trận bắt buộc |
| R4 | Accuracy = Balanced Acc; thiếu ECPB | Major | ACCEPT-MODIFIED | ✅ Bỏ Accuracy, thêm ECPB (trọng số VERIFY) |
| R5 | Cascade chặn đa lỗi (YOLO dương → AD không chạy) | Major | ACCEPT-MODIFIED | ✅ Formalize diagnostic_both; ghi rõ giới hạn |
| R6 | Thiếu Iso-FRR (DET curve) | Major | ACCEPT | ✅ Thêm DET/ROC + Recall@FRR |
| R7 | Resize 256 làm mất lỗi nhỏ | Major | ACCEPT-MODIFIED | ✅ Tiling native 256; bỏ tile 320 |
| R8 | Chưa định nghĩa heatmap→bbox | Major | ACCEPT-MODIFIED | ✅ Khóa pipeline + τ_pixel, A_min, Point-in-Box |
| R9 | Thiếu ImageSource abstraction | Minor | ACCEPT-MODIFIED | ✅ Thêm BaseImageSource §4.4 |
| R10 | Stress test thiếu biến dạng công nghiệp | Minor | ACCEPT | ✅ Thêm specular_glare/motion_blur/defocus_blur |
| S1 | Cascade khóa AD kể cả khi YOLO FP | Minor | ACCEPT | ✅ Ghi limitation vào §3.2/conclusions |
| S2 | Review R7 tự mâu thuẫn 320 vs 256 | Minor | ACCEPT (note) | ✅ Ghi chú chọn 256 native |
| S3 | PatchCore vs PaDiM inconsistency | Minor | ACCEPT (note) | ✅ Giữ B04, note PaDiM đã smoke |
| S4 | Giá trị hybrid chưa chứng minh (cốt lõi) | Major (ẩn) | HUMAN/VERIFY | ⏳ Đưa H3; V3 đo route_rate |
| S5 | Phần cứng Intel Arc CPU, không CUDA | Minor | ACCEPT-MODIFIED | ✅ Điều chỉnh P2.3 deterministic |

## 2. Mục REJECT (ghi rõ dòng)

- **R1 — phần đề xuất LOGO-CV trên 11 groups: REJECT.**
  - Lý do: (a) Test vẫn chỉ 2 groups nên LOGO-CV không tăng N test; (b) train cần 5/11 groups (1.792 ảnh), leave-one-group-out bỏ mất quá nhiều; (c) tốn 11× GPU-hours không cân xứng với sinh viên. PLAN §1.4 & §7.6 đã báo rõ giới hạn N=2, nên Review R1 quá tay ở phần đề xuất.
  - Thay thế bằng: worst-group recall/FRR + pair-level bootstrap (PLAN §7.6).
  - Áp dụng: §1.4, §7.6 plan.v2 ghi rõ "Không thực hiện LOGO-CV 11 groups".

## 3. Mục HUMAN (chỉ user quyết định)

- **H1 — Có làm LOGO-CV 11 groups không?** → Khuyến nghị KHÔNG (đã REJECT đề xuất tương ứng R1). Chờ user xác nhận. [R1]
- **H2 — Trọng số chi phí `C_FN=50×C_FP` (ECPB) có hợp lý?** → Cần user cung cấp tỷ lệ lỗi thực tế (review gợi ý 1%) và chi phí FP/FN nhà máy. [R4]
- **H3 — Chấp nhận kết luận trung thực "hybrid có thể không tốt hơn YOLO đơn"?** → Khuyến nghị CÓ (Occam's Razor, PLAN §7.6 đã hướng). [S4]

## 4. Mục VERIFY (cần dữ liệu/thực nghiệm)

- **V1 — YOLO26 có tồn tại trong bản Ultralytics cài đặt?** Web bị chặn (403) trong env này. Verify sau setup: `pip show ultralytics` + model registry. Nếu không tồn tại → B05 loại vĩnh viễn. [R3]
- **V2 — Kích thước bbox lỗi nhỏ sau tile-256 vs resize-256.** Cần chạy trên annotation (mẫu pin_hole ~25px@640 → ~10px@256, không "biến mất" như R7 cường điệu). [R7]
- **V3 — `route_rate` thực tế trên test.** Cần YOLO11n baseline; dự kiến AD chạy cao trên good, thấp trên defect. Cơ sở đo cho S1/S4. [S1/S4]

## 5. Tự kiểm tra cuối (Giai đoạn 2)

### (a) Mọi mục Critical/Major đều có quyết định
- Critical: R1 (ACCEPT-MODIFIED), R2 (ACCEPT), R3 (ACCEPT) — ✅ đủ.
- Major: R4, R5, R6, R7, R8 (đều có) + S4 (HUMAN/VERIFY) — ✅ đủ.
- Minor: R9, R10, S1, S2, S3, S5 — ✅ đủ.

### (b) plan.v2 nhất quán nội bộ
- Tên metric: `Balanced acc.` (không `Accuracy`) dùng nhất quán §7.2/§7.3/§10.2. ✅
- `ECPB` thêm ở §7.2 và §7.3, kèm VERIFY trọng số (H2). ✅
- Interface: `BaseImageSource` khai báo §4.4, dùng ở §4.1/§8/§10.1. ✅
- Thứ tự làm: tile-256 áp dụng B-step 2/4, ablation §9, lộ trình §12. ✅
- Phạm vi vs thời gian: B05 lùi lại, 4–6 tuần giữ nguyên. ✅
- DET/ROC: §7.4 + §10.4 + §12 (Tuần 5). ✅

### (c) Câu hỏi còn mở
1. H1: LOGO-CV có làm không? (khuyến nghị không)
2. H2: Trọng số ECPB `C_FN=50×C_FP` có đúng bài toán?
3. H3: Có chấp nhận kết luận hybrid có thể thua YOLO đơn?
4. V1: YOLO26 có tồn tại (sau setup env)?
5. V2: Lỗi nhỏ nào mất hoàn toàn ở resize-256?
6. V3: route_rate thực tế trên test là bao nhiêu?

## 6. Tóm tắt

- **Số mục theo quyết định:** ACCEPT = 5 (R2, R3, R6, R10, S1); ACCEPT-MODIFIED = 8 (R1, R4, R5, R7, R8, R9, S5, +S2/S3 note); REJECT (phần) = 1 (R1-LOGO-CV); HUMAN = 3; VERIFY = 3.
- **5 thay đổi lớn nhất:**
  1. Loại YOLO26 (B05) khỏi ma trận bắt buộc — chỉ re-add nếu VERIFY tồn tại [R3].
  2. EfficientAD chuyển từ resize-256 sang **tiling 256 native** + ablation bắt buộc [R7].
  3. Hiệu chuẩn ngưỡng 2 chiều: τ_ad trên toàn defect calib, τ_yolo kiểm soát route_rate [R2].
  4. Bảng benchmark: bỏ `Accuracy`, thêm `ECPB` (prevalence 1%/99%) + `Recall@FRR`/DET curve [R4/R6].
  5. Ghi rõ giới hạn: N=2 test group (không LOGO-CV) và AD bị khóa khi YOLO dương (kể cả FP) [R1/S1].
- **Rủi ro còn lại:**
  - Giá trị hybrid chưa chứng minh (S4/H3) — có thể kết luận YOLO đơn đủ tốt.
  - Phần cứng thực tế là Intel Arc CPU (S5) — chưa train GPU; latency/pareto có thể không đại diện production.
  - Web bị chặn nên V1 (YOLO26) chưa verify; B05 giữ placeholder.
  - N=2 về mặt thống kê — mọi kết luận tổng quát hóa đều yếu, chỉ bootstrap pair-level bù đắp một phần.
