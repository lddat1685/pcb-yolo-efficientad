# TRIAGE — Phán xét plan.md qua review.md

> **Người lập:** Arbiter (tổng hợp/biên tập), không đứng về bên nào.
> **Ngày:** 24/09/2026.
> **Đầu vào bất biến:** `PLAN.md` (tác giả Codex), `Review.md` (tác giả Gemini).
> **Không có** `AGENTS.md`, `SPEC.md` hay thư mục `docs/` trong repo; file thực tế nằm ở gốc.

## 0. Bằng chứng đã kiểm chứng (từ data địa phương + tài liệu)

Đã đọc trực tiếp trong `..\DatasetVer4_Public` (dataset có mặt, hash khớp release):

| Phát biểu | Nguồn kiểm chứng | Kết quả |
|---|---|---|
| Test chỉ có 2 nhóm nguồn (`group44000`, `group92000`), 220 cặp = 440 ảnh | `benchmarks/deeppcb/splits/holdout.json` (`groups.test`, `actual_pair_counts.test=220`) | **ĐÚNG** |
| Tổng 11 nhóm, 2.998 ảnh (1.498 good + 1.500 defect), 5/2/2/2 groups | `holdout.json` + `DATASET_CARD.md` | **ĐÚNG** (1 pair `90100034` bị loại khỏi canonical) |
| 6 lớp, `pin_hole` ≠ `missing_hole` PKU | `configs/classes.json` (`pin_hole_is_not_pku_missing_hole=true`) | **ĐÚNG** |
| Không có pixel mask; không công bố pixel metric | `configs/protocol.json` (`pixel_metrics:null`, `pixel_mask_reason:"No verified pixel masks"`) | **ĐÚNG** |
| `source_group` KHÔNG phải bo/design đã xác minh; scope = tổng quát hóa sang nhóm nguồn chưa thấy | `protocol.json` (`source_group_is_not_verified_board_or_design:true`, `intended_scope:...`) | **ĐÚNG** |
| Phần cứng hiện tại: Intel Arc + PyTorch 2.6.0+cpu, **không có CUDA**, chưa train GPU | `status.json` (`cuda_available:false`, `full_model_training_completed:false`) | **ĐÚNG** — khớp PLAN §1.4 |
| Dataset card nhắc **PaDiM** (đã CPU smoke), không nhắc PatchCore | `DATASET_CARD.md` | Ghi nhận (liên quan S3) |
| Kích thước bbox lỗi nhỏ (mẫu `deeppcb_44000011_defect.txt`): `pin_hole` nhỏ nhất w≈0.039 h≈0.036 → ~25×23 px @640 → ~10×9 px @256 | annotation mẫu | R7 **đúng hướng nhưng cường điệu** "biến mất hoàn toàn/1.2px" |

**Chưa kiểm chứng được:** YOLO26 có tồn tại trong bản Ultralytics hiện tại hay không — web access bị chặn (403) trong môi trường này. PLAN tự ghi "phải kiểm tra tương thích bản cài", tức PLAN cũng không chắc. Để VERIFY (V1) sau khi setup env.

---

## 1. Bảng phán xét các phát hiện của Review.md (R1–R10)

| ID | Tóm tắt | Mức độ (theo Review) | Quyết định | Lý do / bằng chứng | Thay đổi cần làm trong plan.v2 |
|---|---|:---:|---|---|---|
| R1 | Test thực chất chỉ 2 bo mạch (N=2); đề xuất LOGO-CV trên 11 groups | Critical | **ACCEPT-MODIFIED** | Test đúng 2 groups (đã verify). NHƯNG (a) PLAN §1.4 & §7.6 **đã báo rõ** giới hạn này; (b) LOGO-CV trên 11 groups tốn 11 lần train lại, không tăng N test (vẫn 2 groups), và train cần 5 groups (1.792 ảnh) nên leave-one-group-out bỏ mất quá nhiều. Review **quá tay** ở phần đề xuất. | Giữ báo cáo giới hạn (§1.4/§7.6). Thêm `worst-group recall/FRR` theo nhóm (PLAN đã có ý). **KHÔNG** làm LOGO-CV 11 groups; chỉ thêm bootstrap pair-level như PLAN §7.6 đã nói. [R1] |
| R2 | Hiệu chuẩn τ_ad trên tập lỗi quá nhỏ ở tầng 2 | Critical | **ACCEPT** | Đúng: cascade chỉ route AD khi YOLO rỗng, nên defect lọt qua ít. PLAN B-step 5 đã nói "hiệu chỉnh τ_yolo và τ_ad trên calibration, đánh giá quyết định cuối toàn cascade" và "theo dõi riêng calibration good được chuyển tầng hai". Cách sửa của Review (chạy toàn bộ defect calibration qua AD độc lập) khớp hướng PLAN. | Làm rõ: calibrate τ_ad trên **toàn bộ** defect calibration (không chỉ routed), rồi calibrate τ_yolo để kiểm soát route rate, cuối cùng joint-check trên toàn cascade. [R2] |
| R3 | YOLO26 là mô hình không có thật (hallucination) | Critical | **ACCEPT** | Web bị block → chưa verify trực tiếp, NHƯNG: (a) tới 2024 chỉ có YOLO11; (b) PLAN tự ghi "phải kiểm tra tương thích bản cài" → không chắc; (c) B05 là "Mở rộng sau baseline", priority thấp, dễ loại. | Xóa B05 (YOLO26n) khỏi ma trận bắt buộc. Chỉ tái đưa nếu V1 xác nhận tồn tại. Đổi B05 thành placeholder "đối chứng detector mở rộng (verify trước khi chạy)". [R3] |
| R4 | Accuracy = Balanced Acc trên test 50/50; thiếu metric tỷ lệ lỗi công nghiệp | Major | **ACCEPT-MODIFIED** | Accuracy=Balanced Acc trên 50/50 là **đúng toán học** (đã verify). PLAN §7.3 đã chủ động "luôn kèm các metric khác", nên không sai. Nhưng bảng chính nên bớt trùng lặp. Metric prevalence (ECPB) là bổ sung giá trị. | Bỏ `Accuracy` khỏi bảng chính (giữ `Balanced acc.`); thêm cột **ECPB** mô phỏng prevalence 1% defect / 99% good, trọng số `C_FN=50×C_FP` — **VERIFY trọng số (V-H2)**. [R4] |
| R5 | Cascade chặn đa lỗi: YOLO dương → AD không chạy → bỏ sót lỗi lạ thứ 2 | Major | **ACCEPT-MODIFIED** | Đúng logic. NHƯNG PLAN §3.2 **đã thừa nhận**: "Nếu YOLO đã thấy một lỗi, EfficientAD không chạy để tìm các lỗi còn sót". Thiết kế lại routing phức tạp (chạy AD vùng ngoài box) tốn kém cho sinh viên. | Giữ cascade đơn giản; **formalize chế độ `diagnostic_both`** (đã có §3.3) làm nơi đo bù trừ đa lỗi; ghi rõ giới hạn "AD bị khóa khi YOLO dương" vào §3.2 và conclusions. Background-only AD routing để optional/ablation. [R5] |
| R6 | Thiếu so sánh Iso-FRR (DET curve) | Major | **ACCEPT** | PLAN §7.4 đã có "so sánh tại operating point hiệu chỉnh về cùng FRR". DET curve là cụ thể hóa đúng triết lý "không dùng OR-luôn-tăng-recall làm bằng chứng". | Bổ sung vẽ **DET/ROC curve** và cột `Recall @ FRR={1%,3%,5%}` cho B01/B03/H01 trong bảng benchmark. [R6] |
| R7 | Resize 256×256 làm triệt tiêu lỗi nhỏ | Major | **ACCEPT-MODIFIED** | Đúng hướng (lỗi nhỏ suy giảm). NHƯNG: (a) số liệu Review cường điệu — mẫu `pin_hole` ~25px@640 → ~10px@256, **không biến mất hoàn toàn** (verify V2); (b) Review **tự mâu thuẫn**: §2 gợi ý tile 320×320 overlap 32, nhưng Q4 gợi ý 256×256 overlap 15%; (c) PLAN B-step 2 **đã có** hướng dẫn tiling. EfficientAD native là 256, không phải 320. | Giữ chiến lược **tiling** (PLAN B-step 2). Khóa `tile_size=256`, `overlap` cố định, native cho EfficientAD. Ghi rõ so sánh resize-256 vs tile-256 trong ablation §9. Không dùng 320×320 (không native). [R7] |
| R8 | Chưa định nghĩa thuật toán heatmap→bbox | Major | **ACCEPT-MODIFIED** | PLAN §7.2 đã nói class-agnostic localization ở bảng phụ, §7.4 có ý Point-in-Box. Review đúng là cần **khóa thuật toán cụ thể**. | Khóa pipeline: `Map → threshold τ_pixel → Morph Open(k=3) → Connected Components → Area > A_min → Bounding Rect`, công bố `τ_pixel`, `A_min`. Point-in-Box làm metric localization. [R8] |
| R9 | Thiếu trừu tượng hóa `ImageSource` | Minor | **ACCEPT-MODIFIED** | PLAN §4.2 đã nói "Core không phụ thuộc Streamlit; CLI/UI gọi cùng dịch vụ inference". Review đúng là nên formalize interface. | Thêm `BaseImageSource` (`get_frame() -> Tuple[id, np.ndarray]`) tách UI khỏi I/O, bên cạnh `BaseDetector`/`BaseAnomalyDetector` (§4.1). [R9] |
| R10 | Stress test thiếu biến dạng công nghiệp (lóa, rung, mất nét) | Minor | **ACCEPT** | PLAN §9 đã có xoay/mờ/đổi sáng/nén. Bổ sung 3 biến dạng AOI là hợp lý, giữ là stress test riêng (không trộn benchmark chính). | Thêm `specular_glare`, `motion_blur`, `defocus_blur` vào stress suite §9. [R10] |

---

## 2. Phát hiện bổ sung (PLAN sai / lỗ hổng mà Review bỏ sót) — S1–S6

| ID | Tóm tắt | Mức độ | Quyết định | Lý do / bằng chứng | Thay đổi cần làm |
|---|---|:---:|---|---|---|
| S1 | **Cascade khóa AD khi YOLO dương kể cả FP**: một khi YOLO có box đạt ngưỡng (dù là false positive), AD không chạy → hệ thống không thể "cứu" ảnh good bị YOLO báo nhầm. Review R5 chỉ nói về miss, bỏ sót mặt FP. | Minor | **ACCEPT** | PLAN §3.2 đã note "Cascade này không sửa được YOLO báo nhầm ở tầng một". Là limitation đã nhận thức, không phải lỗi mới. | Ghi rõ limitation này vào §3.2 và conclusions. (Không redesign — trùng R5.) [S1] |
| S2 | **Review R7 tự mâu thuẫn nội bộ**: §2 gợi ý tile 320×320 overlap 32, Q4 gợi ý 256×256 overlap 15%. | Minor | **ACCEPT (note)** | Không ảnh hưởng plan, chỉ ghi nhận để plan.v2 chọn nhất quán 256 native. | Không đổi plan; chỉ rõ trong plan.v2 chọn tile 256 native. [S2] |
| S3 | **B04 PatchCore vs PaDiM**: PLAN liệt PatchCore "Nên có", nhưng `DATASET_CARD.md` nhắc **PaDiM** (đã CPU smoke), không nhắc PatchCore. | Minor | **ACCEPT (note)** | Không mâu thuẫn nghiêm trọng (cả 2 đều memory-bank AD). PatchCore vẫn hợp lệ làm đối chứng. | Giữ B04 PatchCore; ghi chú PaDiM đã smoke ở source (tham khảo nếu PatchCore không chạy được). [S3] |
| S4 | **Giá trị hybrid chưa được chứng minh** là lỗ hổng cốt lõi: YOLO11n đã detect 6 lớp (kể cả lỗi nhỏ), có thể không cần EfficientAD. Review Q1 đề cập nhưng không bắt buộc giải quyết. | Major (ẩn) | **HUMAN / VERIFY** | PLAN §1.4, §2.2 H2, §7.6 đã xử lý ("nếu hybrid không có lợi ích đủ lớn, chọn baseline"). Đây là câu hỏi thiết kế cần user quyết định mức độ đặt hypothesis. | Đưa vào HUMAN (H3). Thực nghiệm V3 đo route_rate thực tế. [S4] |
| S5 | **Phần cứng thực tế = Intel Arc + PyTorch CPU, không CUDA** (`status.json` verified). Review P2.3 đề xuất `cudnn.deterministic=True` — **không áp dụng** trên XPU/CPU. | Minor | **ACCEPT-MODIFIED** | status.json xác nhận. Setting cudnn chỉ có ý nghĩa trên CUDA GPU. | Điều chỉnh P2.3: dùng `torch.manual_seed` + `torch.use_deterministic_algorithms(True)`; chỉ set cudnn khi device thực sự là CUDA. [S5] |
| S6 | **Review R1 quá tay** (đề xuất LOGO-CV không khả thi). Đã gộp xử lý tại R1. | — | (gộp R1) | — | — |

---

## 3. Các mục HUMAN (chỉ user quyết định được)

- **H1 — Có nên làm LOGO-CV 11 groups?** Khuyến nghị **KHÔNG**: tốn 11× GPU-hours, không tăng N test (vẫn 2 groups), train cần 5 groups. Dùng worst-group + pair-level bootstrap (PLAN §7.6) thay thế. → Quyết định: [R1]
- **H2 — Trọng số chi phí `C_FN=50×C_FP` (ECPB)** có phản ánh đúng bài toán nhà máy không? Cần user cung cấp tỷ lệ lỗi thực tế (review gợi ý 1%) và chi phí FP/FN. → Quyết định: [R4] / VERIFY
- **H3 — Chấp nhận kết luận trung thực "hybrid có thể không tốt hơn YOLO đơn"** vào bảo vệ luận văn? Khuyến nghị **CÓ** (Occam's Razor, PLAN §7.6 đã hướng). → Quyết định: [S4]

## 4. Các mục VERIFY (cần dữ liệu/thực nghiệm trước khi chốt)

- **V1 — YOLO26 có tồn tại trong bản Ultralytics cài đặt không?** Web block; verify sau setup env: `pip show ultralytics` + kiểm tra model registry. Nếu không tồn tại → B05 bị loại vĩnh viễn (R3).
- **V2 — Đo thực tế kích thước bbox lỗi nhỏ sau `resize-256` vs `tile-256`** (R7). Cần chạy trên annotation để biết lỗi nào (pin_hole/mouse_bite) mất hoàn toàn.
- **V3 — Đo `route_rate` thực tế trên test** (S1/S4/S5). Cần YOLO11n baseline để biết AD chạy bao nhiêu % (dự kiến cao trên good, thấp trên defect).

---

## 5. Tổng kết giai đoạn 1 (tạm dừng chờ xác nhận)

- **ACCEPT:** R2, R3, R6, R10, S1
- **ACCEPT-MODIFIED:** R1, R4, R5, R7, R8, R9, S5
- **REJECT:** (không có mục review sai hoàn toàn; R1 phần LOGO-CV bị bác)
- **VERIFY:** V1, V2, V3 (gắn với R3/R4/R7/S1/S4)
- **HUMAN:** H1, H2, H3

