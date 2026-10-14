# Giới hạn nghiên cứu (Limitations)

Tài liệu này liệt kê minh bạch các giới hạn đã biết của đồ án, bám sát PLAN §1.4, §3.3,
§7.6 và checklist nghiệm thu §11.2. Mọi kết luận trong báo cáo phải đọc kèm các giới hạn này.

## 1. Tập test giới hạn ở N=2 nhóm nguồn [R1 / v3-P0.4]

- Tập test gồm đúng 2 nhóm nguồn (`group44000`, `group92000`); calibration và
  development_selection (fusion) mỗi tập cũng chỉ có 2 nhóm. Số lượng ảnh (440 test)
  không đồng nghĩa số quan sát vật lý độc lập.
- Bootstrap pair-level (nếu dùng) chỉ phản ánh độ bất định nội suy (sampling variance)
  trên chính 2 thiết kế bo này, **không** đại diện cho phương sai chuyển miền
  (domain shift) sang bo mạch bất kỳ.
- Trên 220 good test, 1 FP ≈ 0,45 điểm phần trăm FRR; trên 220 defect, 1 FN ≈ 0,45 điểm
  phần trăm recall. Không kết luận vượt trội từ chênh lệch rất nhỏ.
- **Không thực hiện LOGO-CV 11 groups** (tốn 11× GPU-hours, không tăng N test vì test
  vẫn 2 groups, train cần 5 groups nên leave-one-group-out bỏ mất quá nhiều dữ liệu).
  Thay thế bằng worst-group recall/FRR + pair-level bootstrap (PLAN §7.6).
- Hệ quả: scope được định nghĩa là **chuyển giao giữa các thiết kế mạch trần tương đồng
  (cross-design transfer)** — train trên 5 nhóm nguồn, đánh giá trên 2 nhóm nguồn chưa
  thấy nhưng tương đồng; **không** phải chứng nhận sản xuất biết trước design, **không**
  phải universal zero-shot AOI.

## 2. Phạm vi bare PCB, không linh kiện

- Toàn bộ 2.998 ảnh là PCB trần (bare board, không gắn linh kiện), 640×640.
- `source_group` chưa xác minh được thiết kế hay bo vật lý; đơn vị kết quả là ảnh/crop,
  chưa phải một bo nguyên vẹn.
- Trên bo có linh kiện hoặc layout quá dị biệt, EfficientAD bắt buộc phải fine-tune lại
  (phân phối đường mạch tổng quát đã học không còn đúng). YOLO học đặc trưng vi mô của
  lỗi có thể chuyển giao tốt hơn, nhưng chưa được kiểm chứng ngoài DeepPCB.
- Ảnh good là template đã được tác giả làm sạch, chưa đại diện đầy đủ ảnh tốt tự nhiên
  từ camera; một phần ảnh lỗi được bổ sung thủ công. Overlay chỉ kiểm tra tính hợp lý
  nhãn; không đủ xác nhận hay loại trừ dấu vết tổng hợp hoặc shortcut học được
  (tên file/ID/split không được làm feature — đã kiểm tra, 2252/2252 tên train/calibration
  chứa `_good`/`_defect` nên nguy cơ shortcut tên file là có thật nếu loader bất cẩn).

## 3. Domain shift giữa các thiết kế bo mạch

- Bằng chứng shift đã đo được: B01 calibration FRR 4,78% nhưng test FRR 10,00% tại cùng
  ngưỡng τ_yolo=0,14; worst-group FRR group44000 (16%) gấp ~3× group92000 (5%).
  Fusion (development_selection) FRR còn cao hơn (38,56% cho H01) — tập phát triển cũng
  không còn độc lập sau khi chọn cấu hình.
- Calibration đồng thời phục vụ chọn checkpoint và ngưỡng nên điểm trên tập này có thể
  lạc quan. Test cuối mới dùng để báo chất lượng chưa tham gia lựa chọn; không gọi quy
  trình này là hiệu chuẩn có bảo đảm thống kê.
- Muốn khẳng định khả năng phát hiện loại lỗi chưa thấy cần protocol riêng
  (leave-one-defect-type-out) — là mở rộng, không chặn MVP sáu lớp. Thí nghiệm sáu lớp
  đã biết không tự chứng minh khả năng phát hiện mọi loại lỗi mới.
- Stress test công nghiệp (specular_glare, motion_blur, defocus_blur, alignment_jitter
  ±1–2 px) là track riêng trên tập phát triển, không trộn vào benchmark chính.

## 4. Nguyên lý AD bị khóa khi YOLO dương [S1]

- Trong cascade H01/H02, **khi YOLO đã báo DEFECT (kể cả false positive), EfficientAD
  không chạy trên ảnh đó** — bị khóa hoàn toàn cả về:
  - (a) bỏ sót lỗi lạ thứ hai trên cùng ảnh (cascade không phát hiện đa lỗi khi YOLO
    đã dương tính với một lỗi), và
  - (b) "cứu" ảnh good bị YOLO báo nhầm (cascade không thể giảm FP của tầng một).
- Hệ quả đo được: `rescued_defects = 0`, `added_false_rejects = 0` trên cả fusion và
  test. **H01 không cứu thêm ảnh lỗi nào so với YOLO11n trên test canonical**, vì
  YOLO11n đạt 100% recall tại τ=0.14. Lợi ích của early-exit là giảm tính toán so với H00 OR. Recall mức ảnh không giảm so với YOLO đơn theo cấu trúc luật OR, nhưng
  báo nhầm có thể tăng (hợp của hai nhánh ở H00).
- Recall ảnh tốt không đồng nghĩa phát hiện đủ tất cả box. Chế độ `diagnostic_both`
  (chạy cả hai mọi ảnh) chỉ dùng phân tích bù trừ đa lỗi, không dùng latency này thay
  cho cascade.
- Nếu yêu cầu nghiệm thu gồm định vị và nhận diện đủ sáu loại lỗi, chỉ chọn sản phẩm có
  đầu ra detector phù hợp; EfficientAD-only là đối chứng binary, không đủ chức năng thay
  thế detector sáu lớp (nhãn `anomaly_unclassified`, không ép lớp).

## 5. Giới hạn đo lường và báo cáo

- Không có mask chuẩn nên **không** báo pixel AUROC, Dice, IoU segmentation, AUPRO —
  mọi localization từ heatmap chỉ đánh giá class-agnostic ở bảng phụ
  (Point-in-Box có ràng buộc diện tích, §7.7), không thay mAP sáu lớp.
- YOLO image score (max confidence) và anomaly score (max heatmap) không so sánh được
  trực tiếp; cascade đầu tiên để AUROC/AUPRC là N/A. Không lấy max/trung bình hai thang
  điểm thô làm score hợp nhất.
- Latency Step 7 cũ là **ước tính cộng nhánh**, chưa gồm decode và chưa phải log thực thi cascade. So với H00 OR, các số cũ cho H01 **tiết kiệm 52.7% latency mean và p50 nhanh hơn 21.5 lần** nhờ giả lập ngắt sớm. Cần đo E2E lại cùng tập ảnh trước khi xác nhận hai tỷ lệ này; không dùng chúng làm mức tăng tốc so với YOLO-only.
- Evaluator sửa lỗi dùng `InferenceService` để ghi quyết định, routing và thời gian của cùng lượt chạy, gồm decode, warmup từng pipeline, cold model load riêng và các lượt timing. Các artifact cũ vẫn giữ nguyên; kết quả mới phải có thư mục/version riêng.
- Step 9 đã đo pipeline thật nhưng H00 chỉ có 20 ảnh đối chiếu với 460 ảnh H01. Không suy ra tỷ lệ tăng tốc công bằng từ hai tập khác nhau. YOLO ONNX/OpenVINO chưa được kiểm chứng; fallback PyTorch không phải parity export. TorchScript chỉ được PASS sau khi backend thật chạy đủ mẫu và đạt tolerance.
- B01 đơn có cùng quyết định với H01 trên test tại ngưỡng khóa, với chi phí thấp hơn trong số liệu cũ. Không có bằng chứng rằng H01 sẽ thắng trong kịch bản YOLO bỏ sót tương lai; điều đó cần protocol và dữ liệu mới.
- Bổ sung remediation trên calibration/fusion là phân tích khám phá sau khi biết test: DET và Recall@FRR={1%,3%,5%}, lưới H02 đầy đủ, cùng pilot tile-256 vs resize-256 giữ nguyên checkpoint và normalization. Pilot không thay thế ablation đầy đủ hoặc baseline resize được huấn luyện riêng. Hiện chỉ có seed42; không kết luận độ ổn định qua nhiều seed.
- Checkpoint `last.pt` lịch sử của YOLO đã bị upstream loại optimizer/epoch. Bản sửa bảo toàn trạng thái cho các lần train mới, nhưng không thể khôi phục optimizer đã mất của hai lần train cũ mà không có backup hoặc train lại.
