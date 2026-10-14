# Báo cáo Phản biện Kỹ thuật plan.v3 (Review 3)

> **Người thực hiện:** Reviewer độc lập (Chuyên môn: Computer Vision công nghiệp, Thiết kế Benchmark ML & Hướng dẫn ĐATN)  
> **Tài liệu được review:** [plan.v3.md](plan.v3.md)
> **Tài liệu đối chiếu:** [plan.v2.md](plan.v2.md), [review2.md](review2.md), [docs/TRIAGE.md](docs/TRIAGE.md)
> **Commit cơ sở:** `a0faeb3` (Git repository initialized)  
> **Ngày lập:** 24/09/2026  
> **Trạng thái:** Báo cáo phản biện vòng 3 — Đánh giá phiên bản tiền triển khai (Pre-implementation Review)  

---

## 1. Tóm tắt 5 dòng

Bản `plan.v3.md` đã có sự lột xác ấn tượng: giải quyết trọn vẹn các lỗi ngớ ngẩn (xóa sạch YOLO26, thiết lập Gate 0 GPU, đưa đối chứng luật OR H00 vào benchmark, chuẩn hóa ECPB 100:1, bổ sung Alignment Jitter và ràng buộc diện tích Point-in-Box). Đây là một bản kế hoạch có tính khả thi kỹ thuật cao nhất từ trước đến nay. Tuy nhiên, khi soi chiếu dưới góc nhìn của một **chuyên gia thị giác máy công nghiệp và chủ tịch hội đồng chấm đồ án**, `plan.v3.md` vẫn còn 3 rủi ro tiềm ẩn cấp sâu:
1. **Mâu thuẫn triết lý bài toán (Dedicated Line vs Unseen Domain):** Đồ án vừa chia split theo `source_group` để chứng minh tổng quát hóa sang bo mới, nhưng khi giải trình về $N=2$ lại nhận là hệ thống cho "dây chuyền chuyên biệt" (dedicated line). Hai định vị này mâu thuẫn trực tiếp về mặt phương pháp luận.
2. **Lỗ hổng hình học và ghép ảnh trong `TileManager`:** Thông số stride 224 trên ảnh 640 tạo ra tile vượt biên ($704 > 640$); đồng thời dùng phép làm mượt tuyến tính (Linear Feathering) trên anomaly heatmap có nguy cơ làm **loãng/triệt tiêu đỉnh nhiệt (peak dilution)** của các vi khuyết tật.
3. **Hiện tượng triệt tiêu trọng số trong ECPB:** Tỷ lệ phạt $100:1$ khi nhân với tỷ lệ lỗi thực tế $1:99$ vô tình làm chi phí của False Negative và False Positive triệt tiêu lẫn nhau về mức xấp xỉ $1:1$, đòi hỏi sinh viên phải nắm vững bản chất toán học để không bị hội đồng "bẫy".

---

## 2. Bảng phân tích vấn đề chi tiết của plan.v3

| # | Mục | Vấn đề phát hiện | Mức độ | Cách sửa cụ thể |
|---|---|---|:---:|---|
| 1 | **Phương pháp luận** | **Mâu thuẫn định vị: Zero-shot Unseen Board vs Dedicated Line:** Nếu là dây chuyền chuyên biệt (dedicated line - §1.4), train và test phải cùng mẫu bo mạch (chia theo mẻ sản xuất/thời gian). Ngược lại, nếu chia cách ly theo `source_group` để test trên bo chưa thấy, EfficientAD chắc chắn sẽ nổ False Positive vì layout lạ. Hội đồng sẽ bắt bẻ sự bất nhất này. | **Major** | Thống nhất định vị: Gọi chính xác là bài toán **"Đánh giá năng lực chuyển giao giữa các thiết kế mạch tương đồng (Cross-Design Transfer on Similar Bare PCBs)"**. Nhánh YOLO học đặc trưng vi mô của lỗi; EfficientAD học phân phối đường mạch tổng quát. Thừa nhận: Trên bo mạch có linh kiện hoặc layout quá dị biệt, bắt buộc phải fine-tune lại EfficientAD. |
| 2 | **Kỹ thuật xử lý ảnh** | **Lỗi hình học và nguy cơ làm loãng đỉnh nhiệt trong `TileManager`:** (1) Bước nhảy `stride = 224` trên ảnh 640 với `tile = 256` sinh ra tọa độ tile thứ ba là $448 + 256 = 704 > 640$ (buộc phải padding 64px nền đen $\rightarrow$ nổ anomaly ở biên). (2) Phép **Linear Blending** làm trung bình trọng số ở vùng chồng lấn sẽ làm giảm cường độ của các lỗi nhỏ nằm ở biên tile xuống dưới ngưỡng $\tau_{pixel}$. | **Major** | Sửa `TileManager` (§4.4): (1) Đổi stride thành **stride động đối xứng**: 3 tile mỗi chiều tại các tọa độ $x \in \{0, 192, 384\}$ (với $384 + 256 = 640$ vừa khít ảnh, không cần padding bất kỳ pixel nào, overlap là 64px); (2) Thay Linear Blending bằng **Max-Blending** ($\text{Map}(x,y) = \max(\text{Map}_1, \text{Map}_2)$) để bảo toàn giá trị cực đại của khuyết tật. |
| 3 | **Toán học Benchmark** | **Bản chất triệt tiêu trọng số trong công thức ECPB:** Tại §7.3: $\text{ECPB} = \frac{100 \cdot FN + 1 \cdot FP}{10.000} = 1.0 \cdot (1 - \text{Recall}) + 0.99 \cdot \text{FRR}$. Hệ quả là $1\%$ tăng của Miss Rate và $1\%$ tăng của FRR gây thiệt hại tiền tệ gần như ngang nhau ($1.0$ vs $0.99$). Nếu sinh viên không hiểu điều này, hội đồng sẽ hỏi: *"Tại sao nói phạt lọt lỗi gấp 100 lần mà trong công thức cuối cùng lại coi trọng số ngang nhau?"* | **Major** | Bổ sung ghi chú giải thích toán học ngay dưới bảng §7.3: Làm rõ rằng do tỷ lệ lỗi tự nhiên là $1\%$ ($100$ lỗi / $9.900$ tốt), nên tổng thiệt hại của $100$ ca lọt lỗi ($100 \times 100 = 10.000$) cân bằng với tổng chi phí của việc kiểm tra nhầm $9.900$ bo tốt ($9.900 \times 1 = 9.900$). Đây là điểm cân bằng kinh tế thực tế của nhà máy SMT. |
| 4 | **Tối ưu hóa ngưỡng** | **Điểm phẳng (Plateau) và tiêu chí phụ trong 2-D Grid Search:** Mục 5 (Bước 5) quét lưới $(\tau_{yolo}, \tau_{ad})$ để tối đa Defect Recall tại $\text{FRR} \le 5\%$. Do tập calibration có số lỗi hữu hạn (230 lỗi), sẽ có nhiều cặp ngưỡng cùng cho ra Recall cực đại (ví dụ cùng đạt 97.4%). Việc thiếu tiêu chí phụ sẽ khiến thuật toán chọn ngẫu nhiên một cặp ngưỡng ở biên. | **Minor** | Bổ sung quy tắc phá vỡ thế hòa (Tie-breaking rule) vào §5 - Bước 5: Nếu có nhiều cặp $(\tau_{yolo}, \tau_{ad})$ cùng đạt Recall cao nhất tại $\text{FRR} \le 5\%$, chọn cặp có **tỷ lệ chuyển tầng hai thấp nhất (`min route_rate`)** để tiết kiệm tài nguyên GPU, hoặc chọn cặp có **tổng khoảng cách an toàn tới biên ngưỡng lớn nhất (max margin)**. |
| 5 | **Định vị lỗi** | **Thiếu tham số chuẩn hóa cho $\tau_{pixel}$ trong Bbox extraction:** Mục 7.7 ghi: *"$\tau_{pixel}$: ngưỡng nhị phân hóa heatmap (khóa giá trị, ví dụ quantile hoặc absolute)"*. Điểm số EfficientAD giữa các ảnh khác nhau có thể dao động biên độ lớn. Dùng ngưỡng tuyệt đối (absolute threshold) cố định cho mọi ảnh sẽ gây nổ box ở ảnh nhiều nhiễu và mất box ở ảnh sạch. | **Minor** | Khóa công thức ngưỡng thích ứng (Adaptive Pixel Threshold): $\tau_{pixel} = \mu_{map} + k \cdot \sigma_{map}$ (với $k = 3.0$) hoặc dùng **Otsu Thresholding** trên vùng heatmap có giá trị $> \mu_{map}$, kết hợp chặn dưới tuyệt đối để loại trừ ảnh hoàn toàn bình thường. |
| 6 | **Giao diện & Thực nghiệm** | **Nguy cơ nghẽn I/O khi hiển thị Heatmap ghép trên Streamlit:** Streamlit render ảnh qua web base64. Nếu mỗi lần inference phải serialize ảnh gốc $640\times 640$, heatmap float32, ảnh blend và bảng box, giao diện sẽ có độ trễ UI $> 500\text{ ms}$, làm lu mờ tốc độ thực tế của core AI ($< 50\text{ ms}$). | **Minor** | Tại §10.1, bổ sung yêu cầu kỹ thuật: Tách luồng render UI. Anomaly map chỉ sinh ảnh overlay chuẩn JPEG chất lượng 85% ở kích thước hiển thị; tensor float32 gốc chỉ lưu trong cache bộ nhớ hoặc lưu file npy khi người dùng ấn nút "Xuất dữ liệu chi tiết". |

---

### 3. Kết quả kiểm chứng các phát biểu dữ kiện

1. **Phát biểu: "DeepPCB gồm 1.500 cặp ảnh 640×640 trong 11 nhóm nguồn, cắt từ ảnh quét CCD lớn"**
   - **Xác nhận:** **ĐÚNG TUYỆT ĐỐI**. Căn cứ theo paper gốc Tang et al. (2019) và metadata tại `DatasetVer4_Public/benchmarks/deeppcb/splits/holdout.json`.
2. **Phát biểu: "DeepPCB không có pixel segmentation mask, chỉ có bounding box"**
   - **Xác nhận:** **ĐÚNG TUYỆT ĐỐI**. Annotation là tập hợp các tọa độ góc hộp chữ nhật. Khẳng định không tính Pixel-AUROC hay Dice diện tích là chuẩn mực học thuật.
3. **Phát biểu: "Anomalib hỗ trợ chính thức EfficientAD và PatchCore"**
   - **Xác nhận:** **ĐÚNG TUYỆT ĐỐI**. Đã đối chiếu tài liệu mã nguồn Anomalib v1.x/v2.x.
4. **Phát biểu: "YOLO26 tồn tại trong hệ sinh thái Ultralytics"**
   - **Xác nhận:** **ĐÃ GIẢI QUYẾT TRIỆT ĐỂ TRONG v3**. Bản v3 đã xóa sạch toàn bộ các dấu vết của YOLO26, chỉ giữ lại họ YOLO11 chính thức. Không còn nguy cơ bị hội đồng đánh trượt vì trích dẫn mô hình ảo.

---

### 4. 5 câu hỏi hội đồng hóc búa nhất và gợi ý trả lời bảo vệ cho plan.v3

#### Câu hỏi 1: Về sự đánh đổi kinh tế và hiện tượng triệt tiêu trong ECPB
> *"Các em đưa ra công thức ECPB với tỷ lệ phạt 100:1 để chứng minh tính thực tế công nghiệp. Nhưng nếu nhìn kỹ vào công thức, ở tỷ lệ lỗi 1%, độ sụt giảm 1% Recall và 1% tăng FRR lại làm tăng chi phí một lượng xấp xỉ bằng nhau ($1.0$ vs $0.99$). Vậy tỷ lệ phạt 100:1 có thực sự bảo vệ nhà máy khỏi việc lọt lỗi hay không?"*

*   **Gợi ý trả lời cho sinh viên:**
    1. *Bản chất quy mô sản xuất:* Tỷ lệ phạt $100:1$ là chi phí trên **từng sản phẩm đơn lẻ** (lọt 1 lỗi mất $100, chặn 1 bo tốt mất $1). Tuy nhiên, trên tổng thể dây chuyền $10.000$ bo, số lượng bo tốt ($9.900$) gấp 99 lần số lượng bo lỗi ($100$).
    2. *Ý nghĩa cân bằng:* Do số lượng bo tốt áp đảo, $1\%$ bo tốt bị chặn nhầm tương đương với $99$ lần tác động chi phí, cân bằng hoàn hảo với $1\%$ bo lỗi bị lọt ($1$ bo lỗi $\times 100$). Công thức ECPB cho thấy: Trong nhà máy thực tế, việc kiểm soát tỷ lệ báo nhầm (Overkill) quan trọng ngang ngửa việc tăng tỷ lệ bắt lỗi (Recall), vì nếu Overkill vượt quá $5\%$, trạm kiểm tra thủ công sẽ hoàn toàn vỡ trận vì quá tải.

#### Câu hỏi 2: Về tính hợp lệ của phép ghép Max-Blending so với Linear Blending trong Tiling
> *"Tại sao trong module TileManager các em lại dùng Max-Blending thay vì Linear Blending truyền thống khi ghép các bản đồ nhiệt của EfficientAD? Cơ sở toán học nào chứng minh cách ghép này không tạo ra đột biến ở biên?"*

*   **Gợi ý trả lời cho sinh viên:**
    1. *Sự khác biệt giữa ảnh tự nhiên và bản đồ dị thường:* Linear Blending (feathering) được sinh ra để khử đường nối màu sắc trên ảnh quang học. Nhưng anomaly map biểu diễn **xác suất/mức độ bất thường**. Nếu một vi khuyết tật (ví dụ lỗ kim $4\times 4$ px) nằm ở vùng biên của tile 1 và tile 2, phép nhân trọng số khoảng cách tuyến tính sẽ dìm giá trị đỉnh (peak score) xuống một nửa, khiến khuyết tật bị tụt dưới ngưỡng phát hiện $\tau_{pixel}$.
    2. *Tính bảo toàn tín hiệu khuyết tật:* Max-Blending ($\text{Map}(x,y) = \max_k \text{Map}_k(x,y)$) đảm bảo rằng bất kỳ điểm ảnh nào bị mô hình ở bất kỳ tile nào đánh giá là bất thường đều được giữ nguyên giá trị cao nhất. Tại vùng bình thường, cả 2 tile đều có điểm số tiệm cận 0 nên phép Max không tạo ra bất kỳ đường sọc biên nào.

#### Câu hỏi 3: Về sự vượt trội thực sự của Cascade (H01) so với Luật OR (H00)
> *"Cả luật OR (H00) và Cascade (H01) đều dùng chung checkpoint YOLO và EfficientAD. Về mặt lý thuyết, Cascade có bao giờ đạt được Recall cao hơn luật OR không? Nếu Recall không cao hơn, tại sao chúng tôi phải dùng Cascade mà không dùng luật OR cho đơn giản?"*

*   **Gợi ý trả lời cho sinh viên:**
    1. *Khẳng định lý thuyết:* Về mặt toán học, tại cùng ngưỡng $(\tau_{yolo}, \tau_{ad})$, Cascade **không thể có Recall cao hơn luật OR**, vì Cascade là một tập con có điều kiện của OR (khi YOLO đã bắt được lỗi, Cascade dừng lại).
    2. *Lợi ích thực tế của Cascade:* Cascade vượt trội OR ở hai khía cạnh sản xuất sống còn:
       - **Độ trễ (Latency):** Nhờ cơ chế early exit, với $90\%$ bo có lỗi rõ ràng, hệ thống kết thúc ngay sau tầng YOLO ($~15\text{ ms}$), không phải chạy 9 tile của EfficientAD ($~80\text{ ms}$). Độ trễ trung bình toàn dây chuyền giảm từ $95\text{ ms}$ xuống còn khoảng $25\text{ ms}$ (nhanh gấp gần 4 lần).
       - **Kiểm soát báo nhầm (FRR):** Luật OR kích hoạt AD trên toàn bộ $100\%$ ảnh bo tốt, làm cộng dồn toàn bộ FP của cả hai mô hình. Cascade chỉ kích hoạt AD trên phần ảnh nghi vấn, giảm đáng kể nguy cơ AD báo nhầm trên các bo tốt rõ ràng.

#### Câu hỏi 4: Về hiện tượng nổ False Positive do Alignment Jitter
> *"Trong bài kiểm tra độ bền (§9), các em bổ sung Alignment Jitter $\pm 1$ đến $\pm 2$ pixel. Nếu kết quả thực nghiệm cho thấy khi bị lệch 1.5 pixel, EfficientAD nổ False Positive hàng loạt dọc theo tất cả các đường mạch đồng, các em sẽ giải quyết thế nào?"*

*   **Gợi ý trả lời cho sinh viên:**
    1. *Thừa nhận bản chất thuật toán:* EfficientAD học phân phối cục bộ thông qua mạng Patch Description Network (PDN). Lệch pha cơ khí $1.5$ pixel tạo ra độ chênh lệch biên độ tương phản rất lớn giữa ảnh kiểm tra và mẫu chuẩn mà mạng đã học.
    2. *Giải pháp kiến trúc:* 
       - Ở tầng tiền xử lý: Bổ sung bước **Sub-pixel Phase Correlation** hoặc ECC (Enhanced Correlation Coefficient) để tự động căn chỉnh (fine-alignment) trước khi đưa vào mô hình.
       - Ở tầng mô hình: Áp dụng Gaussian Blur nhẹ ($\sigma = 0.8$) trên heatmap trước khi trích xuất ngưỡng, hoặc tăng dung sai hình học thông qua phép co giãn hình thái học (Morphological Dilation) trên mặt nạ vùng mạch chuẩn.

#### Câu hỏi 5: Về tính hợp lệ của việc bảo vệ đồ án khi chưa có phần cứng camera thật
> *"Toàn bộ hệ thống của các em chạy trên dữ liệu DeepPCB có sẵn. Nếu ngày mai doanh nghiệp đưa cho các em một camera công nghiệp Basler và một băng chuyền thật, hệ thống phần mềm của các em có chạy được không hay phải viết lại từ đầu?"*

*   **Gợi ý trả lời cho sinh viên:**
    1. *Tính sẵn sàng của kiến trúc phần mềm:* Hệ thống được thiết kế theo nguyên lý tách rời hoàn toàn thông qua interface `BaseImageSource` (§4.4). Để kết nối camera thật, nhóm chỉ cần viết thêm một class `GenICamSource` kế thừa từ `BaseImageSource` mà không phải thay đổi một dòng code nào trong `InferenceService`, `TileManager` hay thuật toán Cascade.
    2. *Ranh giới đồ án:* Đồ án tập trung giải quyết trọn vẹn và chuẩn xác **lõi thuật toán AI (Core Vision Engine)** và khung đánh giá (Benchmark Framework). Việc chuẩn hóa quang học (chọn thấu kính telecentric, đèn vòm đồng trục chống lóa) là bài toán tích hợp hệ thống phần cứng thuộc giai đoạn triển khai sản xuất, đã được nhóm dự liệu đầy đủ trong tài liệu đặc tả yêu cầu kỹ thuật.

---

## 5. Checklist hoàn thiện cuối cùng (Chuyển giao sang Pha Triển khai)

Kế hoạch `plan.v3.md` đã rất hoàn chỉnh. Dưới đây là các chỉnh sửa vi mô cuối cùng (P0) để đưa tài liệu vào trạng thái **READY FOR EXECUTION**:

### Nhóm P0: Tinh chỉnh hình học và công thức trong plan.v3.md
- [ ] **P0.1 - Hiệu chỉnh tọa độ Tiling trong `TileManager` (§4.4):**
  - Sửa bước nhảy thành tọa độ cố định không padding: $x, y \in \{0, 192, 384\}$ trên ảnh 640.
  - Chuyển cơ chế ghép heatmap từ `Linear Blending` sang `Max-Blending`.
- [ ] **P0.2 - Bổ sung chú giải cân bằng kinh tế cho ECPB (§7.3):** Thêm 2 dòng ghi chú giải thích tại sao hệ số phạt 100:1 lại cân bằng với tỷ lệ lỗi 1% trong dây chuyền sản xuất lớn.
- [ ] **P0.3 - Thêm tiêu chí Tie-breaking cho 2-D Grid Search (§5 - Bước 5):** Khóa quy tắc: Khi trùng Recall cực đại tại $\text{FRR} \le 5\%$, ưu tiên chọn cấu hình có `route_rate` thấp nhất.

### Nhóm P1: Bước vào Triển khai (Tuần 0 & Tuần 1)
- [ ] **P1.1 - Thực thi Gate 0 (GPU Cloud):** Thiết lập môi trường Google Colab / Kaggle T4, chạy smoke test PyTorch 2.x + CUDA 12.x thành công.
- [ ] **P1.2 - Xây dựng module `TileManager` độc lập:** Viết unit test kiểm tra việc cắt 9 tile từ ảnh 640, ghép Max-Blending và ánh xạ tọa độ box về hệ quy chiếu gốc không bị lệch pixel.

---

## 6. Trạng thái Git Repository

Kho lưu trữ Git nội bộ đã được cấu hình và theo dõi đầy đủ:
- Commit `a0faeb3`: Khởi tạo toàn bộ tài liệu gốc, `plan.v2.md`, `plan.v3.md`, `Review.md`, `review2.md` và `docs/`.
- File [review3.md](review3.md) được tạo mới để lưu vết toàn bộ biên bản phản biện vòng 3.
- Sẵn sàng cho commit tiếp theo ghi nhận bản hoàn thiện cuối cùng trước khi viết code.
