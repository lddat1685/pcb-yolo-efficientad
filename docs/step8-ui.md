# Giao diện kiểm thử PCB Lab

Chạy từ repository bằng PowerShell:

```powershell
$py = '.\.cache\step3-verifier-b\.venv\Scripts\python.exe'
& $py -m pip install 'streamlit==1.55.0' 'altair==6.0.0' 'pandas==2.3.3'
& $py scripts/run_ui.py
```

Mở `http://127.0.0.1:8501`. Đổi cổng bằng `--server.port=8518`. Runner dùng localhost và không tự mở cửa sổ trình duyệt. Với môi trường mới, cài package bằng `pip install -e '.[ui]'` và chuẩn bị các phụ thuộc torch/ultralytics/OpenCV của Bước 3–4. Extra `ui` chỉ bổ sung giao diện, không tải trọng số.

Mô hình đọc từ `artifacts/test/frozen_evaluation.json`, metadata `artifact.json` và các checkpoint của Bước 3–4. SHA256 được đối chiếu trước khi nạp; thiếu/sai artifact trả ERROR. Sáu cấu hình B01, B02, B03, H00, H01, H02 dùng chung ba adapter; H01_sec là điểm vận hành phụ đã khóa. CPU là backend demo hiện tại. Benchmark vẫn mở được khi chưa có checkpoint/dataset, miễn còn freeze và ba CSV.

## Sử dụng

- **Single Inspection:** upload PNG/JPEG, chọn gallery calibration/fusion, chụp webcam trong trình duyệt hoặc nhập URL HTTP(S) trỏ trực tiếp đến ảnh. Bấm **Chạy kiểm tra**; xem route, score, box, heatmap, thời gian và tải JSON/PNG. Thay overlay hoặc độ phủ giữ nguyên kết quả, không chạy lại model. Thay model/ảnh sẽ ẩn kết quả cũ cho tới lần chạy tiếp theo.
- **Gallery test:** bật chế độ xem sau đánh giá đã khóa. Kiểm tra quyền test, freeze, CSV và hash tập sample ID trước khi cho phép nguồn ảnh đọc pixel. Mặc định vẫn là calibration. Không thay ngưỡng hoặc ghi dữ liệu trở lại tập test.
- **Benchmark H00 vs H01:** chỉ đọc các báo cáo đã khóa. Chọn model để lọc bảng và biểu đồ. Mức giảm latency lấy từ CSV, không phải hằng số. Pareto xét Recall cao, FRR/ECPB/latency thấp; bảng nhóm/lớp giữ nguyên phạm vi số liệu nguồn. Downloads giữ nguyên CSV gốc.
- **Batch:** chọn 1–20 ảnh, chạy, xem số GOOD/DEFECT/ERROR và tỷ lệ AD thực sự chạy trên ảnh thành công. CSV/JSON kèm dòng lỗi. Có thể xem lại overlay từng ảnh. Nhãn chỉ lấy từ manifest; filename upload không được xem là ground truth.

Ảnh được chuyển sang RGB uint8, giới hạn 20 MiB và 4 triệu pixel/cạnh 4096 px. YOLO dùng letterbox và trả box về tọa độ gốc; EfficientAD tile-256/stride-224 trên ảnh nguyên kích thước rồi stitch lại đúng kích thước ảnh. Ảnh ngoài 640×640 được cảnh báo. Webcam/URL là snapshot, không phải pipeline video liên tục.

Heatmap dùng min/max từng ảnh chỉ để đổi màu; anomaly score vẫn là max của map gốc, không phải xác suất hay accuracy. Khi early-exit, AD score/map là null và màn hình ghi **Không chạy**. AD-positive không được gán một trong sáu lớp lỗi.

## CLI và kiểm thử

```powershell
& $py scripts/inspect_image.py --model H01 --sample-id deeppcb_12000038_good --split calibration
& $py scripts/inspect_image.py --model H00 --image 'samples/pcb.png'
& $py scripts/inspect_image.py --model B03 --url 'https://example.org/pcb.png'
& $py -m pytest tests/test_step8_ui.py -q -p no:cacheprovider --basetemp=.cache/step8-pytest
& $py scripts/verify_step8_parity.py
```

CLI trả JSON qua stdout, exit 0 khi có GOOD/DEFECT, exit 2 khi ERROR. Log thư viện đi qua stderr. Test parity thật dùng checkpoint hiện có và gallery calibration; so sánh chính xác score, box, decision, routing và hash heatmap giữa tiến trình CLI riêng và Streamlit AppTest. Run ID, timestamp và thời gian không so bằng nhau.

Cache model (`st.cache_resource`) khóa theo checkpoint SHA256, đường dẫn/stat checkpoint, stat artifact và device. Các cấu hình dùng cùng trọng số chia sẻ adapter, được bảo vệ bằng `RLock`. Không cache prediction của ảnh upload giữa người dùng; kết quả nằm trong session. Dataset và báo cáo chỉ đọc.

## Giới hạn số liệu

Latency hybrid trong CSV Bước 7 được tính bằng tổng thời gian các tầng đã đo; H01 bỏ AD ở ảnh early-exit. Live UI đo lời gọi adapter thực tế, nạp model và decode riêng, không bao gồm render. Không gọi số lịch sử là phép đo E2E UI. B03 ở UI/CLI dùng adapter CPU xử lý từng tile để giới hạn bộ nhớ; Bước 7 dùng batch tile. Không cam kết score bit-for-bit giữa hai backend thực thi này hoặc tái lập latency lịch sử.

Tập test chỉ có hai nhóm nguồn và không có mask chuẩn. Không suy rộng thành chứng nhận sản xuất, không báo pixel metrics. B01 có thể chiếm ưu thế Pareto so với hybrid; giao diện giữ nguyên kết quả này.

Tham khảo API đã dùng: [Streamlit cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
