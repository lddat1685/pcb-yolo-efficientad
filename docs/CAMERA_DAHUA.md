# Dahua / iRAYPLE A3504MG100 qua GigE và MV Viewer

Ứng dụng dùng **IMV SDK đi kèm MV Viewer** để quét camera Ethernet, chụp từng
ảnh Mono8, chọn ROI rồi chạy pipeline YOLO/EfficientAD hiện có. Không cần cài thêm
gói Python camera. MV Viewer và driver/SDK của hãng được cài riêng trên **máy đang
nối camera**; chúng không nằm trong gói bàn giao và không do `setup_env.bat` cài.

Phần kết nối trực tiếp hỗ trợ Windows x64 + Python 64-bit. Khai báo ABI được đối
chiếu với `IMVApi.h` / `IMVDefines.h` của **SDK 2.4.1 Build20231102**.
Chưa kiểm thử trên camera vật lý A3504MG100; kiểm thử tự động dùng SDK mô phỏng.
Không coi kiểm thử phần mềm là xác nhận driver, firmware hoặc chất lượng ảnh thực.

## 1. Chuẩn bị trên máy nối camera

1. Chép project đầy đủ checkpoint sang máy đang cắm Ethernet camera.
2. Cài **MV Viewer x64 và SDK/driver GigE của Dahua/Huaray/iRAYPLE** từ nhà cung cấp
   camera hoặc [kho SDK chính thức Huaray](http://download.huaraytech.com/pub/sdk/).
   Bộ tham chiếu: `2.4.1/Windows/Base_ver/MVviewer_2.4.1_Build20231102.zip`.
   Chọn bản phù hợp Windows/firmware với nhà cung cấp; tránh nhầm SDK MindVision
   `mvsdk`, Hikvision MVS hoặc driver webcam. Giữ đầy đủ các DLL runtime phụ thuộc.
3. Trong MV Viewer, tìm camera và cấu hình địa chỉ IP camera/card mạng cùng subnet,
   không trùng IP thiết bị khác. Xác nhận xem được ảnh; dùng Mono8, chỉnh nét,
   ánh sáng và exposure. Không cần đổi IP khi đang xem được ảnh bình thường.
4. **Stop acquisition và Disconnect camera trong MV Viewer**, hoặc thoát MV Viewer,
   trước khi chụp trong project. Hai ứng dụng không được cùng chiếm quyền điều khiển.
5. Chạy `setup_env.bat` nếu chưa tạo môi trường. Chạy `run_camera_check.bat`.
   Kết quả `OK` kèm model/serial/IP nghĩa là đã liệt kê thiết bị, **chưa chứng minh
   đã nhận được ảnh**. `NO_CAMERA` trả mã 2; lỗi SDK trả mã 1.

SDK được tìm theo các thư mục cài phổ biến. Nếu không tự nhận, mở PowerShell tại
project rồi đặt một trong hai biến (thay placeholder bằng vị trí thật trên máy đó):

```powershell
$env:MVSDK_ROOT = '<thu-muc-SDK>'
# Hoặc chỉ rõ DLL x64; ưu tiên cao hơn MVSDK_ROOT:
$env:MVSDK_LIBRARY = '<thu-muc-SDK>/Runtime/x64/MVSDKmd.dll'
.\run_camera_check.bat
.\run_ui.bat
```

Biến đặt trong PowerShell chỉ áp dụng cho phiên đó và các launcher chạy từ phiên đó.
Muốn nhấp đúp launcher ở các lần sau, thêm biến qua Windows Environment Variables
cho tài khoản hiện tại. Không chép DLL riêng lẻ vào repo hoặc `.venv`.

**Camera đang ở máy khác:** cách dùng được hỗ trợ là chạy project ngay trên máy đó.
Trình duyệt trên máy này không chuyển camera GigE của máy khác thành webcam.
Chức năng này chưa cung cấp agent truyền ảnh qua mạng giữa hai máy.

## 2. Chụp và kiểm tra trên giao diện

1. Chạy `run_ui.bat` → **Single Inspection → Camera công nghiệp → GigE trực tiếp**.
2. Nhấn **Quét camera GigE**, chọn đúng model/serial/IP.
3. Mặc định chụp một ảnh bằng free-run. Bật **Dùng software trigger** nếu cần lệnh
   trigger phần mềm. Trigger phần cứng/dây chuyền và streaming liên tục chưa hỗ trợ.
4. Giữ exposure hiện tại, hoặc bật thủ công và nhập µs trong khoảng camera hỗ trợ.
   Timeout 100–10000 ms cần dài hơn exposure và thời gian truyền ảnh.
5. Nhấn **Chụp ảnh camera**, kiểm tra ảnh và có thể tải **ảnh camera gốc PNG**.
6. Chọn ROI (X, Y, rộng, cao) nếu cần, rồi **Chạy kiểm tra**. Đổi overlay/ROI
   chỉ xử lý lại snapshot đã chụp; muốn ảnh mới phải nhấn Chụp ảnh camera.

Mỗi lần chụp, ứng dụng mở camera, tạm dùng Continuous + Mono8 + FrameStart và chế độ
trigger đã chọn, sao chép ảnh khỏi buffer SDK, trả buffer và đóng kết nối. Các giá
trị tính năng mà ứng dụng thay đổi được đọc trước và khôi phục sau chụp; không ghi
UserSet, không đổi IP/firmware. Khi mất kết nối hoặc khôi phục thất bại, giao diện
báo lỗi; kiểm tra lại thông số trong MV Viewer trước lần dùng tiếp theo.

### Ảnh 5 MP và tọa độ

Ảnh toàn cảm biến A3504MG100 có thể vượt giới hạn 4 triệu pixel của pipeline hiện
tại. Nguồn camera nhận tối đa 16 triệu pixel/cạnh 8192, sau đó **crop ROI và thu nhỏ
có ghi nhận tỷ lệ** về giới hạn pipeline; cạnh mặc định 1280. Không phóng lớn ảnh.
Mono8 được nhân thành ba kênh RGB cho mô hình.

Box và heatmap trên UI/JSON thuộc **ảnh đưa vào AI sau ROI/resize**. JSON có
`sensor_size`, `roi_xywh`, `inspection_size`, `sensor_to_inspection_scale`, SHA256
ảnh và metadata camera. Quy đổi một điểm về ảnh camera:
`sensor_x = inspection_x / scale_x + roi_x`, tương tự cho Y.
Kích thước này là ảnh SDK trả về; nếu đã đặt ROI/binning trong MV Viewer thì không
bao gồm offset trên cảm biến vật lý. Nên reset ROI phần cứng trong MV Viewer khi
cần hệ tọa độ toàn cảm biến.

Thu nhỏ toàn ảnh có thể làm mất khuyết tật nhỏ. ROI 640×640 giữ nguyên pixel là điểm
bắt đầu phù hợp để thử. Mô hình hiện tại học trên PCB trần DeepPCB; cần ảnh có nhãn
trên camera/ánh sáng thực tế để đánh giá, điều chỉnh hoặc huấn luyện lại. Các ngưỡng
và benchmark đã khóa không được thay đổi bởi chức năng camera.

## 3. Dùng ảnh lưu từ MV Viewer khi chưa có SDK trên máy chạy project

Chọn **Camera công nghiệp → Ảnh lưu từ MV Viewer** rồi tải PNG/BMP/TIFF/JPEG
Mono8 hoặc RGB8. Định dạng 10/12/16-bit không tự chuyển sang 8-bit; lưu Mono8 từ
MV Viewer trước. Luồng ROI và kiểm tra giống ảnh chụp trực tiếp. Có thể chuyển ảnh
này từ máy camera sang máy đang chạy project; ảnh không có nhãn chuẩn tự động.

## 4. Kiểm tra chụp bằng dòng lệnh

```powershell
.\.venv\Scripts\python.exe scripts/camera_probe.py
# Copy đúng key từ danh sách ở trên; đường dẫn đầu ra phải chưa tồn tại:
.\.venv\Scripts\python.exe scripts/camera_probe.py --capture '<camera-key>' --software-trigger --output exports/camera/first.png
```

Chụp thành công tạo PNG gốc và JSON cạnh ảnh, trả `CAPTURED`. Có thể dùng ảnh đó
trong nguồn **Ảnh lưu từ MV Viewer**. Công cụ không tự chọn camera đầu tiên khi có
nhiều thiết bị và không ghi đè ảnh cũ.

| Triệu chứng | Cách kiểm tra |
|---|---|
| Chưa tìm thấy SDK | Cài MV Viewer + SDK x64; chỉ đúng `MVSDK_ROOT`/`MVSDK_LIBRARY`. |
| Không nạp DLL / WinError 126 hoặc 193 | Giữ đủ runtime DLL, đúng x64 và Python x64; sửa/cài lại SDK chính thức. |
| Không thấy camera / mã -107 | Kiểm tra link Ethernet, nguồn camera, card mạng và subnet trong MV Viewer. |
| Open/Access lỗi | Ngắt kết nối MV Viewer hoặc ứng dụng khác đang mở camera. |
| Timeout -116 | Thử free-run trong MV Viewer, kiểm tra trigger/exposure, packet loss, driver GigE và firewall của máy camera. |
| Feature không hỗ trợ | Ghi lại lỗi IMV và phiên bản SDK/firmware; thử lưu Mono8 bằng MV Viewer rồi nhập ảnh. |

Không tắt firewall toàn hệ thống để thử; kiểm tra quy tắc mạng/driver theo hướng dẫn
SDK trên card Ethernet dùng cho camera. Chức năng hiện tại không dùng RTSP hay
`cv2.VideoCapture` để nhận camera GigE công nghiệp.
