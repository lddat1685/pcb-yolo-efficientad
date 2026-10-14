# Căn cứ, khác biệt nguồn và trạng thái xuất sơ đồ

Ngày: 2026-09-28. Một sơ đồ duy nhất: `PLAN_OVERVIEW.mmd`. Đây là bản đồ thiết kế,
không phải báo cáo hoàn thành triển khai hoặc kết quả benchmark. Chỉ tạo file trong
`docs/diagrams/`; không sửa nguồn, code, tests, không đổi nhánh và không merge.

Ưu tiên cho tác vụ này theo chỉ dẫn người dùng: khi plan mâu thuẫn với hợp đồng
step hoặc DECISIONS, sơ đồ theo hợp đồng/DECISIONS và ghi rõ tại đây. Quy tắc này
chỉ áp dụng cách biểu diễn sơ đồ; không sửa hoặc tự chốt lại protocol.

## N01 — Đường dẫn và phạm vi nguồn

`docs/plan.md` không tồn tại. File thực tế có đủ §1–§13 là `PLAN.md` ở gốc repo;
`docs/handoff/questions-verify.md`, mục đường dẫn plan, cũng ghi xác nhận của điều
phối về đường dẫn này. Đã đọc toàn bộ `PLAN.md`, `docs/DECISIONS.md`,
`docs/TRIAGE.md` và tất cả `step*-contract.md` hiện có: Step 2, 3, 4.
Không có Step 1/5–10 contract trong thư mục handoff ở lần kiểm kê này; không tự
invent giao diện cho các bước đó. Không đọc hay dùng `docs/archive/`.
Các tên review/plan cũ xuất hiện ở nút TREE1 chỉ để mô tả cây thư mục §4.1,
không được dùng làm nguồn mới.

## N02 — ECPB: 100:1 trong plan và trọng số chưa duyệt trong DECISIONS

- **Plan:** §7.2, §7.3 và thẻ UI §10.2 dùng `C_FN=100`, `C_FP=1`, mô phỏng
  `N_sim=10.000` với 1% defect/99% good; có lời giải thích liên hệ IPC.
- **DECISIONS R4/H2 và TRIAGE R4/H2:** đề xuất `C_FN=50×C_FP`, nhưng yêu cầu
  HUMAN/VERIFY theo chi phí và tỷ lệ lỗi thực tế, chưa chốt trọng số.
- **Đã vẽ:** COST giữ công thức tổng quát và mẫu mô phỏng 10.000/1%/99%; nút
  COSTPENDING nét đứt ghi giả định 50:1 còn chờ H2. Thẻ UI dùng ECPB với giả định
  được duyệt, không tự coi 50:1 hay 100:1 là trọng số chính thức. Không dùng lời
  khẳng định về IPC như bằng chứng đã xác minh.

## N03 — B05: loại vĩnh viễn hay placeholder

- **Plan:** §7.1 và §12 loại B05/YOLO26, kèm khẳng định về sự tồn tại của model.
- **DECISIONS R3/V1 và TRIAGE R3:** loại khỏi ma trận bắt buộc; giữ placeholder
  detector mở rộng, chỉ tái đưa khi VERIFY V1 xác nhận. TRIAGE ghi chưa kiểm chứng.
- **Đã vẽ:** B05 là placeholder nét đứt, ngoài danh sách bắt buộc; không có một
  model YOLO26 hoạt động, cũng không khẳng định model có hay không tồn tại ngoài
  nguồn đã cho. Baseline detector B01/B02 vẫn là YOLO11n/YOLO11s. Chưa tự đóng V1.

## N04 — Gate 0 cloud và smoke CPU

- **Plan:** §1.4, §5 Bước 0 và §12 yêu cầu GPU cloud CUDA 12.x, smoke-train
  3 epoch trước khi đếm lịch 4–6 tuần; local chỉ CLI/UI/EDA/audit.
- **Contracts Step 3/4:** cho phép smoke CPU một epoch; Step 3 khởi tạo từ kiến
  trúc và view nhỏ, Step 4 limit 4 good; output `runs/smoke`, không artifact chính thức.
- **Đã vẽ:** Gate 0 vẫn kiểm cloud cho train nặng/lộ trình. SMOKE là kiểm tra kỹ
  thuật CPU riêng theo contract, không được coi là đạt Gate 0 hay baseline.
  DECISIONS S5 yêu cầu ghi device thực; không mặc định máy local có CUDA.

## N05 — Số bước TileManager

- **Plan §4.4:** viết `(640−256)/224+1=3`, biểu thức số học không bằng 3.
- **Contract Step 2, mục đính chính:** nếu H lớn hơn 256 thì
  `ceil((H−256)/224)+1`; còn lại 1. Ảnh 640 tạo 3×3 tile với x/y=0,224,448,
  canvas 704, padding phải/dưới.
- **Đã vẽ:** TILE theo công thức đính chính, vẫn 9 tile và 8.055 tile train good.
  STITCH dùng Linear Blending, crop padding và global NMS IoU .45.

## N06 — diagnostic_both và AD ngoài bbox

- **Plan §3.4:** mô tả diagnostic_both chạy cả hai và dùng AD vùng ngoài box để
  đánh giá lỗi thứ hai.
- **DECISIONS R5 / TRIAGE R5:** giữ cascade đơn giản; diagnostic_both để đo bù
  trừ đa lỗi; background-only AD routing là optional/ablation.
- **Đã vẽ:** DIAG chạy cả YOLO/AD trên mọi ảnh để phân tích. BGDIAG ngoài bbox
  là nhánh nét đứt; không biến nó thành routing bắt buộc hay dùng ROI ground truth.
  Cascade YOLO-positive, kể cả FP, chỉ đi YEXIT→RESULT, không gọi AD (S1).

## N07 — Framework EfficientAD

- **Plan §4.2:** chọn anomalib cho EfficientAD; §5 Bước 4 có yêu cầu tài nguyên
  teacher/ImageNette khi dùng anomalib.
- **Contract Step 4, phạm vi:** cho phép anomalib **hoặc standalone** EfficientAD-S.
- **Đã vẽ:** PyTorch với hai lựa chọn được contract cho phép; không bắt buộc
  anomalib. ImageNette nằm ở nhánh điều kiện nét đứt, không tính vào 895 PCB good.
  Không đưa các chi tiết kiến trúc không có trong nguồn vào sơ đồ.

## N08 — Interface logic, interface cụ thể và hai tầng schema prediction

- **Plan §4.4:** `BaseAnomalyDetector.predict` mô tả tuple score/map; §8 dùng
  record đích có `checkpoint_hashes`, `tiling_params`, `detections.xyxy_original`.
- **Contract Step 4:** `EfficientAdResult` chứa score/map/**boxes**, `Detection`
  có `class_id=None`; Engine nhận RGB uint8 NHWC và trả float32 NHW. Raw JSONL
  Step 4 dùng `checkpoint_sha256`, `tile_params`, `boxes.xyxy`; raw Step 3 có
  `inference_params`, `detections`, `yolo_image_score`. Cả hai raw có `timing_ms=null`.
- **Đã vẽ:** interface cụ thể theo contract; giữ interface logic chung. RAWSCHEMA
  tách record đích §8; không tự đổi tên khóa contract và không invent một module
  chuyển schema. Raw dùng phân tích/calibration, không thay timing pipeline thật.

## N09 — prepare_input và khóa pixel test

- **Contract Step 4:** nhắc `prepare_input(image,"ad")`; **Step 2** chốt enum
  mode `{yolo,ad_tile,ad_resize}`. Sơ đồ dùng `ad_tile` cho nhánh native, theo giao
  diện công khai đã chốt ở Step 2; không thêm alias `ad` chưa được định nghĩa.
- **Plan §5 Bước 2:** nhắc loader test 440 ảnh; **Step 2** cho đọc len/metadata
  nhưng chặn pixel mặc định. Step 3/4 chặn test trong train/extraction.
  Sơ đồ chỉ mở nhánh TESTFINAL sau FREEZE ở Bước 7. Các số lượng test trong
  taxonomy/metrics/inventory là metadata và quy tắc, không là luồng inference sớm.

## N10 — Giá trị khóa được contract bổ sung cho plan

Plan nêu nhiều tham số ở mức khuyến nghị/chưa chốt. Sơ đồ dùng các giao tiếp đã
chốt trong contracts: YOLO conf_floor .001, iou .7, max_det 300, imgsz640,
agnostic_nms=false, half=false, selection=ultralytics_fitness; AD tile256/stride224,
map quantiles .90/.995, score=max, tau_pixel .5, morph_k3, min_area20 strict,
NMS .45 và reload tolerance 1e-5. Giá trị null vẫn được biểu diễn là chưa duyệt
và trainer phải từ chối; không lấy config/code implementation để tự chốt thay.
R8 sinh candidate boxes thuộc inference; Point-in-Box và GT thuộc evaluator.

## N11 — Giới hạn thống kê, giả thuyết và trạng thái mở

Plan §1.4/§7.6 nhấn mạnh bootstrap chỉ đo sampling variance nội bộ N=2, không
đại diện domain shift. DECISIONS R1 giữ worst-group/pair-level và quyết định
không LOGO-CV; phần tóm tắt dùng cách nói “bù đắp một phần”. Sơ đồ giữ giới hạn
sampling variance và không suy rộng thành bằng chứng tổng quát hóa.

H5 và lời kỳ vọng giảm FRR/latency trong §3.2/§7.4 được ghi là giả thuyết cần đo,
không phải kết quả. Luồng Boolean OR/cascade tuân §3.1–3.3 và DECISIONS S1;
không thêm bước AD cứu FP của YOLO. Các mục HUMAN H1–H3 của DECISIONS và
giả thuyết H1–H5 của plan là hai bộ ID khác nhau, được phân biệt. Không tự đóng
câu hỏi chi phí, lợi ích hybrid hay VERIFY. Mốc 95%/5% là mục tiêu kế hoạch.

## Mức gộp và cách đọc

409 mục được gộp trong 99 nút nội dung theo làn. Các nhóm trường của Sample,
PreparedInput, prediction, registry/run/artifact; taxonomy sáu lớp; recipes;
metrics/bảng benchmark; ablation/stress; từng màn hình; tests/checklist và các
mốc tuần dùng nhãn nhiều dòng. Các cổng và nhánh routing quan trọng được tách
nút để cạnh thể hiện rõ điều kiện.

Cạnh chỉ rõ dữ liệu, lời gọi hoặc điều kiện. Thứ tự làn là cách tổ chức nội dung,
không có nghĩa calibration diễn ra sau một lần inference sản phẩm đã khóa;
artifact ngưỡng quay lại core, UI/CLI gọi chung core/evaluator. Chi tiết nội bộ
không được nguồn chốt (ví dụ kiến trúc backbone/head YOLO) không được invent.
Nét đứt luôn dành cho tùy chọn, mở rộng hoặc điều kiện chưa chốt.

## Thử xuất và kiểm tra

Đã thử từ gốc repo, PowerShell 5.1:

```powershell
npx -y @mermaid-js/mermaid-cli -i docs/diagrams/PLAN_OVERVIEW.mmd -o docs/diagrams/export/PLAN_OVERVIEW.svg -w 3600 -b white
```

Lỗi thực tế: `npx` không được nhận diện là command. Node/npm/npx không được
tìm thấy trên PATH hoặc các đường dẫn Node.js thông dụng đã kiểm. Không cài
Node, Chromium/Puppeteer hay công cụ nặng để thay thế. Vì bước SVG không khởi
chạy được nên PNG không được tạo. Không tạo file SVG/PNG giả hoặc rỗng.

**Chưa kiểm cú pháp bằng công cụ** Mermaid và chưa xác minh bố cục sau render.
Đã tự rà một flowchart, nhãn quote, ID duy nhất, cạnh hợp lệ, subgraph/end cân
bằng, ký tự góc chỉ dùng thẻ HTML b/i/br hợp lệ, và đối chiếu nội dung Inventory
với từng nhãn node. `export/validation.json` ghi kết quả rà cấu trúc/bao phủ,
không được hiểu là Mermaid parser PASS. Lỗi command gốc ở `export/mmdc.log`.

Khi có sẵn mmdc/npx phù hợp, hai lệnh xuất dự kiến là:

```powershell
npx -y @mermaid-js/mermaid-cli -i docs/diagrams/PLAN_OVERVIEW.mmd -o docs/diagrams/export/PLAN_OVERVIEW.svg -w 3600 -b white
npx -y @mermaid-js/mermaid-cli -i docs/diagrams/PLAN_OVERVIEW.mmd -o docs/diagrams/export/PLAN_OVERVIEW.png -w 3600 -b white
```

Cần kiểm lại kích thước PNG thực tế tối thiểu 3000 px và độ đọc sau render;
`-w 3600` là yêu cầu viewport của lệnh dự kiến, chưa phải số đo artifact đã xuất.
