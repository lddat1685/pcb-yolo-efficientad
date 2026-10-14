# Thẻ tập dữ liệu (Dataset Card) — DeepPCB cho đồ án

Protocol dự án: `deeppcb_cascade_holdout_v1` · Release nguồn đã khóa: 18/18 tệp xác minh.
Kiểm toán độc lập: `reports/data_audit.json` (**PASS**), EDA: `reports/data_profile.md`.

## 1. Nguồn gốc

- Dataset gốc DeepPCB của tác giả (taxonomy và ảnh tham khảo từ repo công khai của tác giả);
  bản local sử dụng trong đồ án: `..\DatasetVer4_Public` (pin commit ghi trong
  dataset card của nguồn).
- Release nguồn đã khóa gồm manifest canonical, schema lớp, split holdout, protocol,
  thống kê build và verification; phiên lập kế hoạch đã đối chiếu manifest, kiểm tra tồn tại
  ảnh/nhãn và xác minh SHA-256 của **18/18 tệp được khóa trong release** — không thiếu
  đường dẫn ảnh/nhãn.
- Ảnh good là template đã được tác giả làm sạch; một phần ảnh lỗi được bổ sung thủ công
  (nguồn công bố trong dataset card). Overlay kiểm tra tính hợp lý nhãn, không đủ xác nhận
  hay loại trừ dấu vết tổng hợp — xem `docs/LIMITATIONS.md`.
- Dataset gốc chỉ đọc; runner của dự án không sửa nội dung hay ghi lại release nguồn.

## 2. Quy mô và định dạng

- Tổng **2.998 ảnh 640×640** (1.498 good + 1.500 defect), RGB canonical (uint8, HWC).
- Mọi ảnh decode đầy đủ 640×640; hash canonical khớp manifest (check `image_decode`,
  `image_size_640`, `image_sha256` trong `data_audit.json` — đều PASS).
- Nhãn YOLO + manifest box khớp nhau (dung sai chuẩn hóa 1e-9, check
  `label_manifest_match` PASS); box hợp lệ nằm trong ảnh (`box_valid` PASS);
  nhãn good rỗng, nhãn defect không rỗng (hai check PASS).
- Không có pixel mask chuẩn (`pixel_metrics: null`,
  `pixel_mask_reason: "No verified pixel masks"`) — không đánh giá segmentation.

## 3. Sáu lớp khuyết tật

| ID YOLO | Nhãn chuẩn | Diễn giải | Bbox train | Bbox test |
|---|---|---|---:|---:|
| 0 | `open_circuit` | Đứt đường mạch | 1216 | 298 |
| 1 | `short` | Chập / nối tắt đường mạch | 973 | 215 |
| 2 | `mouse_bite` | Khuyết mép đường đồng | 1228 | 250 |
| 3 | `spur` | Gai / thừa đồng nhô ra | 981 | 228 |
| 4 | `spurious_copper` | Đồng thừa rời rạc | 884 | 194 |
| 5 | `pin_hole` | Lỗ nhỏ trong vùng đồng | 865 | 222 |

Một ảnh có thể chứa nhiều box và nhiều loại lỗi — không biến bài toán thành phân loại
một nhãn duy nhất mỗi ảnh. `pin_hole` không tương đương `missing_hole` của dataset PKU;
không gộp domain/schema. Lỗi nhỏ nhất theo diện tích: `mouse_bite` 20×20 px (400 px²),
`pin_hole` 21×23 px (483 px²) trên ảnh 640 — là cơ sở cho ablation tile-256 vs resize-256.

Phân bố chi tiết theo partition (số box / số ảnh chứa lớp) xem `reports/data_profile.md`.

## 4. Phân chia holdout canonical

| Partition | Good | Defect | Tổng | Nhóm nguồn | Vai trò trong dự án |
|---|---:|---:|---:|---:|---|
| `train` | 895 | 897 | 1792 | 5 | YOLO học cả hai; EfficientAD chỉ học 895 good |
| `calibration` | 230 | 230 | 460 | 2 | Validation, chọn checkpoint, chuẩn hóa AD, hiệu chỉnh ngưỡng |
| `fusion` (= `development_selection`) | 153 | 153 | 306 | 2 | So sánh / chọn cấu hình sau calibration |
| `test` | 220 | 220 | 440 | 2 | Đánh giá cuối sau khi khóa cấu hình |
| **Tổng** | **1498** | **1500** | **2998** | **11** | Không chia ngẫu nhiên lại theo ảnh |

- Tên partition `fusion` giữ nguyên từ manifest nguồn; dự án dùng alias
  `development_selection` cho vai trò (cascade cơ bản không cần fit fusion head).
  Thay đổi vai trò được ghi vào protocol mới `deeppcb_cascade_holdout_v1`.
- SHA-256 tập test (canonical, đã khóa): `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24`
  (440 ảnh).
- Kiểm tra rò rỉ trên **mọi cặp partition** (không chỉ train/test): disjoint
  `source_group`, pair-id, hash ảnh và hash pixel RGB (`leak_*` PASS, 6/6 cặp).

## 5. Hai nhóm nguồn của tập test

- Tập test gồm đúng 2 nhóm nguồn: **`group44000`** (200 ảnh: 100 good / 100 defect,
  747 box hợp lệ) và **`group92000`** (240 ảnh: 120 good / 120 defect, 660 box).
- Calibration (group12000 28 ảnh + group13000 432 ảnh) và fusion (group50600 158 ảnh +
  group90100 148 ảnh) mỗi tập cũng chỉ có 2 nhóm — xem bảng partition×nhóm trong
  `reports/data_profile.md`.
- `source_group` chỉ là nhóm nguồn; **chưa xác minh được thiết kế hay bo vật lý**
  (`source_group_is_not_verified_board_or_design: true`). Đơn vị kết quả là
  **ảnh/crop PCB**, chưa phải một bo nguyên vẹn.
- Hệ quả thống kê: test có **N=2 về cấp nhóm nguồn** — bootstrap pair-level chỉ phản ánh
  sampling variance nội bộ 2 bo, không đại diện domain shift. Mọi kết luận tổng quát hóa
  giới hạn trong chuyển giao giữa các thiết kế mạch trần tương đồng (xem
  `docs/LIMITATIONS.md`). Báo cáo `worst-group recall/FRR` theo từng nhóm (PLAN §7.6).

## 6. Cách dùng đúng theo quy tắc công bằng (PLAN §6)

1. Mọi model dùng cùng danh sách test canonical; không thêm template riêng, không bỏ ảnh lỗi.
2. Calibration chọn checkpoint/ngưỡng, fusion chọn cấu hình, test chỉ đánh giá cuối.
3. Cùng điều kiện đo tốc độ mỗi bảng (device/backend/precision/batch/timing).
4. Công bố input size và tile params (tile-256 qua TileManager vs resize-256 trong ablation).
5. Chỉ ảnh có nhãn chuẩn mới có accuracy — kèm dataset/split/N/artifact.
6. Lỗi inference không biến mất: báo error rate và số mẫu thành công.
7. Không gộp dataset khác domain vào cùng bảng.
