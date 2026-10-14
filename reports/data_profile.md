# Hồ sơ dữ liệu PCB — kiểm toán và EDA

Kết quả kiểm toán: **PASS**. Dataset: `..\DatasetVer4_Public`.

Số ảnh lấy từ manifest; box/lớp lấy từ các dòng nhãn YOLO hợp lệ và được đối chiếu với manifest. Mẫu lỗi vẫn nằm trong tổng ảnh. Nhãn không đọc được không được tính như ảnh có 0 lỗi; xem `errors` và các check FAIL trong data_audit.json. Box không hợp lệ không vào thống kê kích thước.

## Partition và nhóm nguồn

| Partition | Ảnh | Good | Defect | Nhóm nguồn | % good |
| --- | --- | --- | --- | --- | --- |
| train | 1792 | 895 | 897 | 5 | 49.9442 |
| calibration | 460 | 230 | 230 | 2 | 50 |
| fusion | 306 | 153 | 153 | 2 | 50 |
| test | 440 | 220 | 220 | 2 | 50 |

Tổng: 2998 ảnh; 1498 good; 1500 defect.

Giữ tên `fusion` của nguồn; vai trò dự kiến của dự án là `development_selection`. Không sửa protocol nguồn hoặc chia lại tập. `source_group` chưa phải design/bo vật lý đã xác minh.

| Partition | Nhóm nguồn | Ảnh | Good | Defect | Box hợp lệ |
| --- | --- | --- | --- | --- | --- |
| train | group00041 | 442 | 221 | 221 | 1642 |
| train | group12100 | 292 | 146 | 146 | 919 |
| train | group12300 | 196 | 98 | 98 | 600 |
| train | group20085 | 648 | 323 | 325 | 2428 |
| train | group77000 | 214 | 107 | 107 | 558 |
| calibration | group12000 | 28 | 14 | 14 | 93 |
| calibration | group13000 | 432 | 216 | 216 | 1451 |
| fusion | group50600 | 158 | 79 | 79 | 432 |
| fusion | group90100 | 148 | 74 | 74 | 483 |
| test | group44000 | 200 | 100 | 100 | 747 |
| test | group92000 | 240 | 120 | 120 | 660 |

## Lớp và số lỗi mỗi ảnh

Bảng test chỉ phục vụ đối chiếu số lượng annotation, không dùng chọn cấu hình/ngưỡng. Một ảnh có thể có nhiều lớp; tổng số ảnh theo lớp có thể vượt số ảnh defect.

| Partition | Lớp | Số box | Số ảnh chứa lớp |
| --- | --- | --- | --- |
| train | open_circuit | 1216 | 814 |
| train | short | 973 | 688 |
| train | mouse_bite | 1228 | 766 |
| train | spur | 981 | 698 |
| train | spurious_copper | 884 | 744 |
| train | pin_hole | 865 | 759 |
| calibration | open_circuit | 234 | 166 |
| calibration | short | 160 | 116 |
| calibration | mouse_bite | 339 | 212 |
| calibration | spur | 277 | 193 |
| calibration | spurious_copper | 269 | 202 |
| calibration | pin_hole | 265 | 212 |
| fusion | open_circuit | 194 | 150 |
| fusion | short | 158 | 122 |
| fusion | mouse_bite | 148 | 113 |
| fusion | spur | 139 | 112 |
| fusion | spurious_copper | 127 | 121 |
| fusion | pin_hole | 149 | 139 |
| test | open_circuit | 298 | 214 |
| test | short | 215 | 175 |
| test | mouse_bite | 250 | 178 |
| test | spur | 228 | 181 |
| test | spurious_copper | 194 | 174 |
| test | pin_hole | 222 | 205 |

| Partition | Box hợp lệ/ảnh có nhãn đọc được | Số ảnh |
| --- | --- | --- |
| train | 0 | 895 |
| train | 1 | 4 |
| train | 2 | 7 |
| train | 3 | 17 |
| train | 4 | 49 |
| train | 5 | 121 |
| train | 6 | 176 |
| train | 7 | 200 |
| train | 8 | 157 |
| train | 9 | 115 |
| train | 10 | 39 |
| train | 11 | 8 |
| train | 12 | 2 |
| train | 13 | 1 |
| train | 15 | 1 |
| calibration | 0 | 230 |
| calibration | 3 | 5 |
| calibration | 4 | 24 |
| calibration | 5 | 39 |
| calibration | 6 | 53 |
| calibration | 7 | 44 |
| calibration | 8 | 24 |
| calibration | 9 | 17 |
| calibration | 10 | 10 |
| calibration | 11 | 2 |
| calibration | 12 | 11 |
| calibration | 13 | 1 |
| fusion | 0 | 153 |
| fusion | 4 | 9 |
| fusion | 5 | 60 |
| fusion | 6 | 50 |
| fusion | 7 | 14 |
| fusion | 8 | 10 |
| fusion | 9 | 4 |
| fusion | 10 | 4 |
| fusion | 12 | 1 |
| fusion | 13 | 1 |
| test | 0 | 220 |
| test | 3 | 3 |
| test | 4 | 21 |
| test | 5 | 40 |
| test | 6 | 56 |
| test | 7 | 52 |
| test | 8 | 25 |
| test | 9 | 16 |
| test | 10 | 7 |

## Kích thước box trên train/calibration

Đơn vị px tại ảnh đã decode, diện tích px². Phân vị tuyến tính tại chỉ số `(n−1)×q`. Không dùng box test/fusion cho EDA kích thước hay chọn ảnh minh họa.

| Lớp | Đại lượng | N box | Min | P5 | P50 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open_circuit | width | 1450 | 22 | 29 | 37 | 64 | 167 |
| open_circuit | height | 1450 | 19 | 26 | 33 | 59 | 133 |
| open_circuit | short_side | 1450 | 19 | 25 | 31 | 48 | 92 |
| open_circuit | area | 1450 | 500 | 834.25 | 1224.5 | 3009 | 8924 |
| short | width | 1133 | 21 | 26 | 40 | 72.4 | 173 |
| short | height | 1133 | 20 | 25 | 35 | 61 | 152 |
| short | short_side | 1133 | 20 | 24 | 30 | 45 | 74 |
| short | area | 1133 | 693 | 910 | 1430 | 2987.8 | 8415 |
| mouse_bite | width | 1567 | 20 | 26 | 34 | 45 | 106 |
| mouse_bite | height | 1567 | 20 | 24 | 29 | 40 | 78 |
| mouse_bite | short_side | 1567 | 20 | 24 | 28 | 37 | 78 |
| mouse_bite | area | 1567 | 400 | 691.8 | 980 | 1634 | 8268 |
| spur | width | 1258 | 24 | 27 | 33.5 | 43 | 73 |
| spur | height | 1258 | 21 | 25 | 31 | 40 | 75 |
| spur | short_side | 1258 | 21 | 25 | 30 | 36 | 47 |
| spur | area | 1258 | 546 | 780 | 1023.5 | 1463.2 | 2773 |
| spurious_copper | width | 1153 | 24 | 28 | 35 | 53 | 78 |
| spurious_copper | height | 1153 | 22 | 27 | 35 | 52 | 92 |
| spurious_copper | short_side | 1153 | 22 | 26 | 33 | 46 | 71 |
| spurious_copper | area | 1153 | 528 | 781.8 | 1260 | 2507.4 | 6390 |
| pin_hole | width | 1130 | 21 | 26 | 33 | 49 | 72 |
| pin_hole | height | 1130 | 20 | 25 | 31 | 47 | 83 |
| pin_hole | short_side | 1130 | 20 | 24 | 30 | 42 | 60 |
| pin_hole | area | 1130 | 483 | 700 | 1064 | 1937.1 | 4565 |

### pin_hole

Box nhỏ nhất theo diện tích: **21 × 23 px** (483 px²), `deeppcb_20085006_defect` thuộc `train`.

| N box | Cạnh ngắn min | P5 | P50 | P95 | % cạnh <16 px | % cạnh <32 px |
| --- | --- | --- | --- | --- | --- | --- |
| 1130 | 20 | 24 | 30 | 42 | 0 | 63.5398 |

| Resize 640→256 | Min cạnh | P5 | P50 | P95 |
| --- | --- | --- | --- | --- |
| pin_hole | 8 | 9.6 | 12 | 16.8 |

### mouse_bite

Box nhỏ nhất theo diện tích: **20 × 20 px** (400 px²), `deeppcb_20085019_defect` thuộc `train`.

| N box | Cạnh ngắn min | P5 | P50 | P95 | % cạnh <16 px | % cạnh <32 px |
| --- | --- | --- | --- | --- | --- | --- |
| 1567 | 20 | 24 | 28 | 37 | 0 | 77.1538 |

| Resize 640→256 | Min cạnh | P5 | P50 | P95 |
| --- | --- | --- | --- | --- |
| mouse_bite | 8 | 9.6 | 11.2 | 14.8 |

Ngưỡng mô tả chọn trước: `min(w,h) < 16` và `< 32` px (bất đẳng thức nghiêm ngặt). Nếu resize ảnh 640→256, các cạnh tương ứng <6.4 và <12.8 px. Đây là thước đo rủi ro mất chi tiết, không phải ngưỡng suy luận và không chứng minh lỗi biến mất. Tile native 256 giữ thang pixel nhưng có thể cắt box ở biên; Step 2 phải kiểm tra ánh xạ/overlap, Step 1 chưa chạy TileManager/model.

## Shortcut và giới hạn diễn giải

- 2252/2252 tên ảnh train/calibration chứa `_good` hoặc `_defect`. Tên file/ID/split không được làm feature mô hình.
- Pair template/test chia sẻ nội dung/layout; kiểm tra tách pair và nhóm trên mọi partition. Không thêm template bằng cách đoán tên hoặc suy số ảnh = 2×pair.
- Giữ nguyên 1 good có nền đồng nhất trong train/calibration. Độ đồng nhất không phải lý do loại mẫu; nguồn đã xác nhận normal.
- Dataset card mô tả good đã làm sạch và một phần lỗi bổ sung thủ công. Overlay chỉ kiểm tra tính hợp lý nhãn; không đủ xác nhận/loại trừ dấu vết tổng hợp hoặc shortcut học được.
- Không có pixel mask chuẩn, không báo pixel AUROC/AUPRO hoặc biến bbox thành ground truth segmentation.
- `pin_hole` khác `missing_hole` của PKU. Không gộp domain/schema.
- Số crop không phải số bo độc lập. Bootstrap nội bộ nhóm không chứng minh tổng quát hóa sang domain mới.

| Source mode train/calibration | Số ảnh |
| --- | --- |
| L | 2224 |
| RGB | 28 |

| Bằng chứng good train/calibration | Số ảnh |
| --- | --- |
| author_checked_and_cleaned_template | 1125 |

| Transform ghi trong manifest train/calibration | Số ảnh |
| --- | --- |
| {"convert": "RGB", "resize": null} | 2252 |

## Kiểm tra và giới hạn lượt chạy

| Check | Trạng thái | Số vi phạm |
| --- | --- | --- |
| box_valid | PASS | 0 |
| class_id_valid | PASS | 0 |
| image_decode | PASS | 0 |
| image_metadata | PASS | 0 |
| image_sha256 | PASS | 0 |
| image_size_640 | PASS | 0 |
| label_defect_nonempty | PASS | 0 |
| label_good_empty | PASS | 0 |
| label_manifest_match | PASS | 0 |
| label_sha256 | PASS | 0 |
| leak_image_hash | PASS | 0 |
| leak_pair | PASS | 0 |
| leak_pixel_hash | PASS | 0 |
| leak_source_group | PASS | 0 |
| manifest_paths_exist | PASS | 0 |
| manifest_schema | PASS | 0 |
| manifest_unique_ids | PASS | 0 |
| partition_counts | PASS | 0 |
| pixel_sha256 | PASS | 0 |
| release_sha256 | PASS | 0 |
| shortcut_filename | WARN | 2252 |

Release: 18/18 tệp khớp SHA-256. Số lỗi ghi nhận: 0. Chi tiết lỗi ở data_audit.json.

Ảnh test chỉ decode/hash trong audit, không xuất/xem. Overlay được tạo riêng bằng CLI hoặc `render_overlays`; xem `overlay_errors.json` và `overlay_index.json` nếu đã render. `run_audit` tự nó không render overlay.

Overlay: 60 ảnh train/calibration; 0 lỗi; 0 cảnh báo thiếu mẫu. Chi tiết: overlay_errors.json, overlay_warnings.json và overlay_index.json.
