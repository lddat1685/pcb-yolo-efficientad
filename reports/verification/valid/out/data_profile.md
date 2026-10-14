# Hồ sơ dữ liệu PCB — kiểm toán và EDA

Kết quả kiểm toán: **PASS**. Dataset: `.\reports\verification\valid\dataset`.

Số ảnh lấy từ manifest; box/lớp lấy từ các dòng nhãn YOLO hợp lệ và được đối chiếu với manifest. Mẫu lỗi vẫn nằm trong tổng ảnh. Nhãn không đọc được không được tính như ảnh có 0 lỗi; xem `errors` và các check FAIL trong data_audit.json. Box không hợp lệ không vào thống kê kích thước.

## Partition và nhóm nguồn

| Partition | Ảnh | Good | Defect | Nhóm nguồn | % good |
| --- | --- | --- | --- | --- | --- |
| train | 1 | 0 | 1 | 1 | 0 |
| calibration | 1 | 0 | 1 | 1 | 0 |
| fusion | 1 | 0 | 1 | 1 | 0 |
| test | 1 | 0 | 1 | 1 | 0 |

Tổng: 4 ảnh; 0 good; 4 defect.

Giữ tên `fusion` của nguồn; vai trò dự kiến của dự án là `development_selection`. Không sửa protocol nguồn hoặc chia lại tập. `source_group` chưa phải design/bo vật lý đã xác minh.

| Partition | Nhóm nguồn | Ảnh | Good | Defect | Box hợp lệ |
| --- | --- | --- | --- | --- | --- |
| train | group_0 | 1 | 0 | 1 | 6 |
| calibration | group_1 | 1 | 0 | 1 | 6 |
| fusion | group_2 | 1 | 0 | 1 | 6 |
| test | group_3 | 1 | 0 | 1 | 6 |

## Lớp và số lỗi mỗi ảnh

Bảng test chỉ phục vụ đối chiếu số lượng annotation, không dùng chọn cấu hình/ngưỡng. Một ảnh có thể có nhiều lớp; tổng số ảnh theo lớp có thể vượt số ảnh defect.

| Partition | Lớp | Số box | Số ảnh chứa lớp |
| --- | --- | --- | --- |
| train | open_circuit | 1 | 1 |
| train | short | 1 | 1 |
| train | mouse_bite | 1 | 1 |
| train | spur | 1 | 1 |
| train | spurious_copper | 1 | 1 |
| train | pin_hole | 1 | 1 |
| calibration | open_circuit | 1 | 1 |
| calibration | short | 1 | 1 |
| calibration | mouse_bite | 1 | 1 |
| calibration | spur | 1 | 1 |
| calibration | spurious_copper | 1 | 1 |
| calibration | pin_hole | 1 | 1 |
| fusion | open_circuit | 1 | 1 |
| fusion | short | 1 | 1 |
| fusion | mouse_bite | 1 | 1 |
| fusion | spur | 1 | 1 |
| fusion | spurious_copper | 1 | 1 |
| fusion | pin_hole | 1 | 1 |
| test | open_circuit | 1 | 1 |
| test | short | 1 | 1 |
| test | mouse_bite | 1 | 1 |
| test | spur | 1 | 1 |
| test | spurious_copper | 1 | 1 |
| test | pin_hole | 1 | 1 |

| Partition | Box hợp lệ/ảnh có nhãn đọc được | Số ảnh |
| --- | --- | --- |
| train | 6 | 1 |
| calibration | 6 | 1 |
| fusion | 6 | 1 |
| test | 6 | 1 |

## Kích thước box trên train/calibration

Đơn vị px tại ảnh đã decode, diện tích px². Phân vị tuyến tính tại chỉ số `(n−1)×q`. Không dùng box test/fusion cho EDA kích thước hay chọn ảnh minh họa.

| Lớp | Đại lượng | N box | Min | P5 | P50 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open_circuit | width | 2 | 32 | 32 | 32 | 32 | 32 |
| open_circuit | height | 2 | 32 | 32 | 32 | 32 | 32 |
| open_circuit | short_side | 2 | 32 | 32 | 32 | 32 | 32 |
| open_circuit | area | 2 | 1024 | 1024 | 1024 | 1024 | 1024 |
| short | width | 2 | 32 | 32 | 32 | 32 | 32 |
| short | height | 2 | 32 | 32 | 32 | 32 | 32 |
| short | short_side | 2 | 32 | 32 | 32 | 32 | 32 |
| short | area | 2 | 1024 | 1024 | 1024 | 1024 | 1024 |
| mouse_bite | width | 2 | 32 | 32 | 32 | 32 | 32 |
| mouse_bite | height | 2 | 32 | 32 | 32 | 32 | 32 |
| mouse_bite | short_side | 2 | 32 | 32 | 32 | 32 | 32 |
| mouse_bite | area | 2 | 1024 | 1024 | 1024 | 1024 | 1024 |
| spur | width | 2 | 32 | 32 | 32 | 32 | 32 |
| spur | height | 2 | 32 | 32 | 32 | 32 | 32 |
| spur | short_side | 2 | 32 | 32 | 32 | 32 | 32 |
| spur | area | 2 | 1024 | 1024 | 1024 | 1024 | 1024 |
| spurious_copper | width | 2 | 32 | 32 | 32 | 32 | 32 |
| spurious_copper | height | 2 | 32 | 32 | 32 | 32 | 32 |
| spurious_copper | short_side | 2 | 32 | 32 | 32 | 32 | 32 |
| spurious_copper | area | 2 | 1024 | 1024 | 1024 | 1024 | 1024 |
| pin_hole | width | 2 | 32 | 32 | 32 | 32 | 32 |
| pin_hole | height | 2 | 32 | 32 | 32 | 32 | 32 |
| pin_hole | short_side | 2 | 32 | 32 | 32 | 32 | 32 |
| pin_hole | area | 2 | 1024 | 1024 | 1024 | 1024 | 1024 |

### pin_hole

Box nhỏ nhất theo diện tích: **32 × 32 px** (1024 px²), `calibration_defect` thuộc `calibration`.

| N box | Cạnh ngắn min | P5 | P50 | P95 | % cạnh <16 px | % cạnh <32 px |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 32 | 32 | 32 | 32 | 0 | 0 |

| Resize 640→256 | Min cạnh | P5 | P50 | P95 |
| --- | --- | --- | --- | --- |
| pin_hole | 12.8 | 12.8 | 12.8 | 12.8 |

### mouse_bite

Box nhỏ nhất theo diện tích: **32 × 32 px** (1024 px²), `calibration_defect` thuộc `calibration`.

| N box | Cạnh ngắn min | P5 | P50 | P95 | % cạnh <16 px | % cạnh <32 px |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 32 | 32 | 32 | 32 | 0 | 0 |

| Resize 640→256 | Min cạnh | P5 | P50 | P95 |
| --- | --- | --- | --- | --- |
| mouse_bite | 12.8 | 12.8 | 12.8 | 12.8 |

Ngưỡng mô tả chọn trước: `min(w,h) < 16` và `< 32` px (bất đẳng thức nghiêm ngặt). Nếu resize ảnh 640→256, các cạnh tương ứng <6.4 và <12.8 px. Đây là thước đo rủi ro mất chi tiết, không phải ngưỡng suy luận và không chứng minh lỗi biến mất. Tile native 256 giữ thang pixel nhưng có thể cắt box ở biên; Step 2 phải kiểm tra ánh xạ/overlap, Step 1 chưa chạy TileManager/model.

## Shortcut và giới hạn diễn giải

- 2/2 tên ảnh train/calibration chứa `_good` hoặc `_defect`. Tên file/ID/split không được làm feature mô hình.
- Pair template/test chia sẻ nội dung/layout; kiểm tra tách pair và nhóm trên mọi partition. Không thêm template bằng cách đoán tên hoặc suy số ảnh = 2×pair.
- Giữ nguyên 0 good có nền đồng nhất trong train/calibration. Độ đồng nhất không phải lý do loại mẫu; nguồn đã xác nhận normal.
- Dataset card mô tả good đã làm sạch và một phần lỗi bổ sung thủ công. Overlay chỉ kiểm tra tính hợp lý nhãn; không đủ xác nhận/loại trừ dấu vết tổng hợp hoặc shortcut học được.
- Không có pixel mask chuẩn, không báo pixel AUROC/AUPRO hoặc biến bbox thành ground truth segmentation.
- `pin_hole` khác `missing_hole` của PKU. Không gộp domain/schema.
- Số crop không phải số bo độc lập. Bootstrap nội bộ nhóm không chứng minh tổng quát hóa sang domain mới.

| Source mode train/calibration | Số ảnh |
| --- | --- |
| unknown | 2 |

| Bằng chứng good train/calibration | Số ảnh |
| --- | --- |

| Transform ghi trong manifest train/calibration | Số ảnh |
| --- | --- |
| null | 2 |

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
| shortcut_filename | WARN | 2 |

Release: 4/4 tệp khớp SHA-256. Số lỗi ghi nhận: 0. Chi tiết lỗi ở data_audit.json.

Ảnh test chỉ decode/hash trong audit, không xuất/xem. Overlay được tạo riêng bằng CLI hoặc `render_overlays`; xem `overlay_errors.json` và `overlay_index.json` nếu đã render. `run_audit` tự nó không render overlay.
