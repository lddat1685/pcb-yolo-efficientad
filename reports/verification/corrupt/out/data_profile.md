# Hồ sơ dữ liệu PCB — kiểm toán và EDA

Kết quả kiểm toán: **FAIL**. Dataset: `.\reports\verification\corrupt\dataset`.

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
| calibration | group_1 | 1 | 0 | 1 | 0 |
| fusion | group_2 | 1 | 0 | 1 | 0 |
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
| calibration | open_circuit | 0 | 0 |
| calibration | short | 0 | 0 |
| calibration | mouse_bite | 0 | 0 |
| calibration | spur | 0 | 0 |
| calibration | spurious_copper | 0 | 0 |
| calibration | pin_hole | 0 | 0 |
| fusion | open_circuit | 0 | 0 |
| fusion | short | 0 | 0 |
| fusion | mouse_bite | 0 | 0 |
| fusion | spur | 0 | 0 |
| fusion | spurious_copper | 0 | 0 |
| fusion | pin_hole | 0 | 0 |
| test | open_circuit | 1 | 1 |
| test | short | 1 | 1 |
| test | mouse_bite | 1 | 1 |
| test | spur | 1 | 1 |
| test | spurious_copper | 1 | 1 |
| test | pin_hole | 1 | 1 |

| Partition | Box hợp lệ/ảnh có nhãn đọc được | Số ảnh |
| --- | --- | --- |
| train | 6 | 1 |
| calibration | 0 | 1 |
| test | 6 | 1 |

## Kích thước box trên train/calibration

Đơn vị px tại ảnh đã decode, diện tích px². Phân vị tuyến tính tại chỉ số `(n−1)×q`. Không dùng box test/fusion cho EDA kích thước hay chọn ảnh minh họa.

| Lớp | Đại lượng | N box | Min | P5 | P50 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open_circuit | width | 0 | N/A | N/A | N/A | N/A | N/A |
| open_circuit | height | 0 | N/A | N/A | N/A | N/A | N/A |
| open_circuit | short_side | 0 | N/A | N/A | N/A | N/A | N/A |
| open_circuit | area | 0 | N/A | N/A | N/A | N/A | N/A |
| short | width | 0 | N/A | N/A | N/A | N/A | N/A |
| short | height | 0 | N/A | N/A | N/A | N/A | N/A |
| short | short_side | 0 | N/A | N/A | N/A | N/A | N/A |
| short | area | 0 | N/A | N/A | N/A | N/A | N/A |
| mouse_bite | width | 0 | N/A | N/A | N/A | N/A | N/A |
| mouse_bite | height | 0 | N/A | N/A | N/A | N/A | N/A |
| mouse_bite | short_side | 0 | N/A | N/A | N/A | N/A | N/A |
| mouse_bite | area | 0 | N/A | N/A | N/A | N/A | N/A |
| spur | width | 0 | N/A | N/A | N/A | N/A | N/A |
| spur | height | 0 | N/A | N/A | N/A | N/A | N/A |
| spur | short_side | 0 | N/A | N/A | N/A | N/A | N/A |
| spur | area | 0 | N/A | N/A | N/A | N/A | N/A |
| spurious_copper | width | 0 | N/A | N/A | N/A | N/A | N/A |
| spurious_copper | height | 0 | N/A | N/A | N/A | N/A | N/A |
| spurious_copper | short_side | 0 | N/A | N/A | N/A | N/A | N/A |
| spurious_copper | area | 0 | N/A | N/A | N/A | N/A | N/A |
| pin_hole | width | 0 | N/A | N/A | N/A | N/A | N/A |
| pin_hole | height | 0 | N/A | N/A | N/A | N/A | N/A |
| pin_hole | short_side | 0 | N/A | N/A | N/A | N/A | N/A |
| pin_hole | area | 0 | N/A | N/A | N/A | N/A | N/A |

### pin_hole

Không có box hợp lệ để thống kê; các phân vị/tỷ lệ là N/A.

| N box | Cạnh ngắn min | P5 | P50 | P95 | % cạnh <16 px | % cạnh <32 px |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | N/A | N/A | N/A | N/A | N/A | N/A |

| Resize 640→256 | Min cạnh | P5 | P50 | P95 |
| --- | --- | --- | --- | --- |
| pin_hole | N/A | N/A | N/A | N/A |

### mouse_bite

Không có box hợp lệ để thống kê; các phân vị/tỷ lệ là N/A.

| N box | Cạnh ngắn min | P5 | P50 | P95 | % cạnh <16 px | % cạnh <32 px |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | N/A | N/A | N/A | N/A | N/A | N/A |

| Resize 640→256 | Min cạnh | P5 | P50 | P95 |
| --- | --- | --- | --- | --- |
| mouse_bite | N/A | N/A | N/A | N/A |

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
| box_valid | FAIL | 3 |
| class_id_valid | FAIL | 2 |
| image_decode | FAIL | 1 |
| image_metadata | FAIL | 1 |
| image_sha256 | FAIL | 1 |
| image_size_640 | FAIL | 1 |
| label_defect_nonempty | FAIL | 1 |
| label_good_empty | PASS | 0 |
| label_manifest_match | FAIL | 2 |
| label_sha256 | FAIL | 2 |
| leak_image_hash | PASS | 0 |
| leak_pair | PASS | 0 |
| leak_pixel_hash | FAIL | 1 |
| leak_source_group | PASS | 0 |
| manifest_paths_exist | FAIL | 1 |
| manifest_schema | PASS | 0 |
| manifest_unique_ids | PASS | 0 |
| partition_counts | PASS | 0 |
| pixel_sha256 | FAIL | 1 |
| release_sha256 | PASS | 0 |
| shortcut_filename | WARN | 2 |

Release: 4/4 tệp khớp SHA-256. Số lỗi ghi nhận: 17. Chi tiết lỗi ở data_audit.json.

Ảnh test chỉ decode/hash trong audit, không xuất/xem. Overlay được tạo riêng bằng CLI hoặc `render_overlays`; xem `overlay_errors.json` và `overlay_index.json` nếu đã render. `run_audit` tự nó không render overlay.
