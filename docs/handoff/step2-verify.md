# Xác minh VERIFIER — Step 2 (Loader, TileManager và tiền xử lý dùng chung)

Người thực hiện: VERIFIER (branch `step2/tests`).
Đối tượng: Step 2 của PLAN (Bước 2 "Xây loader, TileManager và tiền xử lý dùng chung")
do IMPLEMENTER làm ở `step2/impl`.
Nguyên tắc: kiểm chứng độc lập từ spec (PLAN §1.2, §3, §4.4, §5 Bước 2, §6, §11;
`docs/handoff/step2-contract.md`), không bị ảnh hưởng bởi cách implementer code.
Dataset (`$env:DATASET_ROOT`) chỉ đọc; không decode/mở/xem ảnh test (loader test chỉ metadata).

> Ghi chú về nguồn spec: `step2-contract.md` nằm ở repo gốc (chưa merge vào worktree này),
> được dùng làm hợp đồng bất biến. VERIFIER KHÔNG sửa nó; thấy sai/thiếu -> `questions-verify.md`.

---

## Giai đoạn A — Viết test từ spec (chưa merge impl)

Trạng thái: các test này được viết TRƯỚC khi merge code implementer. Test đỏ ở giai đoạn này là bình thường
(module `pcb_lab.inference.tiling`, `pcb_lab.inference.preprocessing`, `pcb_lab.data.samples`,
`pcb_lab.data.augment` chưa có trên `step2/tests`). Commit: `[codex] step2: tests <nội dung>`.

### A.0. Cơ sở hạ tầng test
- Giữ `tests/conftest.py` của Step 1 (thêm `../src` vào `sys.path`, marker `dataset` tự skip nếu thiếu
  `DATASET_ROOT`). Mở rộng thêm các hàm import module Step 2 **bên trong** test/fixture:
  `import_tile_manager`, `import_preprocessing`, `import_samples`, `import_augment`,
  và fixture `synthetic_dataset_root` tái dùng `helpers.dataset_builder.build_clean_fixture`.
- **Bỏ `tests/__init__.py`**: để `tests/` là namespace dir trên `sys.path` (không phải package),
  cho phép `from conftest import ...` như mọi test Step 1. `tests/helpers/__init__.py` giữ nguyên
  (cần cho `from helpers.dataset_builder import`).
- Dataset giả: 4 partition, mỗi partition 1 good + 2 defect (fixture mặc định), 6 lớp, 6 cặp KHÔNG rò rỉ.
  Dùng làm oracle cho số đếm loader (tính từ fixture, KHÔNG hard-code số dataset thật).

### A.1. `tests/test_tile_manager.py` (tính chất toán học, không dựa công thức trộn)
Bảo vệ plan §4.4 / §11.1 và contract (`docs/handoff/step2-contract.md` §inference/tiling.py):

| Test | Mục bảo vệ |
|---|---|
| `test_plan_640_gives_9_tiles` | plan(640,640) = 9 tile, x0/y0 ∈ {0,224,448}, overlap 32 |
| `test_plan_640_row_major` | thứ tự row-major (col tăng nhanh nhất) |
| `test_plan_count_formula[...]` | số tile mỗi chiều = ceil((D-256)/224)+1 nếu D>256, ngược lại 1; (256,256)/(300,300)/(700,500)/(1,1)/(640,300) |
| `test_plan_tiles_cover_every_pixel[...]` | hợp các tile phủ kín mọi pixel ảnh (không lõi trống) |
| `test_split_nonpadded_equals_original` | phần KHÔNG đệm của mỗi tile = pixel ảnh gốc |
| `test_split_padding_reflect_replicate[...]` | đệm reflect/replicate đúng bản chất (gradient cột, mod 16 để nằm uint8) |
| `test_split_padding_constant_is_uniform` | đệm constant đồng nhất (không nhiễu) |
| `test_stitch_constant_map_recovers_constant` | (a) map hằng c -> mọi pixel = c |
| `test_stitch_linear_function_recovered` | (b) f(x,y)=ax+by -> khôi phục f (sai số <= 1e-5), độc lập công thức trộn |
| `test_stitch_pixel_owned_by_single_tile` | (c) pixel đơn sở hữu = giá trị tile đó |
| `test_stitch_overlap_monotone_no_large_jump` | (d) vùng chồng lấn nằm trong [0,1], đơn điệu, bước nhảy kề <= 2/overlap |
| `test_stitch_dtype_float32_finite` | (e) float32, không NaN/inf (kể cả mép tile) |
| `test_boxes_to_global_offset` | box (10,10,50,50) ở tile x0=224,y0=0 -> (234,10,274,50), score giữ |
| `test_boxes_to_global_clip_and_score_kept` | box lố ảnh -> clip; score giữ |
| `test_boxes_to_global_drops_box_in_padding` | box hoàn toàn trong vùng đệm -> bỏ |
| `test_global_nms_merge_high_iou` | 2 box IoU>0.45 gộp còn 1, giữ score cao |
| `test_global_nms_threshold_strict_045_kept` | IoU ĐÚNG 0.45 -> GIỮ (nghiêm ngặt >), ca float64 nguyên |
| `test_global_nms_above_045_dropped` | IoU > 0.45 -> gộp |
| `test_global_nms_order_independent` | kết quả không phụ thuộc thứ tự đầu vào |
| `test_global_nms_empty_ok` | danh sách rỗng không lỗi |
| `test_tile_manager_rejects_non_256[...]` | tile_size != 256 (320/512/128) -> ValueError |

### A.2. `tests/test_preprocessing.py`
Bảo vệ contract §inference/preprocessing.py và plan §5 Bước 2, §4.4:

| Test | Mục bảo vệ |
|---|---|
| `test_to_canonical_gray_three_equal_channels` | L -> 3 kênh bằng nhau |
| `test_to_canonical_rgba_drops_alpha` | RGBA -> RGB |
| `test_to_canonical_red_stays_channel0` | RGB đỏ thuần ở kênh 0 (không đảo BGR) |
| `test_to_canonical_from_path` | chấp nhận đường dẫn file |
| `test_letterbox_640_scale1_pad0` | 640x640 -> scale 1.0, pad 0 |
| `test_unletterbox_roundtrip[...]` | round-trip lề thư (640/800x600/300x700) sai số <= 1px |
| `test_normalize_twice_raises` | normalize 2 lần -> ValueError |
| `test_resize_for_ad_shape_and_const_preserved` | resize_for_ad -> (img256, meta), shape 256x256 |
| `test_upsample_map_const_preserved_no_overshoot` | map hằng giữ, không overshoot, đúng shape |
| `test_prepare_input_meta_keys_and_jsonable` | meta JSON-được, đủ khoá, color_order=rgb, normalized=false |
| `test_prepare_input_ad_tile_shape` | ad_tile -> pixels [N,256,256,3], N=9 với ảnh 640 |
| `test_prepare_input_sha256_stable_and_mode_dependent` | sha256 ổn định; khác mode khác hash |
| `test_prep_preview_cli_matches_api` | `prep_preview.py` (CLI) trùng meta+sha256 với gọi thẳng `prepare_input` |

### A.3. `tests/test_loader.py` (dataset giả)
Bảo vệ contract §data/samples.py và plan §1.2, §5 Bước 2:

| Test | Mục bảo vệ |
|---|---|
| `test_yolo_train_set_counts` | YoloTrainSet = toàn bộ train (good+defect) |
| `test_ad_train_set_only_good_train` | AdTrainSet chỉ good train |
| `test_eval_set_each_image_once` | EvalSet mỗi ảnh đúng 1 lần |
| `test_test_partition_len_and_metadata_ok` | test + allow_test=False: len() + metadata OK |
| `test_test_partition_load_image_forbidden` | test + allow_test=False: load_image() -> PermissionError |
| `test_test_partition_allow_test_loads` | allow_test=True -> load_image() hoạt động |
| `test_development_selection_equals_fusion` | `development_selection` == `fusion` (cùng mẫu, cùng thứ tự) |
| `test_order_deterministic_two_iterations` | thứ tự tất định (2 lần duyệt) |
| `test_order_sorted_by_sample_id` | sort theo sample_id |
| `test_ad_train_iter_tiles_partition_and_count` | mọi tile thuộc train, khớp ảnh nguồn; num_tiles = ảnh * tiles_per_image |
| `test_ad_train_iter_tiles_deterministic` | iter_tiles tất định |

### A.4. `tests/test_augment.py` (phần đã duyệt)
Bảo vệ contract §data/augment.py và plan §5 Bước 2:

| Test | Mục bảo vệ |
|---|---|
| `test_augment_rejects_non_train[...]` | split != train -> ValueError |
| `test_augment_allows_train` | split 'train' được phép |
| `test_augment_deterministic_by_seed` | cùng input + seed -> cùng output |
| `test_augment_different_seed_may_differ` | đổi seed -> output có thể khác |
| `test_augment_no_inplace_mutation` | không sửa mảng đầu vào tại chỗ |
| `test_augment_boxes_stay_in_image` | box sau biến đổi vẫn trong ảnh |

> Lưu ý: contract KHÔNG pin tên symbol / signature của hàm augment. `import_augment` thử nhiều
> entry point (`apply_augment`/`augment`/`...`); test dùng adapter gọi linh hoạt. Nếu API merge vào
> khác hình thức, các test augment sẽ **skip** kèm lý do (sẽ chốt lại chính xác ở Giai đoạn B).

### A.5. `tests/test_integration_real.py` (marker `dataset`, oracle plan §1.2)
Số kỳ vọng là ORACLE từ plan §1.2 (KHÔNG lấy từ code impl). Bảo vệ:

| Test | Mục bảo vệ |
|---|---|
| `test_yolo_train_set_counts` | YoloTrainSet train = 1.792 (895 good + 897 defect) |
| `test_ad_train_set_good_only_and_num_tiles` | AdTrainSet 895 good; num_tiles = 895*9 = 8.055 |
| `test_calibration_count` | calibration 460 |
| `test_fusion_equals_development_selection` | fusion 306; development_selection == fusion |
| `test_test_partition_unique_and_forbidden` | test 440 (220/220), sample_id duy nhất, load_image -> PermissionError |
| `test_check_loaders_deterministic` | `check_loaders.py` 2 lần -> `loader_check.json` GIỐNG HỆT |
| `test_to_canonical_matches_pillow_independent` | 3 ảnh train đọc PIL độc lập == to_canonical (không dùng ảnh test) |

### A.6. Tiêu chí chưa kiểm được ở Giai đoạn A và lý do
- **Chạy thật toàn bộ (A.3/A.5) chưa thực thi**: dataset thật chỉ chạy ở Giai đoạn B sau khi merge impl;
  ở Giai đoạn A chỉ viết test, impl chưa merge, nên các test marker `dataset` chưa có kết quả. Số kỳ vọng
  giữ nguyên từ plan §1.2, sẽ ghi độ lệch (nếu có) vào Giai đoạn B.
- **Kiểm hành vi thực tế của impl** (stitch/boxes_to_global/nms/letterbox từ code): chưa chạy vì impl chưa
  merge vào `step2/tests`. Các test A.1-A.4 định nghĩa kỳ vọng tính chất, sẽ chạy ở Giai đoạn B.
- **Đọc/đánh giá code impl** (diff, hard-code, nuốt lỗi, glob, ghi dataset, logic trùng lặp API/CLI,
  dependency thừa, secret/commit): thuộc Giai đoạn B.2-B.4, chưa làm.
- **Nhật ký phá hoại & tự xác minh độc lập**: thuộc Giai đoạn B.5-B.6, chưa làm.
- **Xác nhận contract không đổi** (`git diff main -- docs/handoff/step2-contract.md` rỗng): Giai đoạn B.3.

### A.7. Cập nhật trạng thái
- Branch: `step2/tests` (worktree `../PCB_Lab_verify2`). Tái dùng `conftest.py`,
  `helpers/dataset_builder.py` từ Step 1 (checkout từ `step1/tests` — Step 1 tests chưa merge vào main
  khi worktree này được tạo).
- Đã tạo: `tests/test_tile_manager.py`, `tests/test_preprocessing.py`, `tests/test_loader.py`,
  `tests/test_augment.py`, `tests/test_integration_real.py`; mở rộng `tests/conftest.py`;
  `docs/handoff/step2-verify.md`.
- Chưa merge `step2/impl` (đúng quy trình: merge chỉ ở Giai đoạn B khi IMPLEMENTER báo xong).
- Chưa sửa bất kỳ code nào của implementer.

> **DỪNG Ở ĐÂY — BÁO NGƯỜI DÙNG.** Giai đoạn A đã hoàn tất (viết + commit tests). Chờ IMPLEMENTER báo
> "xong" để bắt đầu Giai đoạn B (merge, chạy, review, nhật ký phá hoại, tự kiểm chứng).
