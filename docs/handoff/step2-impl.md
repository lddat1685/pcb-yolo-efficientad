# Bàn giao IMPLEMENTER — Step 2

## Phase 0

### Tóm tắt, căn cứ và tiêu chí hoàn thành

- Làm việc ở `..\PCB_Lab_impl`, branch `step2/impl`, base `84e2c02`. Workspace mặc định của phiên vẫn là worktree `main`; không sửa ở đó hoặc worktree `step2/tests`.
- Không có AGENTS.md trong worktree/cha. Đã đọc `PLAN.md` §1.2, §3, §4.4, §5 Bước 2, §6, §9, §11; DECISIONS/TRIAGE; handoff Step 1, `reports/data_profile.md`, code `data/manifest.py`. Không dùng archive. Repo vẫn có `PLAN.md` ở gốc, không có `docs/plan.md`.
- Contract chưa có trong worktree impl; đọc bản người điều phối tại `.\docs\handoff\step2-contract.md`, SHA-256 ban đầu `3e1fd25a0f07377ef1dc05d2091cc9dc15210b64e5650ad53c3cc167c7c5ce9a`. Không sao chép/sửa hoặc commit contract. Đọc các escape Markdown theo nghĩa hiển thị.
- Mục tiêu Bước 2: chuẩn hóa Sample/loader và một đường tiền xử lý CPU dùng chung cho API/CLI; bảo toàn split, màu, tọa độ, provenance; hỗ trợ letterbox YOLO, resize AD và tile native 256. TileManager phải ghép map có tính chất toán học đúng và gộp box theo Global NMS.
- Đầu vào: metadata canonical qua adapter Step 1, pixel RGB của train/phân vùng phát triển được phép, cấu hình có phiên bản và seed cho augment. Metadata test được đọc/đếm; không decode, xem hoặc xuất ảnh test trong công việc này.
- Đầu ra: các module data/inference được giao; `configs/preprocessing.yaml`; CLI kiểm một batch mỗi loader và preview metadata/hash; `reports/loader_check.json` tất định, bằng chứng kiểm chứng và handoff này.
- Hoàn thành khi loader giữ đúng good/train cho AD, canonical RGB/uint8/HWC và chuẩn hóa đúng một lần; plan/split/stitch/box/NMS đúng trên 640×640 và kích thước lạ; cùng ảnh qua API/CLI có pixel/hash/tọa độ nhất quán; augment đã được người dùng duyệt và kiểm tra; đã chạy thật, ghi cả lỗi nếu có. Không có model, threshold, benchmark/UI, protocol mới hoặc Gate 0.

### Schema và bằng chứng Step 1

- Tái dùng `load_dataset`, `load_manifest`, `dataset_path`, `resolve_dataset_root` và guard output của Step 1; không parse JSONL lần nữa trong module mới.
- Manifest dùng `split`, `pair_ids` list, `is_defect` bool, width/height, image, sha256, mask_status và `boxes[{class_id,class_name,xyxy,...}]`; metadata không chứa danh tính bo/design đã xác minh. Canonical đã là RGB PNG.
- Đọc metadata bằng adapter: mọi ảnh có kích thước 640×640. `deeppcb_20085159_good` có ba pair alias `20085159`, `20085167`, `20085168`; không được mất quan hệ pair khi tạo Sample.
- Step 1 EDA train/calibration: source mode L ở 2224 ảnh, RGB ở 28 ảnh; transform canonical là RGB, không resize. L phải nhân bản kênh, không đảo RGB/BGR.
- Pin_hole có cạnh ngắn min 20 px, P5/50/95 = 24/30/42; mouse_bite min 20 px, P5/50/95 = 24/28/37; spur min cạnh 21 px. Đây là kích thước bbox, không phải kích thước cấu trúc lỗi thật. Resize có rủi ro làm mất chi tiết dù bbox còn tồn tại.
- Biến môi trường DATASET_ROOT trong các process tool hiện chưa có giá trị (dù người dùng đã đặt ở terminal). Adapter vẫn xác định đúng `..\DatasetVer4_Public` từ YAML. Khi chạy kiểm chứng sẽ đặt biến này rõ trong process lệnh; không sửa thiết lập hệ thống hoặc dataset.

### Ba quyết định mở và rủi ro

| Quyết định | Hướng áp dụng / trạng thái | Lý do và rủi ro |
| --- | --- | --- |
| pad_mode mặc định reflect | Dùng mặc định reflect đúng contract; hỗ trợ replicate và constant. Chỉ pad phải/dưới, không đổi tọa độ gốc. | Tránh dải hằng nhân tạo lớn ở biên, nhưng reflect có thể phản chiếu đường đồng thành cấu trúc không có thật. Phần pad không thuộc kết quả stitch; ảnh có chiều 1 cần xử lý biên rõ ràng. Chưa chứng minh mode này tốt nhất cho AD. |
| AD train dùng mọi tile, 9 tile/ảnh 640×640 | Tính số tile bằng TileManager.plan theo công thức ceil đã đính chính, không hard-code 9. Chỉ good train; mỗi tile giữ Sample/partition của ảnh nguồn. | Bao phủ native pixel giúp giữ lỗi nhỏ khi suy luận; 9 tile tăng chi phí và tương quan giữa mẫu, tile biên có pad. Num tiles không phải số ảnh/bo độc lập. |
| recipe aug_v1 | Đề xuất bên dưới; **chờ người dùng duyệt, chưa viết augment.py hoặc áp augmentation**. | Cần recipe nhẹ, seed tái lập, giữ cấu trúc lỗi nhỏ và không trộn biến dạng stress test vào train. |

### Recipe aug_v1 đề xuất để duyệt

1. Chỉ partition `train`; seed tạo RNG cục bộ, không đổi RNG toàn cục.
2. Chọn đều `k ∈ {0,1,2,3}`, xoay `k×90°` ngược chiều kim đồng hồ bằng hoán vị pixel; áp cùng phép lên box.
3. Sau xoay, lật ngang với xác suất 0.5; ảnh/bbox cùng biến đổi.
4. Không nội suy, crop, resize hoặc đổi scale; không brightness/contrast/color jitter, blur, noise, cutout, mosaic, MixUp, elastic, morphology. Toàn bộ pixel/diện tích box được giữ; W/H có thể hoán đổi với ảnh không vuông. Không sửa mảng/box input tại chỗ.
5. Chọn 8 phép đối xứng hình học đồng xác suất; giữ chính xác hình thái pin_hole/mouse_bite/spur. Rủi ro: phân phối hướng/handedness layout thay đổi; không bảo đảm tốt hơn train không augment. Các thay đổi photometric/alignment thuộc recipe/ablation khác có tên, không tự thêm.

Đã gửi yêu cầu duyệt qua câu hỏi bất đồng bộ. API bổ sung dự kiến: `apply_augmentation(image, boxes, *, partition, seed, recipe_id="aug_v1")`; tên callable này chưa do contract khóa, sẽ ghi rõ kiểu trả về sau khi recipe được duyệt.

### Mơ hồ và giả định

- **Chặn Sample/loader:** contract có `Sample.pair_id` số ít nhưng nguồn có nhiều `pair_ids`, chưa chốt kiểu/ý nghĩa. Đề xuất `pair_id: str` là alias nhỏ nhất theo từ điển và thêm `pair_ids: tuple[str,...]` đầy đủ; chờ người dùng chốt trước khi làm phần phụ thuộc. Ghi `questions-impl.md`.
- Contract đính chính công thức ceil cho tile được ưu tiên hơn phép tính sai trong PLAN; 640 có origins 0/224/448, canvas704, pad64 phía phải/dưới. `tile_size !=256` phải raise ValueError.
- Box dùng tọa độ biên liên tục xyxy (x2/y2 có thể bằng W/H), không cộng/trừ một pixel. NMS class-agnostic, giữ tie theo thứ tự input; chỉ suppress khi IoU > ngưỡng.
- Stitch dự kiến feather tuyến tính từ rìa overlap về phía tâm, chuẩn hóa tổng trọng số; rìa canvas không có láng giềng giữ trọng số dương. Kiểm cả map hằng, hàm tuyến tính theo tọa độ toàn ảnh, pixel chỉ thuộc một tile và crossfade giữa map tile khác nhau để phát hiện bước nhảy.
- `to_canonical` nhận uint8 array/PIL/path; L nhân ba kênh; LA/RGBA bỏ alpha như Pillow convert RGB, không composite lên nền mới; không tự xoay EXIF. Mảng float bị từ chối để không suy đoán đã normalize hay chưa.
- Normalize dùng `(uint8/255 - mean)/std` float32, chỉ nhận uint8; dữ liệu normalize lần đầu là float32 nên lần hai raise ValueError. Mean/std và trạng thái normalized được ghi meta; không tự normalize trong prepare_input (contract yêu cầu pixels uint8).
- Meta hình học giữ các field contract và thêm version/hash; PreparedInput.meta là dict JSON được. Hash cấu hình từ JSON chuẩn hóa có defaults; hash PreparedInput sẽ ràng buộc pixel/shape/dtype/meta, không phụ thuộc đường dẫn input/out hay thời gian.
- Letterbox/resize dùng bilinear, letterbox pad114 cân hai phía (nếu lẻ phần dư về phải/dưới). Upsample map bilinear float32, không overshoot. Các tham số thực dùng được ghi meta.
- Metadata test lấy được qua Dataset nhưng Sample.load_image bị chặn mặc định; check_loaders chỉ kiểm metadata/gating cho test, không bật allow_test để decode. Preview đường dẫn canonical sẽ kiểm membership trước khi mở ảnh; ảnh ngoài dataset dùng canonical conversion thông thường.
- `AdTrainSet.tiles_per_image` suy từ kế hoạch của metadata mọi good train. Nếu dataset khác có số tile/ảnh không đồng nhất, không báo num_tiles bằng công thức sai: raise ValueError với thông báo rõ, giữ contract num_tiles=n_samples×tiles_per_image. Loader canonical hiện đồng nhất.

### File dự định tạo/sửa

- Tạo `src/pcb_lab/data/{samples,augment}.py`, `src/pcb_lab/inference/{__init__,preprocessing,tiling}.py`, `configs/preprocessing.yaml`, `scripts/{check_loaders,prep_preview}.py`.
- Thêm `requirements/step2.txt`; cập nhật dependencies `pyproject.toml` để khai báo NumPy cho API; không sửa manifest adapter trừ khi bắt buộc. NumPy dùng array, tile/map/vectorized blending/NMS, không framework nặng; khóa phiên bản sau khi cài/chạy thật.
- Ghi `docs/handoff/{step2-impl,questions-impl}.md`, `reports/loader_check.json` và bằng chứng kiểm chứng text. Dùng `reports/verification/` đã ignored cho fixture, preview, script tự kiểm chứng tạm; không sửa tests hoặc cấu hình pytest.
- Không sửa PLAN, DECISIONS, TRIAGE hoặc contract. Mỗi commit kiểm git status/staged paths, không stage dataset/ảnh/key/token; không merge/push/rebase/đụng branch khác.

## Phase 1

Phase 0 đã commit trước code (`2335efe`). Sau đó người dùng duyệt **"Duyệt recipe hình học này"**, chốt **"Chốt chuỗi đại diện + tuple đầy đủ"** và yêu cầu tiếp tục. Trạng thái chờ trong phần Phase 0 là lịch sử đề xuất; hai câu hỏi nay đã đóng ở questions-impl.md, contract giữ nguyên.

- `Sample.pair_id: str` là min(pair_ids); `Sample.pair_ids: tuple[str,...]` giữ đầy đủ alias. Không dùng ID đại diện để thay kiểm tra toàn bộ quan hệ pair.
- `apply_augmentation(image, boxes, *, partition, seed, recipe_id="aug_v1")` trả `(np.ndarray uint8 HWC, tuple[Box,...], meta dict JSON được)`. Áp đúng recipe được duyệt; thứ tự RNG k rồi flip; không tự áp augment vào eval/prepare_input. Recipe khai báo và khóa trong preprocessing.yaml.
- NumPy **2.4.3** đã cài và chạy thật với Python **3.14.7**, Pillow **12.1.1**, PyYAML **6.0.3**, setuptools **80.10.2** trong `.venv` riêng của worktree impl. Thêm NumPy cho tile/map/NMS/augmentation; pyproject khai báo dependency để API editable install đầy đủ. Không sửa Step 1 manifest adapter.
- Đã chạy check_loaders lần đầu: PASS, errors=[]; các kiểm chứng bổ sung đang thực hiện ở Phase 2.

### Đã triển khai và lựa chọn cụ thể

- Sample/Box là frozen dataclass; boxes/xyxy/pair_ids chuyển thành tuple để không giữ list mutable của caller. Loader lấy thứ tự lexical sample_id từ adapter, kiểm membership với holdout, không decode khi khởi tạo/len/duyệt metadata. Sai metadata không được biến thành good hoặc bỏ mẫu im lặng.
- YoloTrainSet lấy toàn bộ train; AdTrainSet lọc good train rồi tính kế hoạch từ kích thước metadata; EvalSet mỗi canonical một lần. Alias development_selection trả cùng Sample.partition=`fusion` và cùng thứ tự.
- Test gate nằm ở Sample.load_image và đường đọc path canonical của preprocessing (kiểm manifest + holdout trước Image.open); allow_test=False mặc định. Hỗ trợ allow_test=True theo contract nhưng không bật cờ đó trên dataset thật trong phiên này. `check_loaders` chỉ đọc metadata test, ghi `metadata_only=true`, shape/dtype/min/max/hash=null, batch_size=0 và access_blocked=true.
- RGB conversion giữ nội dung màu, bỏ alpha, nhân L/LA thành ba kênh; không sửa input. Normalize trả float32/meta, từ chối float input để chặn áp hai lần. Letterbox dùng scale chung, bilinear, pad114; phép nghịch clip bbox vào ảnh gốc. AD resize dùng bilinear, meta có scale_xy vì ảnh không vuông có hai tỷ lệ khác nhau.
- TileManager: size256/stride224, pad phải/dưới; reflect mặc định, replicate dùng edge, constant dùng **0**. NumPy reflect trên chiều dài 1 giữ giá trị duy nhất; đã kiểm 1×1, 1×513 và 513×1. Kế hoạch sử dụng integer ceil, không làm tròn số tile xuống hoặc dồn tile cuối về origin khác.
- Feather mỗi trục giảm tuyến tính từ vùng tâm ra rìa overlap; với overlap32, độ dốc dùng `1/(32−1)`, endpoints 0/1. Trọng số vùng giữa được chặn ở 1; rìa canvas không có tile láng giềng không bị taper về 0. Ghép 2D bằng tích trọng số x/y và chia tổng trọng số từng pixel, accumulate float64 rồi trả float32. Vì vậy map hằng/trường tuyến tính chung được giữ, pixel có một tile nhận đúng giá trị đó; map của các tile khác nhau chuyển tuyến tính qua overlap, không có bước nhảy tại hai đầu overlap. Cắt padding khi ghi vùng H×W; mọi mẫu số phải dương.
- `boxes_to_global` cộng origin, clip vào ảnh, bỏ box rỗng sau clip; giữ score ở float64 để không mất độ chính xác khi dịch box. NMS class-agnostic, tie giữ thứ tự đầu vào, chỉ loại khi IoU > threshold. Hàm/module global_nms và method TileManager.global_nms dùng cùng implementation.
- Hash config = SHA-256 JSON chuẩn hóa (defaults, key order, kiểu tham số đã kiểm); PreparedInput.sha256 = SHA-256(header JSON chứa dtype/shape/meta + byte NUL + pixels C-contiguous). Hash không chứa path hoặc timestamp. Mọi meta preprocessing có version/hash/color_order/normalized. API helper `preview_input` gọi đúng `prepare_input`; scripts chỉ parse args, kiểm output và ghi JSON. `check_loaders` cũng nằm trong API, script không tự viết logic chuẩn bị ảnh.
- Augmentation là API riêng, áp vào ảnh/box đầy đủ trước tiling nếu caller chọn dùng; không tự augment trong prepare_input hoặc loader/eval. Train-only và recipe được khóa; thay danh sách rotation/p=0.5/photometric phải thành recipe đã duyệt khác, không âm thầm đổi aug_v1.

Các commit triển khai:

- `2335efe`: Phase 0, ghi câu hỏi trước code.
- `3d8f660`: dependency NumPy và TileManager/stitch/NMS.
- `e9b0b2a`: preprocessing có version/hash, config và CLI preview dùng chung API.
- `eca8ee1`: Sample/loader, bảo toàn pair aliases và test gate, CLI check_loaders.
- `66abef3`: augmentation đã được duyệt và ghi hai câu trả lời của người dùng.

## Phase 2

### Lệnh và kết quả chạy thật

Đã chạy ở worktree `..\PCB_Lab_impl`, CPU/Python 3.14.7:

```powershell
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements/step2.txt
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe --no-build-isolation --no-deps -e .
$env:DATASET_ROOT = '..\DatasetVer4_Public'
$env:PATH = (Join-Path (Get-Location) '.venv\Scripts') + ';' + $env:PATH
python scripts/check_loaders.py --dataset-root $env:DATASET_ROOT --out reports/loader_check.json
python scripts/check_loaders.py --dataset-root $env:DATASET_ROOT --out reports/repeat/loader_check.json
Get-FileHash reports/loader_check.json,reports/repeat/loader_check.json -Algorithm SHA256
python reports/verification/step2_self_check.py
python -B -m compileall -q src scripts
uv --cache-dir .cache/uv pip check --python .venv/Scripts/python.exe
```

- Hai lượt check_loaders đều **PASS**, CLI exit **0**, `errors=[]`; JSON giống hệt từng byte.
- SHA-256 loader_check.json: `d76a04afc71586a7607709534a5ea8122bc031baad39efcf0dcc7acb5ad346e6`.
- preprocessing_hash: `69e65565a2402077b41380b2d742ec989d939bf2cf6f3c7dbd80182fe177f770`; version prep_v1.
- compileall và dependency check đều PASS. Không có FAIL chưa giải quyết; các ValueError/PermissionError được chủ ý kích hoạt là kết quả mong đợi trong kiểm tra đầu vào sai/quyền truy cập.

| Loader | n_samples từ manifest | Batch pixel đã đọc | Shape | dtype; min/max |
| --- | ---: | ---: | --- | --- |
| YoloTrainSet | 1792 | 2 ảnh | [2,640,640,3] | uint8; 0/255 |
| AdTrainSet | 895 good train | 2 ảnh → 18 tile | [18,256,256,3] | uint8; 0/255 |
| EvalSet calibration | 460 | 2 ảnh | [2,640,640,3] | uint8; 0/255 |
| EvalSet fusion | 306 | 2 ảnh | [2,640,640,3] | uint8; 0/255 |
| EvalSet test | 440 | **0**, chỉ metadata/gate | null | null |

Mọi batch pixel đều RGB, normalized=false. AdTrainSet tính **tiles_per_image=9**, **num_tiles=8055** từ plan và số good; tất cả tile giữ train/good của Sample nguồn. Iter_tiles cho pixel đúng bằng prepare_input ad_tile. development_selection và fusion có cùng mẫu/thứ tự. Không hard-code các số dataset vào implementation.

### Tính chất toán học và kiểm tra hình học

Script tự kiểm chứng ở `reports/verification/step2_self_check.py` (ignored, không thuộc tests của VERIFIER) đã chạy thành công **12 nhóm kiểm tra**; kết quả text được lưu tại `reports/step2_self_check.json`.

| H×W | Số tile tính được | Sai số max của hàm tuyến tính |
| --- | ---: | ---: |
| 640×640 | 9 | 0 |
| 317×509 | 6 | 0 |
| 257×289 | 4 | 0 |
| 91×103 | 1 | 0 |
| 1×1 | 1 | 0 |
| 1×513 | 3 | 0 |
| 513×1 | 3 | 0 |
| 777×1003 | 20 | 0 |

- Trên cả tám kích thước và ba pad mode: vùng không pad bằng nguyên pixel gốc; stitch từng kênh tái tạo chính xác ảnh; map hằng 3.25 giữ nguyên; map `0.013*x + 0.027*y + 0.3` theo tọa độ toàn ảnh tái tạo đúng float32; pixel chỉ được phủ bởi một tile giữ đúng giá trị tile đó; mọi biên có coverage và kết quả hữu hạn.
- Map tile có giá trị hằng khác nhau ở bốn góc, ảnh 480×480: so với công thức crossfade tuyến tính độc lập hai chiều, sai số max **1.1536382871213391e-7**; hai đầu overlap không có bước nhảy. Kiểm thêm stride128/192/256 giữ map hằng và không chia 0; mặc định contract vẫn stride224/overlap32.
- Box local→global→local trên tám kích thước: sai số **0**, score giữ nguyên, input không bị sửa. Box nằm hoàn toàn trong pad hoặc rỗng sau clip bị loại. NMS giữ hai box có IoU đúng 0.45, loại khi threshold0.449, tie score ổn định; mảng rỗng giữ shape [0,5].
- Letterbox round-trip trên tám kích thước: sai số max **1.1368683772161603e-13 px**, nhỏ hơn yêu cầu ≤1 px. 640×640 giữ scale1/pad0. Upsample bilinear float32 2×2→317×509 nằm trong range [0,1], không overshoot.
- L/LA/RGB/RGBA qua NumPy và PIL cho cùng RGB; kênh màu không đảo. Normalize float32 đặt normalized=true, gọi lại và std0 bị từ chối. Hash config không đổi khi chỉ đảo thứ tự khóa, đổi khi thay pad114→115.
- Aug_v1 với 64 seed đã bao phủ đủ tám phép đối xứng; bbox khớp chính xác vùng pixel đánh dấu, multiset pixel và diện tích giữ nguyên, input không đổi, cùng seed trả cùng ảnh/box/meta. Calibration/fusion/development_selection/test đều bị từ chối.
- Patch Image.open thành hàm lỗi khi tạo/len/duyệt metadata loader: không có decode. Test Sample và test canonical path đều bị chặn trước Image.open. Test allow_test=True chỉ kiểm bằng ảnh **tự sinh** ngoài dataset, không mở ảnh test thật.

### API và CLI preview

Đã gọi prep_preview.py cho ba ảnh train `deeppcb_00041000_defect`, `deeppcb_00041000_good`, `deeppcb_00041001_defect` ở cả yolo/ad_tile/ad_resize (9 lệnh CLI). Mỗi kết quả so với prepare_input gọi trực tiếp bằng Sample và bằng ndarray: **meta, shape và SHA-256 khớp**. Chín digest cụ thể nằm trong `reports/step2_self_check.json`.

Lệnh mẫu đã chạy (tương tự cho hai ảnh và hai mode còn lại):

```powershell
python scripts/prep_preview.py --image "$env:DATASET_ROOT\benchmarks\deeppcb\images\deeppcb_00041000_defect.png" --mode yolo --out-json reports/verification/deeppcb_00041000_defect_yolo.json
python scripts/prep_preview.py --image "$env:DATASET_ROOT\benchmarks\deeppcb\images\deeppcb_00041000_defect.png" --mode ad_tile --out-json reports/verification/deeppcb_00041000_defect_ad_tile.json
python scripts/prep_preview.py --image "$env:DATASET_ROOT\benchmarks\deeppcb\images\deeppcb_00041000_defect.png" --mode ad_resize --out-json reports/verification/deeppcb_00041000_defect_ad_resize.json
```

Đã kiểm thêm ảnh tự sinh 317×509 ở ba mode: API/CLI có cùng hash/meta; box nghịch letterbox dùng meta JSON từ CLI bằng đúng box dùng meta API. CLI chỉ ghi metadata/hash, không xuất ảnh. Spy trong kiểm chứng không cho phép mở bất kỳ đường dẫn ảnh test thật nào; các subprocess preview chỉ nhận ba đường dẫn train nêu trên và ảnh tự sinh.

### Đã làm / chưa làm / rủi ro / câu hỏi mở

- **Đã làm:** toàn bộ phạm vi code Step 2 được giao; recipe và pair_id đã được người dùng duyệt; chạy dataset thật, hai lần loader check, kiểm toán hình học và API/CLI. Không cần sửa code Step 1.
- **Chưa làm theo phạm vi:** train/inference model, pooling score của model, threshold/benchmark/UI/Gate0; không tuyên bố reflect/augmentation tốt nhất về metric. Kiểm thử độc lập Step 2 của VERIFIER chưa nhận kết quả; khi được báo sẽ đọc `git show step2/tests:docs/handoff/step2-verify.md` và xử lý trên branch impl.
- **Giới hạn:** reflect/constant/replicate có thể tạo dấu hiệu biên nhân tạo; padding không được xem là vùng PCB thật khi stitch/clip. Các tile cùng ảnh tương quan, 8055 tile không phải 8055 ảnh độc lập. Augmentation đổi phân phối hướng dù bảo toàn pixel; lợi ích cần ablation model.
- Canonical conversion không thể suy ra một ndarray bên ngoài có thứ tự BGR; contract yêu cầu caller dùng RGB. Float input bị từ chối thay vì tự đoán thang giá trị/normalized. LA/RGBA bỏ alpha đã ghi rõ; không áp EXIF để tránh tự đổi tọa độ nhãn.
- Dữ liệu ngoài canonical có số tile/ảnh không đều sẽ bị AdTrainSet từ chối để bảo toàn công thức num_tiles của contract. Iterator lazy, không materialize toàn bộ 8055 tile cùng lúc.
- Preprocessing giữ uint8 và chỉ normalize khi gọi rõ ràng; không có dataset inference test. Gate test bảo vệ API đọc Sample/path canonical; caller đã tự đưa ndarray thì phải tự bảo đảm provenance của ndarray đó.
- **Câu hỏi chặn còn mở:** không có; Q1/Q2 đã có câu trả lời trong questions-impl.md. Không sửa file contract (SHA-256 kiểm lại bằng ban đầu), PLAN/DECISIONS/TRIAGE/tests hoặc branch khác; không merge.
- Commit chỉ chứa code/config/requirements và báo cáo text. Fixture/ảnh tự sinh/preview JSON thử trong reports/verification và output repeat được ignore; không stage dataset, ảnh, key/token. Dừng bàn giao để người dùng và VERIFIER review.
