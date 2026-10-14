# Bàn giao IMPLEMENTER — Step 3 (Baseline YOLO: B01 YOLO11n, B02 YOLO11s)

## Phase 0

### Căn cứ và mục tiêu

- Làm việc tại worktree `../PCB_Lab_impl3`, branch `step3/impl` (base `2a1545e`). WORKFLOW.md quy định mỗi agent một branch/worktree; worktree này đã được người điều phối tạo sẵn cho `step3/impl`.
- Đã đọc `docs/handoff/WORKFLOW.md` (quy tắc chung), `docs/handoff/step3-contract.md` (hợp đồng Bước 3), `PLAN.md` §1.2–1.3, §3.1, §5 Bước 3, §6, §7.3, §8, §11; `docs/handoff/step2-contract.md`, `step2-impl.md`; code Step 1–2 (`data/samples.py`, `inference/preprocessing.py`, `data/manifest.py`). Không dùng docs/archive.
- **Không** nạp checkpoint cũ, không chạy runner cũ, không đụng branch khác.
- Mục tiêu Bước 3 (PLAN §5): train baseline YOLO11n (B01) + YOLO11s đối chứng (B02) trên 1.792 ảnh train, 6 lớp; trích dự đoán calibration/fusion để Bước 5 hiệu chỉnh ngưỡng. Đầu ra: artifact, model card, báo cáo validation, `run_manifest.json`.
- **Hoàn thành:** checkpoint nạp lại được (có `allow_smoke=True`), class order đúng (0=open_circuit…5=pin_hole), không dùng test chọn epoch/ngưỡng, artifact đầy đủ B01+B02.

### Metadata thật đã xác minh (môi trường đã cài)

Môi trường phát triển CPU local tại worktree, cài mới vào `.venv` riêng (Python 3.14.7, Windows x64, **CUDA unavailable**). Khóa phiên bản theo bản đã cài:

- **torch 2.14.0+cpu**, **torchvision 0.29.0+cpu**, **ultralytics 8.4.161**.
- Step 2: NumPy 2.4.3, Pillow 12.1.1, PyYAML 6.0.3, setuptools 84.0.0.
- Dataset `..\DatasetVer4_Public` **tồn tại** (xác nhận bằng ls). Trong process tool hiện tại biến `DATASET_ROOT` chưa có giá trị; adapter vẫn phân giải đúng từ `configs/dataset.yaml`.
- **Quan trọng — network/telemetry:** import `ultralytics` tự tạo file user-level `<APPDATA>/Ultralytics/settings.json` (ngoài DATASET_ROOT, ngoài repo). Không ghi vào dataset. Đã ghi chú đoạn sau để vô hiệu hóa telemetry (`ULTRALYTICS_NO_ANALYTICS=1`) trong train/extract.

### Đọc mã ultralytics 8.4.161 — xác nhận default thực và điểm khác hợp đồng

Đọc trực tiếp `ultralytics/cfg/default.yaml` và `engine/predictor.py`, `data/dataset.py`, `nn/tasks.py` của bản đã cài (không dùng trí nhớ). Bảng so sánh với hợp đồng:

| Tham số | Default thực (8.4.161) | Hợp đồng step3 | Quyết định |
|---|---|---|---|
| `epochs` | 100 | 100 | dùng 100 |
| `patience` | **100** | **20** | override = 20 (ghi args_used) |
| `batch` | 16 | null → duyệt 16 | dùng 16 (user duyệt) |
| `imgsz` | 640 | 640 | dùng 640 |
| `optimizer` | `auto` | **AdamW** | override = AdamW |
| `lr0` | 0.01, comment "SGD=1e-2, Adam/AdamW=1e-3" | null → duyệt 0.001 | dùng **0.001** (đúng default AdamW thực) |
| `deterministic` | True | True | dùng True |
| `seed` | **0** | CLI mặc định 42 | truyền `seed=42` qua CLI/runner |
| `workers` | 8 | null → phải đặt | đặt (CPU local=0/2; cloud GPU theo VRAM) |
| `cache` | False | False | False (tránh ghi cache) |
| `plots` | True | (muốn lưu plots) | True |
| `conf` | predict 0.25 / val 0.001 | `infer.conf_floor`=0.001 | val/infer dùng 0.001 |
| `iou` (NMS) | 0.7 | 0.7 | dùng 0.7 |
| `max_det` | 300 | 300 | dùng 300 |
| `agnostic_nms` | False | False | False |
| `amp` | True | (không nói) | CPU: tắt (False) để tránh warning/path sinh artifact; ghi args_used. Cloud GPU: bật. |
| `warmup_epochs` | 3.0 | — | giữ mặc định |
| `cos_lr` | False | — | giữ mặc định |
| `close_mosaic` | 10 | — | giữ mặc định (ảnh PCB không mosaic-friendly nhưng đây là mặc định; ablation sau) |
| augmentation photometric | `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, scale=0.5` | `augment=null` → duyệt sau | **Tất cả photometric/mosaic tắt (0.0)** cho recipe `yolo_aug_v1`; chỉ giữ hình học thuần của Step 2 (xoay 90°×k + flip). Lý do: giữ nguyên lỗi nhỏ ở 640px. |

**Các điểm khác hợp đồng / cần lưu ý (ghi vào handoff theo yêu cầu Phase 0.3):**

1. **Kênh màu (rất quan trọng):** `engine/predictor.py:165–179` — khi truyền **mảng numpy** vào `model.predict()`, ultralytics kỳ vọng **BGR uint8** (docstring rõ: "[(H,W,3)xN] for list of BGR uint8 arrays"), và dòng 179 `im = im.flip(1)  # BGR to RGB` đảo kênh. Còn khi truyền **đường dẫn ảnh**, ultralytics dùng `cv2.imread` → cũng BGR. Nghĩa là: cả hai đường đều vào BGR rồi đảo thành RGB nội bộ. Adapter của Step 3 làm việc trên **RGB canonical** (theo Step 2) → phải chuyển `RGB → BGR` (`arr[..., ::-1]`) trước khi đưa mảng cho ultralytics, để kết quả khớp với đường truyền path. Tôi sẽ viết kiểm chứng tại chỗ chứng minh `predict(RGB_array_bgr_converted)` ≈ `predict(image_path)` (cùng box, cùng conf) — chi tiết Phase 1.
2. **Ghi cache nhãn:** `data/dataset.py:109` `cache_labels(path = Path("./labels.cache"))` — đường dẫn **tương đối với thư mục chạy (CWD)**, không phải DATASET_ROOT. Để "không cho ultralytics ghi bất cứ thứ gì vào DATASET_ROOT", runner sẽ (a) đặt `cache=False`, và (b) chdir vào thư mục `runs/...` được kiểm soát trước khi gọi train, đồng thời guard kiểm tra không có file `.cache`/`labels.cache` nào xuất hiện dưới DATASET_ROOT.
3. **Truy cập mạng:** `nn/tasks.py:1827,1891` `attempt_download_asset(weight)` — chỉ kích hoạt khi file trọng số **không tồn tại local**, tải từ `github.com/ultralytics/assets/releases`. Hợp đồng yêu cầu pretrained phải có sẵn (`artifacts/pretrained/*.pt`) và chỉ tải khi `allow_download=True`. Runner sẽ từ chối (`ConfigError`) nếu pretrained thiếu và `allow_download=False`. Khi `allow_download=True` sẽ ghi nguồn+phiên bản.
4. **Kiểm tra AMP:** không tải mạng; chỉ chạy một forward nhỏ trên device. Trên CPU để `amp=False` nên bước này bỏ qua/nhanh. Ghi rõ method đo peak memory.
5. **Font/telemetry:** `utils/__init__.py:1008, SETTINGS_FILE` — import tạo settings.json user-level (đã nêu). Không tải font từ mạng trong flow detect thông thường. Thiết lập `ULTRALYTICS_NO_ANALYTICS=1`.

### Quyết định ảnh hưởng phạm vi (3 đề xuất đã duyệt)

**(a) Recipe `yolo_aug_v1`** — hình học thuần, tái dùng recipe `aug_v1` đã duyệt của Step 2 (xoay 0/90/180/270° + lật ngang p=0.5, RNG cục bộ theo seed, **không** photometric/scale/resample). Lý do giữ lỗi nhỏ: theo `reports/data_profile.md` §"Kích thước box", lỗi nhỏ nhất pin_hole short_side min 20px (P5/50/95 = 24/30/42), mouse_bite min 20px, spur min 21px — ở 640px, biến đổi hình học giữ nguyên pixel → **không làm mất/đổi bản chất lỗi nhỏ**. Mọi photometric (hsv/brightness/contrast), scale, mosaic đẩy sang ablation có tên (Bước 9). Hợp đồng truyền augmentation qua tham số ultralytics `"yolo_aug_v1"` → tôi sẽ set toàn bộ hyp photometric = 0.0 và bật augmentation pipeline hình học riêng (dùng `apply_augmentation` Step 2 trên mỗi ảnh trước letterbox, qua `augment.py` của dataset loader hoặc augment custom). Quyết định cụ thể cơ chế áp augmentation vào YOLO DataLoader ghi rõ ở Phase 1.

**(b) lr0 & batch cho AdamW** — `lr0=0.001`, `batch=16`. Đã duyệt. Khớp default thực AdamW của bản 8.4.161 (`lr0` comment: "Adam/AdamW=1e-3"). `batch` tinh chỉnh theo VRAM trên cloud; nếu B02 (yolo11s) hết VRAM sẽ ghi lý do (hợp đồng cho phép khác batch chỉ khi hết VRAM).

**(c) Tham số `infer{}`** — khoá: `conf_floor=0.001` (lọc cực thấp, không rụng box trước bước chọn `tau_yolo`, đúng PLAN §5 Bước 3), `iou(NMS)=0.7`, `max_det=300`, `imgsz=640`, `agnostic_nms=False`, `half=False`. Giữ nguyên hợp đồng.

### Giả định nhỏ

- Không dùng test để chọn epoch/ngưỡng; checkpoint chọn bằng metric validation trên calibration (`ultralytics_fitness` = weighted combination mAP50/mAP50-95/...) như hợp đồng `selection.metric`.
- `classes_ref` = `benchmarks/deeppcb/configs/classes.json` (tên 0..5). `data.yaml` chỉ có `train`/`val`, không có `test`. names theo thứ tự classes.json.
- `seed` mặc định 42 qua CLI; hợp đồng config không chứa seed (config_hash không gồm seed).
- ARTIFACT: smoke không tạo artifact chính thức (chỉ `runs/smoke/...`); `from_artifact` mặc định `allow_smoke=False` sẽ từ chối artifact smoke.
- Peak memory: CPU RAM đo bằng `tracemalloc`/RSS trước-sau (method ghi rõ); VRAM = null trên CPU (method = "torch.cuda.* không khả dụng trên CPU; ghi null").

### Chỗ mơ hồ / câu hỏi

- **Không có mơ hộng chặn triển khai.** Hợp đồng đã chốt rõ phạm vi, schema, bất biến. Các giá trị null (lr0/batch/workers/augment) đã được duyệt ở trên.
- Một điểm nhỏ chưa chốt bằng văn bản: cơ chế áp augmentation hình học vào YOLO DataLoader (custom `Albumentations`-style transform vs áp trước letterbox). Tôi sẽ dùng transform tùy chỉnh đọc từ `apply_augmentation` Step 2, áp trên ảnh gốc rồi letterbox — không đổi training code ultralytics. Ghi rõ ở Phase 1, không cần dừng hỏi.

### File dự định tạo/sửa (Phase 1)

- Tạo `src/pcb_lab/models/__init__.py`, `src/pcb_lab/models/yolo/{__init__,view,train,adapter,extract}.py`.
- Tạo `configs/models/yolo_common.yaml`, `configs/models/yolo11n.yaml`, `configs/models/yolo11s.yaml`.
- Tạo `scripts/train_yolo.py`, `scripts/extract_yolo_predictions.py`.
- Tạo `requirements/step3.txt` (torch cpu / ultralytics / step2 deps, khóa phiên bản).
- Tạo `docs/RUN_GPU.md` (lệnh CUDA local + Colab/Kaggle: đặt DATASET_ROOT, cài requirements, tải pretrained, mang artifacts về local).
- Cập nhật chính handoff này (Phase 1/2); ghi `docs/handoff/questions-impl.md` nếu phát sinh mâu thuẫn.
- Không sửa tests/, PLAN/DECISIONS/TRIAGE/WORKFLOW/contractor; không commit dataset/checkpoint/preds/ảnh. Mỗi commit kiểm `git status`.

### Trạng thái duyệt

- Đề xuất (a) recipe hình học, (b) lr0=0.001/batch=16, (c) infer{} — **đã được người dùng duyệt** qua AskUserQuestion.
- **DỪNG chờ người dùng duyệt trước khi chạy train thật trên GPU.** Phase 0 chỉ thiết lập môi trường + xác nhận default + ghi handoff này.

---

## Phase 1

### File đã tạo (đã commit sớm, xem git log)

- `src/pcb_lab/models/__init__.py`, `src/pcb_lab/models/yolo/{__init__,view,train,adapter,extract}.py`.
- `configs/models/yolo_common.yaml` (chỉ comment, parse thành mapping rỗng), `yolo11n.yaml`, `yolo11s.yaml`.
- `scripts/train_yolo.py`, `scripts/extract_yolo_predictions.py`.
- `requirements/step3.txt` (torch 2.14.0+cpu, ultralytics 8.4.161, các dep Step 2 khóa phiên bản, kèm lệnh cài `--index-url https://download.pytorch.org/whl/cpu`).
- `docs/RUN_GPU.md` (CUDA local + Colab/Kaggle: đặt DATASET_ROOT, cài requirements, tải pretrained, mang artifact về local).

### Cơ chế đã hiện thực (theo hợp đồng step3-contract.md)

**view.py — `build_yolo_view`**
- Chỉ chấp nhận `train` + `calibration`; `fusion`/`test` → `ViewError(PermissionError)` (`_reject_forbidden`).
- Ghi hardlink ảnh (fallback copy nếu cross-device) + label YOLO `class cx cy w h` (6 decimals) + `data.yaml` **chỉ có `train`/`val`/`names`, không có `test`** (assert cứng).
- Không bao giờ ghi vào DATASET_ROOT: dùng `output_path()` guard + `cache=False` trong runner; `view_manifest.txt` + `view_signature` (sha256 trên `sample_id\timage_sha256\tlabel_sha256` đã sort) để rebuild idempotent.
- `limit={'good':n,'defect':n}` tuỳ chọn; nếu `None` giữ nguyên toàn bộ (train 1792 / calibration 460).

**train.py — `train_yolo`**
- Merge `yolo_common.yaml` + model config; từ chối key không thuộc `ALLOWED_CONFIG_KEYS` (`ConfigError`).
- `_check_nulls` từ chối `lr0/batch/workers/augment` null → runner không chạy với hyperparam thiếu.
- Override ép **toàn bộ** photometric/mosaic = 0.0 (`hsv_*`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr=0.5`, `mosaic`, `mixup`, `copy_paste`, `bgr`, `close_mosaic`); `amp=False` (CPU), `cos_lr`/`warmup_epochs` từ config.
- Recipe `yolo_aug_v1` = hình học thuần, áp **online** qua `_AugmentedYOLODataset.get_image_and_label`: xoay 90°×k (seed per `(aug_seed,epoch,index)`) + lật ngang p=0.5, áp lên cả ảnh BGR và box xywh chuẩn hóa (`_geometric_augment`). **View không bị mở rộng** → giữ invariant #1 (1792/460).
- Quản lý pretrained: `ConfigError` nếu thiếu và `allow_download=False`; nếu `allow_download=True` ghi source. Smoke dùng `arch` + `limit=8/8`, `epochs=1`, không tạo artifact.
- `chdir` vào `runs/...` trước khi gọi train (ngăn ghi `labels.cache` vào DATASET_ROOT). Đo peak RAM (psutil / win32 psapi), VRAM = null trên CPU (`method` ghi rõ, không fake).
- Xuất `run_manifest.json`, `env.json`, copy `best/last.pt` → `artifacts/yolo/<model_id>/seed<seed>`, `calibration_per_class.json`, `artifact.json` (có sha checkpoint + class_order + config_hash + view_signature), `model_card.md` (tiếng Việt, ASCII-safe), `_link_preds`.

**adapter.py — `YoloDetector` / `UltralyticsEngine`**
- `UltralyticsEngine.infer` nhận **RGB uint8 HWC** (canonical Step 2), chuyển `bgr = arr[..., ::-1]` rồi `model.predict(...)` — khớp đường truyền path (cv2.imread=BGR). Trả `[N,6]` xyxy/conf/cls trong không gian letterboxed 640.
- `YoloDetector.predict` dùng **duy nhất** `prepare_input(image,"yolo")` + `LetterboxMeta` + `unletterbox_boxes` (tái dùng Step 2) → trả list `Detection` đã sort theo confidence giảm dần; `image_score` = max conf hoặc 0.0.
- `from_artifact` kiểm: `artifact.json` + `best.pt` tồn tại, sha checkpoint khớp (`ArtifactMismatchError`), class order khớp `classes.json` (`ClassOrderError`), artifact smoke bị từ chối trừ khi `allow_smoke=True` (`SmokeArtifactError`).

**extract.py — `extract_predictions`**
- Chỉ `calibration`/`fusion`; `test`/khác → `PermissionError`.
- Đọc sample qua `ManifestDataset`, chạy adapter, ghi `preds_<partition>.jsonl` (1 dòng/ảnh, sort theo `sample_id`, deterministic, ghi sha256 file).
- Lỗi inference ghi vào `error_reason`, **không** drop/đổi thành GOOD/0; confidence ≥ `conf_floor`.

### Kiểm chứng tĩnh / mức module (đã chạy, PASS — chưa cần checkpoint/train)

1. **Compile:** `py_compile` toàn bộ `src/pcb_lab/models/yolo/*.py` + `scripts/*.py` → OK.
2. **Import symbols:** `import pcb_lab.models.yolo` + `_PCBDetectionTrainer`, `_AugmentedYOLODataset`, `_geometric_augment`, `_merge_configs`, `_config_hash`, `_check_nulls`, `ViewError`, `ConfigError`, `YoloDetector`, `extract_predictions` → OK (sửa lỗi thiếu `from dataclasses import dataclass, field` và `from ultralytics import YOLO`).
3. **`_merge_configs`** trên `yolo_common.yaml`(comment-only) + `yolo11n.yaml` → merge đúng, keys `{architecture,classes_ref,infer,model_id,pretrained,recipe_id,selection,train}`; `_check_nulls` PASS trên config hợp lệ, raise `ConfigError` khi `lr0=None`.
4. **`_geometric_augment`** (k=1,flip=True) trên ảnh 4×4 + box → ảnh xoay/lật, box `[[0.25,0.25,0.5,0.5]]` → `[[0.75,3.75,0.5,0.5]]` (đúng công thức flip ngang: cx'=1-cx; xoay 90° cw: (cx,cy)→(1-cy,cx), w↔h).
5. **Build view thật trên DATASET_ROOT** (`..\DatasetVer4_Public`): `train=1792 (good 895/defect 897)`, `calibration=460 (good 230/defect 230)`; box_counts mỗi lớp hợp lý (open_circuit, short, mouse_bite, spur, spurious_copper, pin_hole); `data.yaml` **không có `test`**; `view_signature` sinh đúng. `limit={'good':2,'defect':2}` → 4/4 ảnh.
6. **Gating partition:** view từ chối `fusion`/`test`; extract từ chối `test`/`train`/`fusion+test` → `PermissionError`.
7. **DATASET_ROOT untouched:** sau build view, `ls` dataset chỉ ra các thư mục gốc (audit/benchmarks/...); không có file cache mới.
8. **Adapter wiring (không cần ckpt):** `prepare_input(path,"yolo")` trên ảnh defect thật trả meta có `scale/pad_left/pad_top/orig_w/orig_h` + `LetterboxMeta(...)` + `unletterbox_boxes` roundtrip → OK.

### Ghi chú kỹ thuật

- `resolve_classes_root`/`load_canonical_names` trong adapter phân giải dataset root từ `configs/dataset.yaml`/env, không hardcode.
- `view.data_yaml` gán `val: images/calibration` (không phải `test`) — khớp hợp đồng `selection.split=calibration`.
- `config_hash` loại trừ `seed` (CLI-only) nên tái lập được từ config.

---

## Phase 2 — Smoke CPU & Tự kiểm chứng

### Sửa lỗi thực thi (phát hiện khi chạy thật trên ultralytics 8.4.161)

1. **Smoke phải random init, không download:** runner gốc truyền `arch` ("yolo11n") làm model string → ultralytics nối `.pt` rồi tải trọng số. Đã sửa: smoke dùng `"<arch>.yaml"` (random init, không mạng). Real run vẫn dùng pretrained `.pt` như hợp đồng.
2. **Run dir path:** `project=runs_base.parent.parent` chỉ lên `runs/yolo`, làm `best.pt` không nằm ở `runs_base` → sửa thành `runs_base.parent` (`runs/yolo/<model_id>/seed<seed>`).
3. **`fraction` (dataloader):** truyền `None` gây `int * NoneType`. Đã sửa thành `1.0`.
4. **API label đổi:** ultralytics 8.4.161 trả `label["instances"]` (đối tượng `Instances`, box normalized xywh trong `_bboxes.bboxes`), **không phải** `label["bboxes"]`. Đã viết lại `_geometric_augment` (hoạt động trong không gian normalized xywh — khớp chính xác recipe `aug_v1` Step 2: xoay 90° ccw `(cx,cy,bw,bh)->(cy,1-cx,bh,bw)`, flip `cx->1-cx`) và `get_image_and_label` ghi qua `instances.update(...)` với `float32`.
5. **Adapter class order:** `model.model.names` trả dict `{idx: name}`, `list(...)` ra key số. Đã chuẩn hóa (dict → list theo index) trước so sánh với `classes.json`.

Tất cả sửa trên đã commit (`a06bc9a`).

### Kết quả chạy (PASS)

**A. Smoke train CPU 1 epoch (real data, limit 8/8 good/defect)**
- `runs/smoke/B01_yolo11n/seed42/` sinh đầy đủ: `run_manifest.json`, `env.json`, `weights/best.pt`(5.47 MB), `weights/last.pt`, plots (labels/results/confusion/curves), `train_batch0.jpg`, `val_batch0_pred.jpg`.
- `run_manifest.json`: `model="yolo11n.yaml"`, `epochs=1`, `optimizer=AdamW`, `lr0=0.001`, `patience=20`, photometric/mosaic = 0.0 (ghi `args_used`), `peak_ram_mb=770.0` (psutil), `peak_vram_mb=null` (CPU, method ghi rõ, không fake), `link_mode=hardlink`, `view_signature=ce24904dbd92a114`, `device=cpu`, `torch=2.14.0+cpu`, `ultralytics=8.4.161`, `dataset_release_sha256` đầy đủ.
- **`artifact_dir=null`** — đúng, smoke không tạo artifact chính thức (hợp đồng). Validation trên calibration ghi `overall` + `per_class` 6 lớp.

**B. Adapter reload + equivalence (chứng minh kênh màu)**
- Tạo smoke artifact dir (copy `best.pt` + viết `artifact.json` smoke, `smoke:true`). `YoloDetector.from_artifact(..., allow_smoke=True)` nạp OK.
- **EQUIVALENCE `predict(RGB_array)` == `predict(image_path)`:** test 12 ảnh calibration, **0 mismatches** (cùng số box, class, confidence, tọa độ). → chứng minh `RGB->BGR` trong `UltralyticsEngine` khớp đường truyền path (cv2.imread BGR).

**C. Extract predictions (calibration, full partition)**
- `extract_predictions(artifact, ..., partitions=("calibration",))` → `preds_calibration.jsonl` 460 dòng, 32.200 detections, deterministic (sha ghi rõ).
- Sanity: confidence ∈ [0,1], xyxy hợp lệ (x1≤x2, y1≤y2, trong bounds) — **0 bad rows**.
- Gating `test` → `PermissionError` (đã verify Phase 1).

**D. Full view + invariants (vs PLAN §1.3)**
- `train=1792` (good 895 / defect 897), `calibration=460` (good 230 / defect 230) — đúng.
- Per-class train box counts **khớp chính xác** PLAN §1.3: open_circuit 1216, short 973, mouse_bite 1228, spur 981, spurious_copper 884, pin_hole 865.
- `data.yaml` chỉ có `path/train/val/names`, **không có `test`**.
- `fusion`/`test` bị `ViewError(PermissionError)` từ chối.
- `view_signature` tái lập được (rebuild → cùng `ce24904dbd92a114`).
- `link_mode=hardlink`.

### Điều kiện/ngoại lệ đã ghi nhận

- Smoke chạy trên CPU (local dev) — không phải GPU đám mây. Hợp đồng Phase 2 chỉ yêu cầu smoke self-verify trên CPU; **train thật 100 epoch B01/B02 chạy trên GPU đám mây (Gate 0)** theo `docs/RUN_GPU.md`.
- `peak_vram_mb=null` trên CPU là đúng (không có CUDA) — ghi method rõ, không fake.
- Predict có warning `'half' is deprecated` (ultralytics 8.4.161) — không ảnh hưởng (`half=False` trong infer{}). Sẽ theo dõi khi chạy cloud.

### Trạng thái duyệt

- **Phase 2 PASS** trên CPU: smoke train OK, adapter equivalence OK, extract OK, full-view invariants OK. Không merge, chờ điều phối viên review rồi chuyển sang train GPU thật (Bước 0/Gate 0).

---

## Gói thực thi Colab (Gate 0 & Full Training 100 epochs)

### Thành phần đã tạo

- `notebooks/train_yolo_colab.ipynb` — notebook "run-all" (Runtime ▸ Run all) 8 cell:
  1. **Gate 0 Hardware**: `!nvidia-smi` + `torch.cuda.is_available()`, in tên GPU + CUDA capability, assert compute capability >= 7.0.
  2. **Dataset Setup**: mount Drive, gán `os.environ["DATASET_ROOT"]`, assert manifest `samples.jsonl` tồn tại.
  3. **Install**: cài từ `requirements/step3.txt` **nhưng bỏ qua torch/torchvision** (giữ CUDA build của Colab), rồi `pip install --no-build-isolation --no-deps -e .`.
  4. **Pretrained**: tạo `artifacts/pretrained/`, tải `yolo11n.pt` + `yolo11s.pt` từ Ultralytics release `v8.3.0`.
  5. **Train B01 YOLO11n**: 100 epoch, seed 42, AdamW, `--device 0`, AMP auto-on GPU.
  6. **Train B02 YOLO11s**: tương tự.
  7. **Extract**: `calibration` + `fusion` cho cả 2 model (tập `test` bị chặn trong code).
  8. **Export**: nén `artifacts/yolo/` -> `yolo_step3_artifacts.zip`, `files.download()`.
- `scripts/package_colab.py` — đóng gói mã nguồn thành `exports/colab_bundle.zip` (61 files, ~177 KB), loại `.venv/ runs/ .cache/ artifacts/ *.pt *.zip` và dataset.

### AMP (thay đổi interface)

- `train_yolo` và CLI `--amp` mới: **mặc định auto** bật trên CUDA GPU, tắt trên CPU; có thể ép bằng `--amp true|false`. Trước đây hardcode `amp=False`. Cloud GPU (Cell 5/6) sẽ ghi `amp=True` vào `run_manifest.json`.

### Cách đưa lên Colab

```powershell
# 1) Dong goi (local)
python scripts/package_colab.py                 # -> exports/colab_bundle.zip
# 2) Tai colab_bundle.zip len Google Drive hoac GitHub
# 3) Mo Colab, upload giai nen, chay notebook/train_yolo_colab.ipynb (Runtime -> Run all)
# 4) O Cell 2 sua DATASET_ROOT neu DatasetVer4_Public o duong dan khac tren Drive
# 5) Sau Cell 8, tai yolo_step3_artifacts.zip ve local, giai nen vao artifacts/yolo/
```

Ghi chu: không commit file `.pt` hay `.zip` lón vao Git (`.gitignore` dã lo). `exports/colab_bundle.zip` cung nam trong ignore (không commit).

## Sửa theo phản hồi Phase B — 2026-09-25

### Phạm vi đã được chỉ định

Người dùng yêu cầu thực hiện vai trò Implementer tại `PCB_Lab_impl3`, branch `step3/impl`,
sửa các mục flip ảnh, pretrained path, output guards, copy/link_mode, tie-break, JSONL Detection
và import subprocess. Base trước sửa: `b6465893ba3da7c4967c3487f6dc16b0d6941233`.
Đã đọc WORKFLOW/contract và báo cáo Verifier tại `PCB_Lab_verify3/docs/handoff/step3-verify.md`
(commit verifier `7ffff82`). Không có điểm mơ hồ chặn các sửa đổi được chỉ định.
Phạm vi file: bốn module YOLO và handoff này; không sửa test hay hợp đồng, không merge branch.

### Thay đổi

- `train.py`: `_geometric_augment` lật ngang cả pixel BGR bằng `np.ascontiguousarray(np.fliplr(...))`
  sau bước xoay, đồng bộ với bbox; `pretrained_path` được resolve absolute trước kiểm file/hash/chdir.
- `train.py`: guard `output_path` ngay đầu hàm; resolve và kiểm các đường run/view/artifact bằng
  `artifact_path` trước mkdir, chặn path escape và junction ra ngoài out_root. Import `subprocess`
  để `_git_state` lấy commit/dirty thật.
- `extract.py`: guard output trước đọc artifact/nạp model/mkdir, kể cả out_dir mặc định;
  kiểm đường file JSONL đã resolve. Đổi khóa detection từ `xyxy` sang `xyxy_original`.
- `view.py`: truyền `link` xuống helper, dùng `shutil.copy2` khi yêu cầu copy. Thay directory entry
  cũ trước ghi để rebuild copy không giữ hardlink tới nguồn. Ghi mode thực; nếu chỉ một số ảnh
  hardlink thất bại, chuyển toàn bộ ảnh được chọn sang copy để `link_mode="copy"` đúng với view.
- `adapter.py`: sort theo `(-confidence, class_id, x1, y1, x2, y2)`.

### Kiểm tra đã chạy

Dùng Python 3.14.7, torch 2.14.0+cpu, Ultralytics 8.4.161, pytest 9.1.1 trong venv CPU
đã cài đúng `requirements/step3.txt` ở `.cache/step3-verifier-b/.venv` của repo gốc.
Chạy nguyên test Verifier, không sửa/copy vào branch impl. Runner tạm thêm `impl3/src` lên đầu
sys.path, giữ `verify3/src` ở cuối để conftest không ưu tiên source cũ; assert và log xác nhận
`pcb_lab.__file__` nằm trong `PCB_Lab_impl3/src`.

```powershell
# cwd ../PCB_Lab_impl3
$base = './.cache/step3-verifier-b'
$py = "$base/.venv/Scripts/python.exe"
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_CONFIG_DIR = "$base/settings"
$env:MPLCONFIGDIR = "$base/mpl"
$env:PATH = (Join-Path $env:ProgramFiles 'Git/cmd') + ';' + $env:PATH
$env:DATASET_ROOT = '../DatasetVer4_Public'
$env:STEP3_ARTIFACT_DIR = ''
$tests = @(Get-ChildItem '../PCB_Lab_verify3/tests/test_yolo_*.py' | ForEach-Object FullName)
& $py .cache/step3-fixes/run_verifier.py @tests -m 'not dataset and not smoke' -q --tb=short -p no:cacheprovider --basetemp=../PCB_Lab_impl3/.cache/step3-fixes/fast-tmp --junitxml=.cache/step3-fixes/fast.xml
& $py .cache/step3-fixes/probe_fixes.py
& $py .cache/step3-fixes/run_verifier.py '../PCB_Lab_verify3/tests/test_yolo_view.py::test_real_view_counts_boxes_isolation_and_read_only' '../PCB_Lab_verify3/tests/test_yolo_smoke.py::test_one_epoch_cpu_train_reload_and_no_official_artifact' -q --tb=short -p no:cacheprovider --basetemp=../PCB_Lab_impl3/.cache/step3-fixes/real-smoke-tmp --junitxml=.cache/step3-fixes/real-smoke.xml
```

Lệnh thực tế lưu stdout/stderr thành `fast.log`, `probes.log`, `real-smoke.log` dưới
`.cache/step3-fixes`. Settings/font Arial đã có từ lượt Verifier B; socket guard của test giữ nguyên.
Việc dùng tài nguyên đã cấp sẵn không chứng minh lỗi network B-C04 đã được sửa.

| Phép kiểm | Kết quả thật |
|---|---|
| Nhóm test nhanh Step 3 | **54 passed, 9 failed, 2 skipped, 3 deselected**, 41.05 s |
| Tám node nhắm đúng lỗi sửa: flip, relative pretrained, train/extract output guard, copy inode, EXDEV fallback, tie-break, Detection schema | **Cả tám PASS** trong lượt nhanh trên |
| Probe thêm | **PASS**: 8 tổ hợp xoay/lật ảnh có box và 8 tổ hợp ảnh không box; oracle lấy biên vùng pixel để so box; rebuild hardlink→copy không gọi os.link/không alias nguồn; fallback một ảnh chuyển cả view sang copy; guard sớm, path escape và Windows junction; snapshot dataset giả không đổi |
| `_git_state` | **PASS**: commit bằng `git rev-parse HEAD`, `git_dirty=true` đúng với worktree đang sửa; không còn null |
| Real view + smoke | **1 passed, 1 failed**, 43.14 s; real view PASS, smoke fail ở `best_epoch=null` |

Real view đạt 1.792 train/460 calibration, box-count đúng plan, round-trip ≤0.01 px,
hash tách fusion/test và snapshot dataset thật không đổi. Smoke thực sự train 1 epoch CPU trên
16 ảnh train + 16 calibration, reload và so adapter RGB/path đạt ≤0.5 px / ≤1e-3 confidence,
checkpoint hashes khớp manifest, không tạo artifact chính thức, dataset thật không đổi.
Smoke ghi đúng `git_commit=b6465893ba3da7c4967c3487f6dc16b0d6941233`, `git_dirty=true`
vì test chạy trước commit sửa. Schema đi qua kiểm git rồi fail ở `best_epoch` (vẫn null);
`train_time_s` cũng vẫn null. Không ghi đè provenance smoke thành commit mới sau khi train.

### Handoff cho Verifier

Các sửa đổi được chỉ định đã qua kiểm cục bộ; **chưa phải Phase B PASS**. Các tuyên bố PASS
ở phần Phase 2 lịch sử phía trên không thay thế báo cáo B của Verifier.
Chín failure nhanh còn lại: run directory rỗng được chấp nhận; class thứ bảy; resume chưa nối
trainer; ba nested config key chưa bị chặn; hai ca run_id extraction không khớp manifest;
view rebuild giữ mẫu cũ khi giảm limit. Smoke còn lỗi metadata epoch/time; các mục khác
trong báo cáo B ngoài phạm vi chỉ định, gồm mạng ngầm, vẫn cần xử lý trước train thật.

Không hạ kỳ vọng test, không sửa test của Verifier; không train B01/B02 thật hay tự merge.
Log/probe/checkpoint chỉ ở `.cache/`, không commit. Diff WORKFLOW/contract so với main rỗng;
`git diff --check` sạch. Chờ Verifier kiểm chứng lại commit sửa này.

## Phase C metadata refresh (2026-09-27)

Completed the requested local metadata correction for B01/B02 seed42 without
training. The three requested real-artifact tests PASS for both models. This is
not a claim that the entire extended Verifier suite is green; findings below
remain visible for final review.

| Model | epochs_run | best_epoch | train_time_s | refreshed mAP50 | pr_conf |
|---|---:|---:|---:|---:|---:|
| B01_yolo11n | 33 | 13 | 1260.5 | 0.9341408318893346 | 0.4964964964964965 |
| B02_yolo11s | 43 | 23 | 1695.3 | 0.9408849607293804 | 0.5795795795795796 |

Changes and evidence:

- `_validate` explicitly uses `rect=False`, 640x640, the recorded inference
  settings including `half=False`, and calibration only. Validation output stays
  in its scratch view. `pr_conf` is the confidence coordinate at the maximum of
  the smoothed mean F1 curve, matching Ultralytics 8.4.161 precision/recall.
- Ground-truth counts come from manifest-backed view counts (fallback:
  `metrics.nt_per_class`), not `metrics.box.nc` (number of classes). Counts:
  open_circuit=234, short=160, mouse_bite=339, spur=277,
  spurious_copper=269, pin_hole=265; total=1544 on 460 calibration images.
- Training summaries are derived from actual `results.csv`, including for future
  smoke runs. Missing/malformed evidence raises instead of inventing 100 epochs.
  No training or smoke training was run during this refresh.
- Source provenance is `7623c11d88e4502c879d6f0efa71aab4ae495839`.
  The full SHA `7623c11a28a3a2e37f07e5223049b49fb7283726` in the request does not
  exist locally. The allowed HEAD alternative was used after comparing all 61
  files in the original `exports/colab_bundle.zip` against that commit. All match,
  allowing recovery of `git_dirty=false` for the original training source.
  Current metadata edits are documented separately; they are not represented as
  a new training run. Future Colab bundles embed `bundle_provenance.json` and the
  runner reads it when `.git` is unavailable.
- `model_card.md` now records the original training versions (Ultralytics 8.4.161,
  Torch 2.14.0+cu130) and actual config hashes. A separate validation context
  records local CPU Torch 2.14.0+cpu / Python 3.14.7 and refresh parameters.
- Canonical LF label files make the Windows view signature match the original
  Colab signature `d4e75ccc65d3e3665ce5cd6fbe03c4788f5b9b69daba60ae515b31f2b69d6bd1`.
  `.gitattributes` preserves LF for committed JSON manifests so Windows checkout
  cannot invalidate recorded hashes.
- Updated the eight requested metadata files and recomputed both manifest hashes:
  B01 `0c94bb6ee08db432cf66d81c2dc57eee2f74b69ea1923e1d728522c5197a2e26`;
  B02 `57139af843e514b6187b111261a64850a9775956c5a73fe9d2bd31541a0f5507`.
- All 14 discovered checkpoint/prediction files retain their before/after SHA-256.
  Dataset directory/file size and modification-time inventory is unchanged.
  No test image was opened, decoded, or used for validation. Real-view verification
  additionally passed the independent dataset snapshot check. Original metadata
  backups remain under `scratch/metadata_refresh/before/`.

Refresh command actually run from `../PCB_Lab_impl3`:

```powershell
$env:PATH = (Join-Path $env:ProgramFiles 'Git/cmd') + ';' + $env:PATH
& .venv/Scripts/python.exe -B scripts/refresh_yolo_metadata.py --source-commit 7623c11d88e4502c879d6f0efa71aab4ae495839 --bundle exports/colab_bundle.zip --device cpu --batch 16
```

The implementation environment lacks pytest. Tests reused the existing pytest
9.1.1 installation in `PCB_YOLO_EfficientAD_Lab/.venv/Lib/site-packages` by appending
that directory to `sys.path`, after keeping the implementation environment's
Torch/Ultralytics ahead of it. The local helper
`PCB_YOLO_EfficientAD_Lab/.cache/phase_c_refresh/run_checks.py` preimports the
implementation's `pcb_lab.models.yolo.train` before collecting the unchanged tests
from `PCB_Lab_verify3`. Its startup output confirms the implementation file path.
All test invocations used `-q -p no:cacheprovider --tb=short`.

| Actual verification | Result |
|---|---|
| `tests/test_yolo_metadata_refresh.py` | 6 passed |
| Requested `test_real_artifact_manifest_schema_and_file_hashes`, `test_real_artifact_calibration_schema`, `test_real_artifact_prediction_rows` | All 3 PASS for each model |
| Seven artifact/GT/AP/isolation checks, B01 | 6 passed, 1 failed, 2 deselected |
| Same seven checks, B02 | 7 passed, 2 deselected |
| Regression + Verifier view/config suites | 31 passed, 4 failed |
| Original commit loaded in memory, rerun four failing view/config cases plus top-level unknown key control | Same 4 failed, 1 passed, 24 deselected |

Extended findings (test expectations and prediction files were not changed):

1. B01 `test_real_ap50_recomputed_with_101_points` flags `open_circuit`:
   saved-prediction oracle AP50=0.9654415972399374 versus refreshed validation
   AP50=0.9863670110986422, absolute delta=0.020925413858704833 (limit=0.02).
   Re-evaluating the saved predictions with Ultralytics' own matcher and AP
   integration gives 0.9700960132123367 for this class, reducing the difference
   to 0.016270997886305483. Thus the independent step-envelope versus Ultralytics
   trapezoid convention contributes to the warning; it does not explain the
   entire prediction/validation difference. No unsupported hardware-only cause
   is claimed. Refreshed B01 pin_hole AP50=0.8182132012674659 versus the independent
   oracle 0.8187333386445058 (delta about 0.00052), resolving the original pin_hole
   discrepancy. Verifier should review the remaining open_circuit warning.
2. The four view/config failures predate this refresh: rebuilding a smaller view
   leaves stale samples, and unknown nested keys under train/selection/infer are
   not rejected. Reproduced with source from the original commit in memory;
   target files and checkpoint bytes were not reverted or rewritten for that check.

Machine-readable evidence: `reports/phase_c_metadata_refresh.json`,
`reports/phase_c_B01_yolo11n.xml`, `reports/phase_c_B02_yolo11s.xml`, and
`reports/phase_c_baseline_checks.xml`. Validation log stays local at
`scratch/metadata_refresh.log`. Ready for the requested final Verifier review,
with the extended findings explicitly retained.
