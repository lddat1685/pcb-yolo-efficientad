# Step 4 IMPLEMENTER — B03 EfficientAD-S

## Phase 0 — Diễn giải và thiết kế (2026-09-27)

Đã đọc WORKFLOW.md, step4-contract.md, PLAN.md §4.4 / §5 Bước 4 / §7 và DECISIONS.md R7, R8, S2, S5. Worktree `../PCB_Lab_impl4`, branch `step4/impl`; HEAD đầu phiên `a385e16` (bổ sung hợp đồng Engine và normalization_stats sau mốc main e1f37e1). Không có AGENTS.md áp dụng.

### Bất biến

- Dataset chỉ đọc, chọn ảnh qua manifest. AdTrainSet chỉ good train; kiểm tra partition, is_defect và boxes trước decode và trước từng optimizer step. Test bị PermissionError ngay từ constructor/extractor.
- TileManager Step 2: 256, stride 224, reflect, 9 tile/ảnh 640; stitch linear. Không resize ảnh PCB thành tile.
- Teacher frozen pretrained PDN-S 384 kênh; student PDN-S 768 kênh (hai nhánh); autoencoder tái tạo đặc trưng. Standalone PyTorch, không framework auto-threshold hoặc tự chia tập.
- Teacher mean/std chỉ train good. Map quantiles chỉ calibration good; calibration defect chỉ dùng báo cáo phân phối score sau khi khóa thống kê. Image score = max stitched map.
- R8: threshold tau_pixel -> opening 3x3 -> components (8-connected) -> area > min_area -> bounding rectangle -> global NMS 0.45. Confidence = max trong component, nhãn anomaly_unclassified.
- Checkpoint chứa mọi weights, teacher stats, branch quantiles và tile params. Reload phải khớp <=1e-5. Artifact kiểm SHA256 và chặn smoke theo mặc định.
- Extraction calibration/fusion: một dòng mỗi sample, thứ tự sample_id, lỗi có error_reason và score null; timing_ms null. Không truy cập pixel test.
- Smoke: 1 epoch, 4 good train (36 tile), CPU, tất cả output ở runs/smoke. Không tuyên bố smoke là baseline 70 epoch hoặc metric chất lượng B03.

### File dự kiến

- `configs/models/efficientad_s.yaml`, `requirements/step4.txt`.
- `src/pcb_lab/models/efficientad/{__init__,config,dataset,architecture,train,adapter,extract}.py`.
- `scripts/train_efficientad.py`, `scripts/extract_efficientad_predictions.py`.
- `tests/test_efficientad_impl.py`: guard dữ liệu/config, R8, checkpoint và smoke integration có dữ liệu tổng hợp.
- Bổ sung `docs/RUN_GPU.md`, `docs/handoff/questions-impl.md`, cập nhật handoff này.
- Output không commit: runs/smoke, weights, prediction JSONL. Runtime chính thức theo cây thư mục trong contract.

### Quyết định triển khai và vấn đề mở

1. Cấu hình chính giữ lr/weight_decay null đến khi có giá trị được chốt; trainer kiểm tra cả smoke. Đã hỏi người dùng chọn 1e-4/1e-5 cho B03 hoặc chỉ smoke (Q3).
2. Contract nhắc prepare_input(image, "ad"), nhưng Step 2 chỉ có "ad_tile"; dùng tên API đang có "ad_tile" vì chính là luồng native tiling đã khóa. Không sửa API Step 2/hợp đồng.
3. Dùng pretrained teacher PDN-S tương thích EfficientAD, không thay bằng teacher ngẫu nhiên khi thiếu weights. allow_download mặc định false; tải chỉ khi được bật và lưu provenance/hash.
4. Standalone recipe không dùng ImageNette penalty; ghi rõ lựa chọn này và optimizer AdamW trong manifest/model card. Không đưa ảnh ngoài PCB vào số đếm normal-only.
5. Hiện tại venv chính thiếu torch; venv Step 3 có torch nhưng sandbox chặn torch.dll (WinError 4551). Cần xác minh runtime ngoài sandbox trước smoke.
6. Workspace đích nằm ngoài writable root của phiên; mọi ghi/commit cần quyền công cụ cho đúng worktree. Không thay đổi branch main hoặc branch verifier.

### Kế hoạch xác minh

Kiểm config khóa và hash bỏ seed; metadata đếm normal-only/calibration/fusion; FakeEngine kiểm số học blending/R8; train thật smoke CPU trên manifest; save/reload đối chiếu map/score; guard hash/smoke/test; extraction kiểm dòng/hash/lỗi. Lệnh và số đo thực tế sẽ bổ sung sau khi chạy.

## Phase 1 & 2 — Triển khai và xác minh thực tế

### Cấu hình và kiến trúc đã triển khai

- Q3 được người dùng chốt trong phiên: **lr=0.0001, weight_decay=0.00001 cho B03**.
  YAML chính đã cập nhật; null, NaN, khóa lạ (kể cả khóa con), batch/tile/stride sai bị ConfigError.
  Seed bỏ khỏi canonical config hash; device CLI hợp nhất trước hash. `--device` cần cho baseline;
  smoke mặc định CPU, epoch thực tế 1.
- Standalone EfficientAD-S: PDN teacher 384 kênh (frozen pretrained), PDN student 768 kênh,
  autoencoder với bottleneck 1x1 và feature output 56x56. Loss = teacher/student hard quantile
  0.999 + teacher/AE reconstruction + AE/student distance. Không thêm ImageNette penalty,
  không augmentation; đây là recipe PCB của dự án, không tuyên bố tái lập điểm số bài báo.
- [Architecture reference, anomalib v2.0.0](https://github.com/openvinotoolkit/anomalib/blob/v2.0.0/src/anomalib/models/image/efficient_ad/torch_model.py).
  Teacher tải từ anomalib pretrained release, kiểm SHA256 archive trước khi lấy đúng small weights.
  Không có teacher thì không tự thay bằng random teacher, kể cả smoke.
- `dataset.py` tái sử dụng AdTrainSet/EvalSet + prepare_input(..., "ad_tile"). Tensor [9,3,256,256]
  float [0,1]; ImageNet normalization đúng một lần trong model. Kiểm metadata normal-only trước
  decode, trước teacher/map statistics và trước optimizer step. Test/partition lạ bị chặn từ API.
- Teacher mean/std tính streaming float64 từ train good. Branch map quantiles tính chính xác
  bằng NumPy trên toàn bộ raw feature pixels của calibration good, không subsampling; normalize
  từng nhánh bằng `0.1*(map-q90)/max(q995-q90,1e-12)`, lấy trung bình, pad 4 rồi nội suy map về
  256. Scalar quantile_min/max/mean/std trong metadata mô tả raw combined calibration map;
  branch_quantiles lưu riêng và nằm trong checkpoint. Không min-max lại theo từng ảnh.
- Adapter dùng Engine protocol nhận RGB uint8 NHWC, TileManager stitch float32 HxW, max map.
  Threshold `>` tau_pixel, opening 3x3 zero border, components 8-connected, area **strictly >**
  min_area; confidence max trong component, bounding rectangle exclusive upper coordinates,
  stable global NMS .45, class_id=None/class_name=anomaly_unclassified.
- Artifact có weights đầy đủ, stats và tile/infer params; kiểm checkpoint SHA256 và metadata
  khớp checkpoint khi reload; smoke bị chặn mặc định. Resume có optimizer/RNG/epoch checkpoint,
  từ chối khác recipe/hash/seed/release/smoke; không ghi đè run nếu chưa chọn resume.
- Extractor giữ một dòng mỗi ảnh, sort sample_id, kiểm image hash, lỗi trả score=null và
  error_reason; CLI exit 1 nếu có lỗi, timing_ms=null. `--limit` chỉ dùng với artifact smoke.

### Môi trường và các lỗi đã gặp

1. `git` không có trong PATH: dùng `<ProgramFiles>/Git/cmd/git.exe`.
2. Venv workspace chính thiếu torch (`ModuleNotFoundError`). Venv Step 3 torch 2.14 bị Windows
   Application Control chặn torch.dll (`WinError 4551`) cả trong và ngoài sandbox; không sửa venv đó.
3. Dùng môi trường riêng `./scratch/step4-env`:
   Python 3.12.10, torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 2.2.6, OpenCV 4.11.0.
   Cài đúng requirements/step4.txt. uv/Python WindowsApps và ghi worktree cần quyền ngoài sandbox.
4. Các quyền công cụ này đã được cấp trong phiên; không sửa main/branch verifier hoặc dataset.

### Smoke CPU thật — lần train đầu

Các lệnh dưới đây chạy tại `../PCB_Lab_impl4`; `python` là executable venv riêng
nêu trên, không phải venv Step 3. Đã chạy thật:

```powershell
python scripts/train_efficientad.py --config configs/models/efficientad_s.yaml --seed 42 --device cpu --smoke --allow-download
python scripts/extract_efficientad_predictions.py --artifact runs/smoke/B03_efficientad_s/seed42/artifact --device cpu --allow-smoke --limit 4
python -m pytest tests/test_efficientad_impl.py tests/test_tile_manager.py tests/test_preprocessing.py -q -p no:cacheprovider --basetemp runs/smoke/pytest-final-impl4
```

| Kiểm tra / số đo | Kết quả thực tế |
|---|---|
| Bộ kiểm tra cuối | **75 passed in 20.06s** (thêm 7 kiểm tra nội suy deterministic) |
| Train / tile / optimizer step | 4 good / 36 / 36; 1 epoch; 0 defect |
| Teacher stats | train good, 4 ảnh, 36 tile, 112.896 feature pixels/kênh |
| Map normalization | calibration good, 4 ảnh, 36 tile; 0 defect |
| Báo cáo score (sau normalization) | 4 good + 4 defect calibration |
| Mean training loss | 77.07076431 (smoke, không là metric chất lượng) |
| Thời gian optimizer loop | 28.037960899993777 giây; không gồm teacher stats/calibration |
| Peak RAM lần train đầu | 658.47265625 MiB, RSS lấy mẫu mỗi 50ms |
| Peak VRAM | 0.0 MiB (CPU) |
| Reload map max absolute error | **0.0** |
| Reload anomaly score absolute error | **0.0** |
| Extractor thật từ artifact | 4 calibration + 4 fusion; 0 lỗi |
| Regression Step 2 | TileManager + preprocessing trong cùng bộ 75 tests |

Provenance lần train đầu: git_commit=`24eb9075648297d8cff1cec5257fba2f8786c90b`, git_dirty=true
(implementation chưa commit khi chạy smoke). Không sửa provenance để giả là run trên clean commit.

- Dataset release SHA256: `9719b5cc2e4475bb1a911a9e6fab345306b3239c084522d556b7237c83266de6`.
- Teacher checkpoint SHA256: `a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d`.
- Config hash smoke: `e6d1c9197f63e038d336306ee5a7578ff35ddf71cbd64aeeddf4d7c5584ac33c`.
- Model checkpoint SHA256 lần train đầu: `a3fa8c5aada7adb2098e1b55cc2beaac91cde1dc2505e672c29bf9bc9d0dc3a1`.
- JSONL calibration (4) SHA256: `c45f77a4e6aca4c503c896e865dc8cd5d20d52787612fdfc704bf9eb30b7a047`.
- JSONL fusion (4) SHA256: `74fc38c04ece492bf6cd6988c73e3510f19cdf3289281a70ae17840b44c03579`.

Output model/report smoke ở `runs/smoke/B03_efficientad_s/seed42/`. Không tạo model artifact
chính thức `artifacts/efficientad/`; chỉ có cache teacher pretrained trong artifacts/pretrained.
Checkpoint, predictions và dữ liệu không commit.

### Kiểm tra độ phủ, guard và giới hạn

- Metadata dataset thật: 895 train good/8.055 tile, 460 calibration, 306 fusion; test không decode.
- Full extractor được kiểm tra trên **toàn bộ 460/306 manifest rows bằng detector giả**: đủ dòng,
  sort đúng, không lỗi; chạy lại cho SHA256 JSONL giống hệt. Kiểm thêm ảnh mất file vẫn giữ dòng,
  score=null và FileNotFoundError. Đây là kiểm tra extractor, không phải full B03 inference.
- Test có defect chèn vào AdTrainSet và calibration stats trước decode; test khóa test trước mọi
  file I/O; thiếu/null/NaN config; hash bỏ seed; teacher frozen sau backward; blending qua hai seam;
  opening loại singleton; area 20 loại/21 giữ; confidence từ component thay vì bbox; NMS với hai
  component rời nhưng bbox IoU>0.45; NaN map; checkpoint hash sai; smoke load; overwrite/resume thiếu checkpoint.
- Chưa train baseline 70 epoch hoặc chạy CUDA/Colab. Chưa có preds 460/306 từ checkpoint baseline
  B03 thật; lệnh train/extract và kiểm đếm đã có trong docs/RUN_GPU.md, chạy sau VERIFIER Giai đoạn B PASS.
- Chưa chọn tau_ad, không benchmark test, không suy diễn chất lượng từ smoke. Các số đo RAM/thời gian
  chỉ thuộc CPU smoke, không đại diện GPU/production. Không phân loại sáu lớp lỗi; giới hạn N=2 giữ nguyên.

### Kiểm tra resume và CUDA deterministic trước commit

Đã chạy thêm:

```powershell
python scripts/train_efficientad.py --config configs/models/efficientad_s.yaml --seed 42 --device cpu --smoke --resume
python scripts/extract_efficientad_predictions.py --artifact runs/smoke/B03_efficientad_s/seed42/artifact --device cpu --allow-smoke --limit 4
python -m pytest tests/test_efficientad_impl.py tests/test_tile_manager.py tests/test_preprocessing.py -q -p no:cacheprovider --basetemp runs/smoke/pytest-deterministic-impl4
```

Resume thực tế không chạy lại optimizer: vẫn 1 epoch/36 step, map/score reload error=0.0;
score của cả 8 ảnh trước/sau resume sai số tối đa **0.0**. SHA256 checkpoint thay đổi do
serialization lại, nên đã tái xuất predictions và đối chiếu hash của mọi dòng với artifact hiện tại.
Hash run_manifest đã kiểm lại với artifact.run_manifest.sha256.

- Checkpoint hiện tại sau resume: `3d211d96c3603f59cee6a5776763a485be3807e3bea821519057b2add50e1bcf`.
- preds_calibration.jsonl hiện tại (4): `07a58f8a5c6c753d6ae6a5a9add8978990a65a5c4da0fc71e241c79ebe3eb0b5`.
- preds_fusion.jsonl hiện tại (4): `1aceee4059766985a57d2b3b03f76906a5e8f99d38e09d50615be1349853d6de`.

Rà docstring package torch 2.6 đã cài phát hiện native CUDA bilinear backward không hỗ trợ
strict deterministic và sẽ RuntimeError. Đã dùng nội suy bilinear separable qua index_select
cho autoencoder trên CUDA (index_select backward có deterministic implementation). CPU giữ
native bilinear; không thay đổi đường smoke đã đo. 7 tests so sánh forward và gradient bằng
float64 với native interpolation qua đủ kích thước decoder đều PASS (atol 1e-12/1e-10).
CUDA chưa được chạy thật; đây là kiểm tra tương đương số học trên CPU, không là GPU validation.

Một lệnh audit metadata qua PowerShell `python -c` gặp NameError do quote của chuỗi `defect`
bị shell biến đổi; đã chuyển sang file tạm `runs/smoke/review_metadata.py` để tránh lỗi quoting.
Lệnh file đã PASS: đọc JSONL manifest độc lập cho train good=895/defect=897,
calibration good=230/defect=230, fusion good=153/defect=153; toàn bộ sample ID thống kê
smoke đúng split good/không boxes; `artifacts/efficientad/B03_efficientad_s/seed42` không tồn tại.

### Bổ sung sau smoke — notebook GPU và fix CUDA (HEAD `b2f1052`)

- `e22874b`: `notebooks/train_efficientad_colab.ipynb` (9 cells, JSON hợp lệ): Gate 0 kiểm CUDA,
  dataset/repo auto-extract từ Drive ZIP sang SSD local, cài torch 2.6.0+cu124 và
  `requirements/step4.txt`, train 70 epoch seed 42 `--device 0`, extract calibration/fusion,
  verify đủ 460/306 dòng không lỗi, đóng gói ZIP artifacts/runs về Drive.
- `b2f1052`: `notebooks/train_efficientad_kaggle.ipynb` (9 cells, JSON hợp lệ): cùng recipe cho
  Kaggle (dataset read-only từ `/kaggle/input`, codebase copy sang `/kaggle/working`,
  output ZIP ra `/kaggle/working` để tải từ tab Output khi Save & Run All).
- `328f21d` fix(train): `torch.cuda.set_device(selected_device)` trước `manual_seed_all`;
  bọc `reset_peak_memory_stats` và `max_memory_allocated` trong try/except có fallback.
  Chỉ chạm CUDA path khi `device=cuda`; smoke CPU và toàn bộ math path giữ nguyên
  (`git diff 328f21d..b2f1052 -- src/ configs/ scripts/ requirements/` rỗng +
  notebook-only ở `b2f1052`).
- Tự kiểm ở HEAD `b2f1052` (Python 3.12.10, torch 2.6.0+cpu,
  `DATASET_ROOT=../DatasetVer4_Public`):
  - Suite spec Verifier (mượn tạm 5 file test + `test_efficientad_phase_b.py` từ
    `step4/tests`, chạy trong workspace này rồi xóa, không commit):
    **84 passed, 5 skipped** (artifact-thật opt-in `STEP4_ARTIFACT_DIR`), 1 deselected.
  - Phase B `real_cpu_smoke` với `STEP4_TEACHER_PATH=artifacts/pretrained/efficientad_teacher_s.pt`:
    **1 passed (~341s)**: teacher stats và map quantiles đối chiếu oracle độc lập,
    contamination guards, pre-save probe reload delta ≤1e-5.
  - Suite 75 tests (`test_efficientad_impl` + `test_tile_manager` + `test_preprocessing`):
    **75 passed**.
- Chưa train baseline 70 epoch; chưa chạy CUDA thật. Baseline chạy sau VERIFIER
  Giai đoạn B PASS theo contract (`docs/RUN_GPU.md` §5).
