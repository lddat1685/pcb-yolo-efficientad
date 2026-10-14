# Xác minh VERIFIER — Step 3: Baseline YOLO

## Giai đoạn A — viết test từ spec

Ngày: 2026-09-25. Branch đích: `step3/tests`, worktree `../PCB_Lab_verify3`.
Base: `2a1545e`. Workspace mở ban đầu là `main` ở `PCB_YOLO_EfficientAD_Lab`; không sửa main.

Đã đọc `docs/handoff/WORKFLOW.md`, `step3-contract.md`, `step2-verify.md`, hợp đồng Step 2
để tái dùng API, fixture/test Step 1–2, `docs/DECISIONS.md`, `docs/TRIAGE.md`, và các mục plan
được yêu cầu. Giám sát xác nhận plan chính thức là `PLAN.md` tại gốc; `docs/plan.md` không tồn tại.
Handoff Step 2 hiện có chỉ ghi Giai đoạn A, **không suy diễn Step 2 đã PASS Giai đoạn B**.

Không đọc source/handoff Step 3 implementation; chưa merge `step3/impl`. Chỉ dùng spec và
giải đáp giao diện của giám sát trong hội thoại, được chép tại `questions-verify.md`.

### Phạm vi và oracle

| File | Điều kiện được bảo vệ |
|---|---|
| `tests/conftest.py` | Đăng ký `smoke`, `artifacts`; thiếu `STEP3_ARTIFACT_DIR` tự skip, đường dẫn đã cấp nhưng sai thì fail. Smoke thiếu dependency skip rõ lý do. Chặn truy cập mạng Python trong test YOLO. |
| `tests/helpers/step3_spec.py` | Oracle độc lập đọc manifest/classes/hash/mtime; kiểm schema, nhãn, số lượng và bbox; fake engine không torch; checkpoint giả chỉ dùng kiểm guards. |
| `tests/test_yolo_config.py` | Trainer từ chối riêng từng null cả normal/smoke; typo cấp cao/lồng nhau; `_check_nulls` có positive control; `_merge_configs` từ chối khóa lạ; B01/B02 chỉ khác ba trường identity; hash ổn định, độc lập thứ tự khóa/seed, đổi infer thì hash đổi. |
| `tests/test_yolo_view.py` | Chặn fusion/test cả trong tuple hỗn hợp; YAML không test, đúng class order; good label rỗng; 6 chữ số thập phân; round-trip ≤0.01 px với `Sample.boxes` và manifest; copy/hardlink cùng nội dung/signature; limit chọn đầu sample_id; rebuild không giữ mẫu thừa; giả lập EXDEV; đổi một ảnh fixture đổi signature. |
| `tests/test_yolo_adapter.py` | Engine nhận RGB bất đối xứng kênh; describe đúng metadata; box 640×640, 800×600, 300×700 theo tọa độ tính tay; sort confidence/class/box; image_score max hoặc 0; thiếu file, checkpoint đổi một byte, smoke opt-in, class order engine sai. |
| `tests/test_yolo_extract.py` | Chặn test/train/partition lạ trước nạp artifact; dùng adapter thật + engine giả; một dòng mỗi ảnh, sample_id sorted, schema/provenance/timing null; lỗi ảnh còn dòng và error_reason, score null; inference rỗng hợp lệ có score 0; hai lần xuất giống byte. |
| `tests/test_yolo_artifacts.py` | Marker artifacts: schema run/artifact/calibration, checkpoint và manifest hash khớp file thật; smoke=false, git_dirty=false; đủ 460/306 dòng, không trùng/lỗi, conf_floor, báo saturation max_det. |
| `tests/test_yolo_smoke.py` | Marker smoke+dataset: CPU, một epoch, pretrained không tồn tại, view nhỏ, reload checkpoint bằng Ultralytics thuần; nạp adapter với allow_smoke qua bundle verifier dưới runs/smoke; so RGB adapter với predict(path) ≤0.5 px / 1e-3 conf; không sinh artifacts/ chính thức. |

Mỗi hàm test có comment liên kết plan/contract. Module Step 3 chỉ import trong test/fixture;
thiếu implementation phải fail khi chạy, không bị `importorskip` che thành PASS.

Test real-view giữ nguyên oracle plan: train **1.792 = 895 good + 897 defect**, calibration
**460 = 230 + 230**; bbox train theo lớp **1.216, 973, 1.228, 981, 884, 865**. Hash ảnh thực
trong view phải rời hash fusion/test lấy từ metadata. Các số này là kỳ vọng, chưa phải kết quả đo A.

Snapshot dataset so cả đường dẫn, mtime_ns, size, SHA-256 trước/sau, bao gồm file cache.
Riêng ảnh test và ảnh không xác định quyền đọc chỉ stat, không mở/hash/decode theo WORKFLOW.
Việc duyệt cây bằng os.walk chỉ để phát hiện ghi file; mẫu view/extraction được chọn qua manifest.

Giá trị `lr0=0.001`, `batch=2`, `workers=0`, augmentation tắt trong helper là **cấu hình test**,
không phê duyệt recipe train thật. Không hạ ngưỡng/kỳ vọng để ép số liệu khớp plan.

### Lệnh và kết quả thực chạy

Môi trường A: Python **3.14.7** từ `.venv` có sẵn của repo gốc. Venv đó có pytest/Pillow/PyYAML
nhưng thiếu numpy; chỉ bổ sung đường dẫn site-packages của venv Step 2 có sẵn để lấy numpy,
không đọc source implementation và không cài/sửa môi trường của implementer. Torch/Ultralytics chưa có.
**Đây không thay thế venv riêng + requirements CPU của Giai đoạn B.**

```powershell
$py = './.venv/Scripts/python.exe'
$env:PYTHONPATH = '../PCB_Lab_impl/.venv/Lib/site-packages;../PCB_Lab_verify3/src'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Vòng kiểm bản nháp tại $env:TEMP/pcb-step3-verifier-a trước khi ghi worktree:
& $py -m pytest tests --collect-only -q -p no:cacheprovider --rootdir=. --confcutdir=tests
& $py -m pytest tests -m 'not dataset and not smoke' -q --tb=no -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=./.cache/pcb-step3-verifier-a/pytest-fast-a1
& $py -m pytest tests -m smoke -q -rs -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=./.cache/pcb-step3-verifier-a/pytest-smoke-a1
```

| Kiểm tra bản nháp | Kết quả thật |
|---|---|
| Collection | **51 tests collected**, exit 0 |
| Fast | **46 failed, 2 skipped, 3 deselected**, 51.24 s; chưa có module YOLO Step 3 |
| Marker smoke | **1 skipped, 50 deselected**; thiếu torch, ultralytics; chưa train |
| Kiểm riêng oracle bằng script tạm stdlib/Pillow + loader Step 2 | Parse Python/comments PASS; nhận view hợp lệ + round-trip PASS; phát hiện cache mới PASS; từ chối class 6 PASS; fixture train/calibration mỗi tập 3 ảnh (1 good, 2 defect) |

Lần gọi pytest ban đầu bị sandbox chặn đọc thư mục cha `<user-home>`; lần smoke dùng tmp mặc định
cũng gặp quyền truy cập `pytest-of-ASUS`. Đã chạy lại bằng root/confcutdir và basetemp riêng với quyền
được cấp, nhận kết quả ở bảng trên. Không tính lỗi môi trường đó là lỗi implementation.

Vòng cuối chạy **trên worktree `step3/tests`**, sau khi đưa các file đã kiểm vào đúng branch:

```powershell
Set-Location '../PCB_Lab_verify3'
$env:PYTHONPATH = '../PCB_Lab_impl/.venv/Lib/site-packages'
$env:STEP3_ARTIFACT_DIR = '' # A chỉ kiểm skip; chưa nghiệm thu artifact thật
$specTests = @(Get-ChildItem tests/test_yolo_*.py | ForEach-Object FullName)
& $py -m pytest @specTests --collect-only -q -p no:cacheprovider --rootdir=. --confcutdir=tests
& $py -m pytest @specTests -m 'not dataset and not smoke' -q --tb=no -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=./.cache/pcb-step3-verifier-a/pytest-fast-final --junitxml=./.cache/pcb-step3-verifier-a/fast-final.xml
& $py -m pytest @specTests -m 'smoke or artifacts' -q -rs -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=./.cache/pcb-step3-verifier-a/pytest-optional-final
git diff main -- docs/handoff/step3-contract.md docs/handoff/WORKFLOW.md
git diff --check
```

- Collection Step 3: **51 tests**, exit 0.
- Collection toàn repo với `pytest tests --collect-only -qq -p no:cacheprovider --rootdir=. --confcutdir=tests`:
  **125 tests** (74 test có sẵn + 51 Step 3), exit 0; không chạy hành vi Step 1–2 trong lượt này.
- Fast: **46 failed, 2 skipped, 3 deselected**, **25.82 s**, exit 1.
  Đọc XML bằng `xml.etree.ElementTree` xác nhận **46/46 lỗi cùng nguyên nhân**
  `ModuleNotFoundError: No module named 'pcb_lab.models'`; **0 setup errors**.
  Không đổi test để che lỗi import; chờ merge ở B.
- Optional: **4 skipped, 47 deselected**, exit 0; ba test artifacts thiếu biến môi trường,
  một smoke thiếu torch/ultralytics. Đây là kiểm cơ chế skip, không phải PASS hành vi model.
- Diff hai file bất biến **rỗng**; `git diff --check` exit 0 (chỉ cảnh báo LF/CRLF của Windows).

Bản nháp và log/XML kiểm tra chỉ ở thư mục tạm, không commit. Trước commit kiểm `git status`;
chỉ stage 8 file thuộc `tests/` và 2 file thuộc `docs/handoff/`.

### Giới hạn và điểm bàn giao

- Chưa có kết luận PASS/PASS-WITH-CONDITIONS/FAIL cho implementation. Test đỏ A là dự kiến;
  collection thành công không chứng minh mô hình đúng.
- Test dataset thật, CPU train/reload và artifact thật **chưa được thực thi thành công**.
- Một số spelling/type schema chưa chốt được ghi rõ ở `questions-verify.md`; không sửa contract.
- Bắt lỗi bằng fake engine không chứng minh Ultralytics xử lý RGB đúng; smoke và C so với
  Ultralytics predict(path) mới kiểm ranh giới thư viện thật. Checkpoint ngẫu nhiên có thể trả rỗng.
- Chặn socket Python không phải chứng minh tuyệt đối không có mạng qua subprocess/native library;
  cần code review Giai đoạn B cho auto-download/cache/side effect.
- Không thay `step3-contract.md`, `WORKFLOW.md`, `PLAN.md`, code impl, requirements; không commit
  dataset, ảnh, checkpoint, dự đoán hay log chạy test.

## Giai đoạn B — kiểm implementation

Ngày 2026-09-25, người dùng cho phép đọc và merge. Đã merge **`step3/impl@b646589`** vào
`step3/tests` bằng `git merge --no-edit step3/impl`; merge commit **`3b78289`**, không xung đột.
Không merge main, không sửa source implementation.

### B.1. Môi trường và phạm vi kiểm

- Venv riêng: `./.cache/step3-verifier-b/.venv`.
  Cài **nguyên `requirements/step3.txt` + pytest** bằng uv, không dùng site-packages venv impl.
- Python 3.14.7; torch **2.14.0+cpu**, torchvision **0.29.0+cpu**, Ultralytics **8.4.161**,
  numpy 2.5.3, Pillow 12.3.0, PyYAML 6.0.3, pytest 9.1.1. `torch.version.cuda=None`, CUDA unavailable.
- Cài đặt ban đầu bị sandbox chặn socket; chạy lại có quyền mạng để **cài dependency** thành công.
  Mạng trong test YOLO vẫn bị chặn bằng fixture; `YOLO_OFFLINE=true` tránh DNS probe lúc import.
- Đọc diff `main...step3/impl`, toàn bộ bốn module YOLO + CLI/config, notebook, packager, RUN_GPU,
  các API Step 1–2 liên quan, và **source Ultralytics của đúng bản cài** để kiểm hành vi thư viện.
- `git diff main -- docs/handoff/step3-contract.md docs/handoff/WORKFLOW.md` **rỗng**.
  Diff implementation không có checkpoint/prediction/ảnh/zip đã commit; số dataset chỉ nằm trong
  docstring/handoff, không dùng làm oracle đếm trong code chạy view.

Lệnh chính (PowerShell, cwd `../PCB_Lab_verify3`):

```powershell
$base = './.cache/step3-verifier-b'
$py = "$base/.venv/Scripts/python.exe"
uv --cache-dir "$base/uv-cache" venv --python 3.14.7 "$base/.venv"
uv --cache-dir "$base/uv-cache" pip install --python $py --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements/step3.txt pytest
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:DATASET_ROOT = '../DatasetVer4_Public'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_CONFIG_DIR = "$base/settings"
$env:MPLCONFIGDIR = "$base/mpl"
$env:PATH = (Join-Path $env:ProgramFiles 'Git/cmd') + ';' + $env:PATH
$env:STEP3_ARTIFACT_DIR = ''
# Các lượt dùng tên basetemp/xml/log riêng: fast-original, fast-corrected, full.
& $py -m pytest -m 'not dataset and not smoke' --tb=short -q -p no:cacheprovider --basetemp="$base/fast-original-tmp" --junitxml="$base/fast-original.xml"
& $py -m pytest -m 'not dataset and not smoke' --tb=short -q -p no:cacheprovider --basetemp="$base/fast-corrected-tmp" --junitxml="$base/fast-corrected.xml"
& $py -m pytest --tb=short -q -p no:cacheprovider --basetemp="$base/full-tmp" --junitxml="$base/full.xml"
```

Các lệnh thực tế redirect stdout/stderr vào các file `.log` cùng prefix, không commit log/checkpoint.
Trước lượt full, tạo settings dir và **copy font Arial có sẵn từ Windows** vào
`settings/Ultralytics/Arial.ttf`. Đây là chuẩn bị của verifier để đi tiếp sau khi đã bắt được
network request tìm font ở lượt đầu, **không phải implementation đã sửa lỗi mạng**.

### B.2. Sửa test chỉ khi có bằng chứng test sai

| Sửa trong verifier | Bằng chứng và phần kỳ vọng giữ nguyên |
|---|---|
| `names` chấp nhận khóa YAML số hoặc chuỗi số; zero-count class có thể vắng trong `box_counts` | Contract chốt index/thứ tự/số lượng, không chốt representation. Chuẩn hóa rồi vẫn so đủ sáu lớp và đúng số box. |
| Gọi `detector.image_score` | `adapter.py:169` cung cấp method; contract không bắt buộc module-level function. Kỳ vọng max/empty=0 giữ nguyên. |
| Mock thêm `ultralytics.YOLO` khi checkpoint chứa bytes giả | `adapter.py:119–138` deserialize **trước** constructor engine. Mock cũ làm UnpicklingError dù guard có thể đúng. Giữ nguyên `from_artifact`, hash, smoke guard, so `model.model.names`; chỉ thay deserializer/engine. |
| Failed-row score không bắt buộc null nếu đã có `error_reason` | Contract yêu cầu không bỏ dòng/không GOOD im lặng, không định kiểu score khi lỗi. Test độc lập xác nhận lỗi rõ ràng và đủ dòng; không gọi score=0 kèm error là lỗi “im lặng”. |
| Smoke kiểm `.yaml` + pretrained SHA null, không ép `args.pretrained=False` | Source Ultralytics `engine/trainer.py:858–865`: YAML + boolean True vẫn khởi tạo không weights. Không đổi yêu cầu random init/no-download. Chuyển kiểm schema xuống sau reload để không mất bằng chứng reload khi metadata sai. |
| Sửa `sys.environ` thành `os.environ` trong hai test CLI Step 2; `os_pathsep()` chưa định nghĩa thành `os.pathsep` | AttributeError/NameError đã tái hiện; đây là lỗi test, không phải lỗi implementation. Hai CLI được chạy lại thành công. |
| Ghép bbox một-một theo class và dung sai thay vì sort rồi zip | Hai GT có cùng x1=261 bị đảo thứ tự khi label sáu chữ số làm x1 thành 260.99968/261.00032 (`deeppcb_12100121_defect`), khiến zip báo lệch 407 px. Đo lại tọa độ chỉ lệch tối đa 0.00032000000010157237 px. Dùng ghép một-một, giữ nguyên số box và ngưỡng **0.01 px**; regression vẫn bắt box lệch 1 px. |
| Chặn model creation trong test config bị từ chối | Lượt đầu typo lồng nhau lọt tới trainer/font download. Sentinel cho lỗi chính xác tại ranh giới, tránh vô tình train ở một test “nhanh”; vẫn yêu cầu đúng ConfigError. |
| Tạo settings directory trước import smoke | Ultralytics 8.4.161 yêu cầu parent tồn tại, nếu không có thể fallback ngoài thư mục dự kiến. Đây là sửa fixture cô lập import. |

Các test phá hoại bổ sung nằm trong `tests/test_yolo_adversarial.py`. Không giảm oracle plan,
ngưỡng tọa độ/confidence hoặc sửa implementation để làm xanh test.

### B.3. Kết quả chạy

| Lượt | Kết quả thật |
|---|---|
| Fast nguyên test A, sau merge/cài đúng CPU deps | **14 failed, 90 passed, 11 skipped, 10 deselected**, 43.02 s |
| Fast sau sửa binding/test sai và bổ sung 13 ca phá hoại | **15 failed, 102 passed, 11 skipped, 10 deselected**, 118.90 s |
| Full (thêm hai probe pretrained/resume; gồm dataset và smoke) | **20 failed, 108 passed, 12 skipped**, 297.14 s |
| Targeted sau sửa oracle bbox và test CLI, thêm kiểm tính tất định độc lập schema | **3 failed, 11 passed**, 50.76 s; hai lỗi view thật, một NameError `os_pathsep` của test cũ |
| Chạy lại riêng CLI sau sửa `os_pathsep` | **1 passed**, 9.51 s |
| Collection cuối sau mọi chỉnh sửa test | **142 tests collected**, 0.40 s, exit 0 |

Không ghi một lượt full mới chưa chạy: trong 20 failure của full, hai failure thuộc test
(CLI cũ và cách ghép bbox) đã được sửa có bằng chứng rồi kiểm lại. **18 ca còn lại là lỗi
implementation**, bao gồm smoke fail metadata sau khi train/reload/predict đã hoàn tất.
Lượt targeted chạy toàn bộ `test_yolo_view.py` và xác nhận real-view PASS: số ảnh/box đúng,
hash ảnh view rời fusion/test, snapshot nguồn không đổi, round-trip trong ngưỡng.
Hai test thêm cuối xác nhận oracle ghép box và extraction hai lần giống byte cũng PASS.

Các node chạy lại (cùng các biến môi trường B.1, log/XML `targeted` và `loader-recheck`):

```powershell
& $py -m pytest tests/test_yolo_view.py tests/test_integration_real.py::test_check_loaders_deterministic tests/test_preprocessing.py::test_prep_preview_cli_matches_api tests/test_yolo_adversarial.py::test_roundtrip_oracle_handles_equal_x1_without_relaxing_tolerance tests/test_yolo_adversarial.py::test_extraction_byte_determinism_independent_of_schema --tb=short -q -p no:cacheprovider --basetemp="$base/targeted-tmp" --junitxml="$base/targeted.xml"
& $py -m pytest tests/test_integration_real.py::test_check_loaders_deterministic --tb=short -q -p no:cacheprovider --basetemp="$base/loader-recheck-tmp" --junitxml="$base/loader-recheck.xml"
```

Smoke thực tế: **1 epoch CPU, 16 ảnh train + 16 ảnh calibration**, không tạo `output/artifacts/`.
Checkpoint được nạp bằng Ultralytics thuần, kiểm names và so adapter RGB với predict(path)
đạt **≤0.5 px / ≤1e-3 confidence**; cuối test mới fail vì `git_commit=null`.
Snapshot dataset thật trước/sau smoke không đổi; cache nhãn nằm dưới output/data_refs,
không nằm trong DATASET_ROOT. AP của smoke đều 0; đây không phải đánh giá chất lượng baseline.

Skip artifact là chủ ý: B chưa có artifact train thật để nghiệm thu C. Chín test augmentation
Step 2 có sẵn tự skip vì binding A của Step 2 không nhận API `apply_augmentation(partition=...)`;
không tính những skip này là PASS. Test riêng Step 3 về augmentation chạy thật và phát hiện lỗi.

### B.4. Bảng phát hiện

Các line dưới đây tham chiếu source sau merge, cùng nội dung `step3/impl@b646589`.

| ID / mức | File:dòng | Bằng chứng / hệ quả | Đề xuất cho IMPLEMENTER |
|---|---|---|---|
| B-C01 **Critical** | `src/pcb_lab/models/yolo/train.py:103–113`, `:139–146` | `_geometric_augment(..., flip=True)` đổi `cx` của bbox nhưng không lật pixel. Probe ảnh bất đối xứng sai **36/300 giá trị kênh**, còn box đã chuyển sang nửa kia. Nhánh ảnh defect bị lệch ảnh–GT. | Dùng lại augmentation Step 2 hoặc bảo đảm xoay/lật đồng nhất ảnh và box; kiểm cả good/defect. |
| B-C02 **Critical** | `train.py:329–360`, `:398–401` | Pretrained tương đối được kiểm tồn tại trước `chdir(runs_base)` rồi truyền nguyên chuỗi sau chdir. Đường dẫn local hợp lệ trở thành đường dẫn thiếu; nhánh YOLO loader có thể tự download dù `allow_download=False`. | Resolve checkpoint thành absolute trước chdir; kiểm lại đường truyền vào trainer và khóa tải phụ. |
| B-C03 **Critical** | `train.py:347–353`; `extract.py:41–42` | Với dataset **giả**, train tạo `ILLEGAL/runs/smoke/.../seed42` rồi mới bị view guard chặn; extract ghi thẳng `ILLEGAL/preds_calibration.jsonl` trong dataset. | Guard tất cả output ngay đầu, trước bất kỳ mkdir/write; áp cả resolve/junction và model_id/path escape. |
| B-C04 **Critical** | `train.py:33–39`, `:301`, `:377`; Ultralytics `data/utils.py:667`, `utils/checks.py:452`, `:1020` | Lượt đầu đã bắt request `requests.head` tìm font khi allow_download=False. Env `ULTRALYTICS_NO_ANALYTICS` không được thư viện bản cài đọc. Review source: import có DNS probe; nhánh AMP GPU có thể nạp/tải checkpoint phụ. | Cô lập settings/cache dưới out_root, khai báo offline trước import, cấp sẵn tài nguyên và quản lý mọi download. Nhánh AMP chưa chạy được trên CPU, chỉ kết luận từ source. |
| B-M01 **Major** | `view.py:97–103`, `:157–165` | `link='copy'` vẫn gọi `os.link`; `samefile=True`. EXDEV được bắt ở helper rồi copy nhưng YoloView vẫn báo `hardlink`. | Truyền mode vào helper, không link khi yêu cầu copy; trả mode thực đã dùng. |
| B-M02 **Major** | `view.py:69–70`, `:97`, `:115` | Rebuild full→limit 1 good/1 defect báo 2 nhưng còn 3 ảnh, stale sample vẫn ở view. Ảnh đích đã tồn tại không được so hash/refresh. | Dựng staging rồi thay view có kiểm soát hoặc dọn/đối chiếu đúng manifest trước reuse. |
| B-M03 **Major** | `train.py:54–67`, `:304–308` | Unknown key ở train/selection/infer không bị ConfigError, lọt tới model creation; top-level và bốn null được chặn đúng. | Validate schema lồng nhau/type/recipe trước đọc dataset/tạo output. |
| B-M04 **Major** | `train.py:345–346`, `:361–383`, `:451` | Empty run directory được chấp nhận. `resume=True` chỉ bỏ guard, không truyền resume/checkpoint cho trainer; `resumed_from` luôn null. | Từ chối mọi run directory đã tồn tại trừ resume thật; nối đúng last.pt/trạng thái optimizer/epoch và provenance. |
| B-M05 **Major** | `adapter.py:160–162`, `:127–128` | Sort bỏ x2/y2 nên đảo thứ tự engine làm output khác. Model có canonical six-name prefix + lớp thứ 7 vẫn load. | Sort đủ bốn tọa độ; so class list đầy đủ và metadata/preprocessing provenance. |
| B-M06 **Major** | `extract.py:45`, `:76` | JSONL dùng `xyxy` thay `xyxy_original`. Run ID được tự dựng thay vì dùng run_manifest; fixture có run ID hợp lệ khác thì provenance không khớp. | Xuất đúng Detection schema và lấy run_id từ manifest đã xác minh. |
| B-M07 **Major** | `train.py:267–273`, `:443–451` | `_git_state` dùng `subprocess` chưa import, nuốt NameError thành `(None,None)`. `best_epoch`/`train_time_s` luôn null; epochs_run là cấu hình, early_stopped=(not smoke), args_used chỉ overrides chứ chưa phải toàn bộ args thật. | Ghi trạng thái/args/epoch/timing thật từ trainer; không nuốt lỗi provenance. |
| B-M08 **Major** | `train.py:514`, `:522–531` | pr_conf lấy từ `metrics.box.f1` (vector F1, không phải confidence), `_safe` biến thành null. Đếm GT tìm field `gt_nb` sai API bản cài. Smoke ghi `pr_conf=null` và cả sáu `n_gt_boxes=null` dù validator đã đếm 56 GT. | Lấy đúng confidence F1-optimal; đếm GT từ manifest hoặc trường đúng (`DetMetrics.nt_per_class` ở bản cài), giữ phân biệt operating threshold. |
| B-M09 **Major** | `train.py:496–506`, `:454–489`, `:577–580` | Validation cuối không truyền project/name dưới out_root; checkpoint reload bỏ project trong overrides. Không có code tạo train.log/report validation yêu cầu; model card versions/config_hash luôn n/a do đọc từ cfg không có các trường này. | Định vị mọi output và hoàn thiện files/model card theo schema, ghi phiên bản/hash thật. |
| B-M10 **Major** | `docs/RUN_GPU.md:93–101`; `notebooks/train_yolo_colab.ipynb:205` | Hướng dẫn chỉ mang artifacts về và notebook zip artifacts/yolo; artifact.json tham chiếu run_manifest dưới runs/ bị bỏ. Không có lock CUDA riêng, notebook giữ torch/torchvision hiện có. | Bundle cả manifest/env/args/log có hash/path portable; khóa và ghi lại môi trường CUDA đã kiểm. |
| B-m01 **Minor** | `train.py:292–296`, `:336` | allow_download=True vẫn gọi helper chỉ raise nếu file thiếu; source chỉ ghi local hoặc chuỗi chung, không URL/releaseversion. | Thực hiện đúng tính năng đã công bố hoặc báo rõ chưa hỗ trợ; provenance ghi nguồn/phiên bản cụ thể. |
| B-m02 **Minor** | `train.py:219–226`, `:240–263`, `:411` | Hook đổi epoch augmentation không được đăng ký; `_last_dataset` bị val ghi đè. “peak RAM” hiện lấy RSS một thời điểm, fallback dùng sys chưa import. | Nối hook thực và đo peak đúng method, hoặc ghi null + method trung thực. |

### B.5. Nhật ký phá hoại

Mọi mutation và thử đường output trái phép dùng fixture riêng; không thay ảnh/nhãn dataset thật.
“PASS” ở bảng này chỉ nói phép thử bảo vệ tương ứng, không phải toàn Step 3 PASS.

| # | Cách phá / phép thử | Kết quả |
|---|---|---|
| 1 | Đặt class_id=6 trong manifest fixture | **PASS**: ValueError, snapshot nguồn không đổi. |
| 2 | Box x1=-1 ngoài ảnh | **PASS**: ValueError. |
| 3 | Box x1=x2 suy biến | **PASS**: ValueError. |
| 4 | Đổi một byte best.pt, giữ SHA metadata cũ | **PASS**: ArtifactMismatchError. |
| 5 | Đảo hai tên trong checkpoint model.names | **PASS**: ClassOrderError; giữ metadata artifact đúng. |
| 6 | Thêm tên lớp thứ bảy sau sáu lớp chuẩn | **FAIL**: loader chấp nhận prefix. |
| 7 | Yêu cầu test/fusion ở view, test/train/unknown ở extraction, cả tuple hỗn hợp | **PASS**: PermissionError; không nạp model cho extraction bị cấm. |
| 8 | Lần lượt lr0/batch/workers/augment=null, cả smoke/normal | **PASS**: ConfigError. |
| 9 | Unknown key cấp cao và nested train/selection/infer | Cấp cao **PASS**; ba nested **FAIL**, đến model creation. |
| 10 | Run directory đã có sentinel / directory rỗng | Có sentinel **PASS**, nội dung giữ nguyên; rỗng **FAIL**. |
| 11 | resume=True với đường last.pt có sẵn | **FAIL**: spy tại trainer không nhận resume/checkpoint; dừng trước train, không sửa checkpoint. |
| 12 | Dựng cùng view hai lần / đổi một ảnh fixture + metadata hợp lệ | **PASS**: signature giữ nguyên / thay đổi. Không tuyên bố đã train hai seed-run đầy đủ. |
| 13 | Ép os.link ném EXDEV (khác ổ đĩa giả lập) | Nội dung copy được nhưng **FAIL** link_mode vẫn hardlink; chưa thử bằng ổ đĩa vật lý khác. |
| 14 | Yêu cầu link='copy', so inode bằng samefile | **FAIL**: vẫn hardlink với nguồn. |
| 15 | Rebuild view đầy đủ thành limit 1 good/1 defect | **FAIL**: file ảnh/nhãn cũ không bị loại. |
| 16 | out_root/out_dir là thư mục con dataset giả | **FAIL**: train tạo directory trước reject; extraction ghi JSONL. |
| 17 | Engine đảo các box có cùng conf/class/x1/y1, khác x2/y2 | **FAIL**: thứ tự output thay đổi. |
| 18 | Ảnh bất đối xứng + flip=True trong augmentation | **FAIL**: pixel không lật, box đã lật. |
| 19 | Engine ném RuntimeError trên ảnh calibration | **PASS** ca độc lập: đủ dòng, mỗi error_reason chứa sentinel; không lỗi im lặng. |
| 20 | Pretrained tương đối có sẵn, allow_download=False; ghi nhận path khi nạp | **FAIL**: đến YOLO factory, path không còn tồn tại sau chdir; spy chặn trước nạp/download. |

### B.6. Đo độc lập bằng stdlib

`tests/helpers/step3_independent.py` chỉ dùng stdlib; không import pcb_lab/torch/Ultralytics.
Đã chạy với dataset thật, chỉ mở metadata, label train và release; không mở ảnh test.

- Manifest: train **1.792 (895/897)**, calibration **460 (230/230)**, fusion **306 (153/153)**,
  test metadata **440 (220/220)**.
- Tổng bbox train từ manifest **6.147**; đếm lại dòng nhãn train trực tiếp cũng **6.147**.
  Theo ID 0..5, cả hai cách cho **1216 / 973 / 1228 / 981 / 884 / 865**.
- SHA-256 **file** manifest: `0ea6bdb4182b9adbd19b15ca00d8a1711d668f5dc06c1cbff9d9555ace250073`.
- SHA-256 **file** release.json: `9719b5cc2e4475bb1a911a9e6fab345306b3239c084522d556b7237c83266de6`.
  Lưu ý runner `_release_files_sha` hash JSON của map files, không hash bytes release.json;
  chưa đánh đồng hai định nghĩa này thành lỗi checkpoint.

Lệnh stdlib có thể chạy lại để tái lập cả đếm nhãn và hash checkpoint:

```powershell
$smokeRun = "$base/full-tmp/test_one_epoch_cpu_train_reloa0/output/runs/smoke/B01_yolo11n/seed42"
& $py tests/helpers/step3_independent.py --dataset-root $env:DATASET_ROOT --smoke-run $smokeRun
```

Đã tính độc lập trên file thật và so với `run_manifest.json` của smoke:

| File | Bytes | SHA-256 | So manifest |
|---|---:|---|---|
| weights/best.pt | 5,468,250 | `fb3a41c6494ffb40df02f2ef98e4bc5b1ef76cde4a78b6e90009ccfc09941899` | Khớp |
| weights/last.pt | 5,468,250 | `6830e4b5aec2db292b05f78e5a36de12a45b92e5462bf3c9832a2e9e96833699` | Khớp |

Smoke view_signature: `c2a1ffca7b49f12749e4d820baabd66955e1b95610a48224831837c39c2101f9`.
Hash khớp không bù được metadata thiếu: run có `git_commit=null`, `git_dirty=null`,
`best_epoch=null`, `train_time_s=null`; thư mục run không có `train.log`.
Các phép đo này chỉ nghiệm thu hành vi smoke ở B, không phải artifact B01/B02 của C.

### B.7. Kết luận

**FAIL — chưa đủ điều kiện chạy huấn luyện thật B01/B02.** Lỗi ảnh–nhãn augmentation,
đường pretrained, bảo vệ dataset, view tái sử dụng và provenance cần IMPLEMENTER sửa.
VERIFIER không sửa code impl; các test đỏ giữ làm bằng chứng và regression cho lượt kiểm tiếp theo.

Kiểm cuối: `git diff --check` sạch; diff hai file hợp đồng/WORKFLOW so với main vẫn rỗng;
diff `src`, `configs`, `requirements`, `scripts`, `notebooks`, `docs/RUN_GPU.md` so với
`step3/impl` rỗng. Chỉ commit tests và báo cáo này. Log/XML, JSON phép đo, checkpoint smoke
và file settings do lần import đầu sinh ra được giữ dưới `.cache/step3-verifier-b`, không commit.

## Giai đoạn C — nghiệm thu artifact thật, 2026-09-25

**B01: FAIL. B02: FAIL. Chưa đóng Step 3 theo hợp đồng hiện hành.** Cả hai model nạp và suy luận
được; kết luận FAIL nói về điều kiện bàn giao/provenance/nhất quán đánh giá, không kết luận model kém.
Người điều phối yêu cầu kiểm C sau khi tải artifact GPU về. Lúc bắt đầu, branch `step3/tests`
đã ở merge commit `33c538d`, có sửa `d453536`; verifier không tự merge thêm.
Báo cáo B trước đó vẫn là FAIL, chưa có lần nghiệm thu toàn bộ B thành PASS.

### C.1. Phạm vi, môi trường và nguyên tắc kiểm

- Cwd `../PCB_Lab_verify3`; lần lượt hai thư mục
  `artifacts/yolo/B01_yolo11n/seed42` và `artifacts/yolo/B02_yolo11s/seed42`.
- Dùng venv CPU riêng của B: Python 3.14.7, torch 2.14.0+cpu, Ultralytics 8.4.161,
  pytest 9.1.1. Metadata môi trường train ghi Python 3.13.15 / torch 2.14.0+cu130 / CUDA 13.0.
- Dataset chỉ đọc. **Test có 440 ảnh (220 good/220 defect), không phải 460**;
  460 là calibration. Chỉ dùng metadata test, không mở/hash pixel/decode/suy luận trên ảnh test.
- Không sửa artifact, run, source impl, contract hay PLAN. Giữ nguyên các kiểm schema provenance
  và số liệu null; lời giải thích “zip không có .git” không khôi phục được chứng cứ nguồn mã/dirty.
- Thêm phép kiểm độc lập để lỗi schema sớm không che mất kết quả hash, reload, GT, AP và partition.
  `tests/helpers/step3_ap.py` chỉ dùng stdlib; không gọi metric của pcb_lab/Ultralytics.
  `step3_pure_reload.py` chạy trong subprocess và assert **không import pcb_lab**.

### C.2. Lệnh và kết quả thật

```powershell
$base = './.cache/step3-verifier-b'
$py = "$base/.venv/Scripts/python.exe"
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_CONFIG_DIR = "$base/settings"
$env:MPLCONFIGDIR = "$base/mpl"
$env:DATASET_ROOT = '../DatasetVer4_Public'
# Chạy lần lượt với B01_yolo11n và B02_yolo11s:
$env:STEP3_ARTIFACT_DIR = '../PCB_Lab_verify3/artifacts/yolo/B01_yolo11n/seed42'
& $py -m pytest tests/test_yolo_artifacts.py -v --tb=short -p no:cacheprovider --basetemp=.cache/step3-verifier-c/B01_yolo11n-original-tmp --junitxml=.cache/step3-verifier-c/B01_yolo11n-original.xml
& $py .cache/step3-verifier-c/audit_metadata.py
& $py .cache/step3-verifier-c/probe_rounding.py
& $py .cache/step3-verifier-c/investigate_validation.py
# Lượt cuối B01, gồm kiểm regression raw adapter và oracle AP:
& $py -m pytest tests/test_yolo_artifacts.py tests/test_yolo_ap_oracle.py tests/test_yolo_adapter.py tests/test_yolo_adversarial.py::test_all_four_coordinates_break_adapter_sort_ties -v --tb=short -o junit_family=xunit1 -p no:cacheprovider --basetemp=.cache/step3-verifier-c/B01-final-tmp --junitxml=.cache/step3-verifier-c/B01-final.xml
$env:STEP3_ARTIFACT_DIR = '../PCB_Lab_verify3/artifacts/yolo/B02_yolo11s/seed42'
& $py -m pytest tests/test_yolo_artifacts.py -v --tb=short -o junit_family=xunit1 -p no:cacheprovider --basetemp=.cache/step3-verifier-c/B02-expanded-tmp --junitxml=.cache/step3-verifier-c/B02-expanded.xml
& $py -m pytest tests/test_yolo_artifacts.py::test_real_artifact_prediction_rows -q --tb=short -p no:cacheprovider --basetemp=.cache/step3-verifier-c/B02-preds-final-tmp --junitxml=.cache/step3-verifier-c/B02-preds-final.xml
```

Các script chẩn đoán, stdout/stderr, XML và JSON kết quả nằm trong `.cache/step3-verifier-c/`,
không commit log/checkpoint. Các lệnh thật redirect stdout/stderr vào file log cùng tên lượt.
Settings/font offline đã được cấp sẵn từ B; pure reload và pytest chặn socket Python.

| Lượt | Kết quả |
|---|---|
| B01, nguyên ba test A | **3 failed**, 1.03 s: provenance null, calibration null, serialized tie |
| B02, nguyên ba test A | **2 failed, 1 passed**, 1.04 s: hai lỗi metadata |
| B01, mở rộng + oracle AP ban đầu | **4 failed, 6 passed**, 30.07 s; thêm cảnh báo AP pin_hole |
| B01, lượt cuối sau chứng minh/sửa phép kiểm serialized tie | **3 failed, 19 passed**, 40.32 s; riêng chín test artifact là **3 failed, 6 passed** |
| B02, mở rộng chín test artifact | **2 failed, 7 passed**, 35.16 s |
| B02, kiểm lại prediction rows sau sửa helper chung | **1 passed**, 1.29 s |

Không ghép các lượt thành một lần full-suite chưa chạy. Các failure cuối giữ nguyên để người
triển khai xử lý: provenance/schema cho cả hai, AP50 pin_hole của B01.

### C.3. Checksum, định dạng checkpoint và reload

Đã tính SHA-256 độc lập, so **file artifact = file runs/weights = artifact.json = run_manifest.json**
cho cả best và last, đồng thời hash run_manifest bằng giá trị artifact trỏ tới. Tất cả khớp:

| Model / file | SHA-256 |
|---|---|
| B01 best.pt | `c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05` |
| B01 last.pt | `4de28ac1fbb70c3243c33e97aec0f2fb3b99f09784cd257338ade41c4dfbbfad` |
| B01 run_manifest.json | `8ab6fcd1d304d0cdedb13977b8726985733e9bc2de2134a3e8d94d77bb6c1532` |
| B02 best.pt | `aa079248bd566b3adf8863e3b43dde93d3392307a3ef17107e4c126bf410936c` |
| B02 last.pt | `1f1ce96cf729af3f998bf6f9d4b47a5b94782088b8569ccf1627f6ea656d5850` |
| B02 run_manifest.json | `4d83dd6aa812bee5a11bc2f6b639117e557e6ebce157b45f1494ddd4de60498d` |

- Cả bốn file có cấu trúc ZIP PyTorch hợp lệ, CRC không lỗi, có data.pkl và nạp bằng YOLO thuần
  thành `DetectionModel`. Class order của **best và last** đều khớp đủ sáu lớp canonical.
- `YoloDetector.from_artifact` nạp được cả B01/B02. Trên năm ảnh calibration mỗi model,
  sai khác lớn nhất adapter RGB vs pure YOLO predict(path): **0.0 px và 0.0 confidence**.
  Năm ID: `12000001_defect`, `12000001_good`, `12000017_defect`, `12000017_good`,
  `12000038_defect` (đều có tiền tố `deeppcb_`).
- Checkpoint đã strip: epoch=-1, optimizer không còn, kể cả last.pt. Các file hợp lệ cho inference;
  không có đủ optimizer/epoch để khẳng định resume đầy đủ trạng thái train từ last.pt.

### C.4. Số liệu calibration, GT và AP50 độc lập

JSON báo đúng các tỷ lệ người điều phối cung cấp (làm tròn phần trăm):

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| B01 | 92.39% | 71.59% | 92.60% | 86.03% |
| B02 | 93.79% | 73.59% | 90.68% | 90.64% |

Đếm GT bằng **raw manifest stdlib, ManifestDataset và dòng nhãn calibration** đều khớp:

| Lớp theo thứ tự canonical | GT calibration | B01 AP50 JSON | B01 AP50 độc lập | B02 AP50 JSON | B02 AP50 độc lập |
|---|---:|---:|---:|---:|---:|
| open_circuit | 234 | 0.979363 | 0.965442 | 0.992473 | 0.995199 |
| short | 160 | 0.900008 | 0.909293 | 0.932247 | 0.937645 |
| mouse_bite | 339 | 0.958904 | 0.960742 | 0.968162 | 0.963937 |
| spur | 277 | 0.936668 | 0.946402 | 0.968878 | 0.967058 |
| spurious_copper | 269 | 0.971372 | 0.956544 | 0.947012 | 0.927298 |
| pin_hole | 265 | 0.797107 | **0.818733** | 0.818908 | 0.829348 |
| Tổng GT / macro AP | **1.544** | **0.923904** | **0.926193** | **0.937947** | **0.936748** |

Oracle riêng: sort confidence giảm dần, ghép một-một same-class trong từng ảnh bằng IoU≥0.5,
envelope precision, trung bình 101 mức recall từ 0 đến 1. Có control giải tích cho perfect/empty,
duplicate và box nhầm ảnh; không gọi evaluator implementation. Giữ thứ tự JSONL trong tie đã làm tròn.
Ngưỡng **0.02** không đổi: B01 pin_hole lệch **+0.021625856**; B02 mọi lớp và overall nằm trong ngưỡng
(spurious_copper sát ngưỡng: -0.019713816). Đây là kiểm độ nhất quán, không phải ngưỡng chất lượng.

**Đã điều tra B01 thay vì nới ngưỡng:**

1. Kiểm riêng quy ước AP của bản thư viện: `utils/metrics.py:762–793` dùng nội suy tuyến tính
   101 điểm + trapezoid với precision cuối bằng 0, khác trung bình step-envelope 101 điểm.
   Cài cách tích phân đó độc lập trong helper cho pin_hole **0.821679246**, vẫn lệch **+0.024571763**;
   đổi quy ước tích phân không giải thích hết chênh lệch.
2. Nguồn Ultralytics `engine/model.py:631` đặt `val(rect=True)` mặc định;
   `data/build.py:311` đặt pad=0.5 cho validation; `data/base.py:425` làm batch shape thành **672×672**.
   Trong khi adapter/predict trên ảnh canonical dùng **640×640**. Runner `_validate` không khóa rect.
3. Dựng view kiểm tra mới dưới `.cache/` bằng copy (không coi đó là data.yaml gốc đã thất lạc),
   chạy lại toàn bộ 460 calibration B01 trên CPU, giữ conf=.001, IoU=.7, max_det=300:

| Chẩn đoán | Shape thực đo | mAP50 | AP50 pin_hole | pr_conf F1-optimal |
|---|---|---:|---:|---:|
| val rect=True | 672×672 | **0.9239040327276649** | **0.7971074825051101** | 0.6616616616616616 |
| val rect=False | 640×640 | **0.9341408318893346** | **0.8182132012674659** | 0.4964964964964965 |

Rect=True tái lập **đúng toàn bộ sáu AP50 trong JSON**; riêng thay rect làm pin_hole tăng
**0.021105719**. Đây là bằng chứng thực nghiệm cho khác biệt preprocessing giữa validation và
prediction; không phải lý do hợp thức hóa cùng một infer metadata cho hai đường khác nhau.
Hai đường đều dùng cùng checkpoint, GT, conf_floor/NMS; RGB adapter/path đã khớp.
GT của validator là `[234,160,339,277,269,265]`; snapshot dataset trước/sau chẩn đoán không đổi.

**Metadata null không phải giới hạn không thể lấy số liệu của Ultralytics:**
runner `train.py:517` lấy vector F1 (`metrics.box.f1`) làm pr_conf rồi `_safe` trả null;
`train.py:529` tìm `metrics.box.gt_nb`, trong khi API bản cài có `DetMetrics.nt_per_class`
(`utils/metrics.py:1176`). Chỉ số confidence có thể lấy ở cực đại đường F1 đã smooth
(`metrics.py:885`). Không tự điền các số chẩn đoán vào artifact; cần tái xuất report nhất quán.

### C.5. Predictions, tie và partition

| Model / split | Dòng | Detection | Max box/ảnh | Confidence nhỏ nhất | Dòng lỗi |
|---|---:|---:|---:|---:|---:|
| B01 calibration | 460 | 2.976 | 54 | 0.001000 | 0 |
| B01 fusion | 306 | 5.438 | 74 | 0.001002 | 0 |
| B02 calibration | 460 | 2.615 | 52 | 0.001008 | 0 |
| B02 fusion | 306 | 2.865 | 61 | 0.001002 | 0 |

Mỗi ảnh đúng một dòng, sample_id sorted/không trùng, provenance từng dòng khớp metadata,
đủ bốn trường bbox, tọa độ hợp lệ, confidence≥conf_floor, timing_ms=null, image_score=max confidence.
Không có dòng error bị giấu. Max thấp hơn 300 nên không thấy dấu hiệu max_det cắt cụt.
Intersection sample_id, image SHA-256 và source_group với metadata **440 test đều bằng 0**.

B01 có bốn cặp fusion cùng confidence sau làm tròn nhưng secondary order không tăng. Replay
bốn ảnh qua adapter xác nhận raw confidence giảm đúng; ví dụ `deeppcb_50600040_defect`:
**0.017011236399412155 > 0.017010806128382683**, cả hai thành **0.017011** khi ghi JSONL;
box replay khác tọa độ đã làm tròn dưới 0.005 px. Do đó sửa oracle JSONL chỉ yêu cầu score
không tăng; không suy ra raw tie từ số đã lượng tử hóa. Test adapter raw vẫn so đủ class/x1/y1/x2/y2
và đã chạy PASS. Không sửa hoặc sắp xếp lại prediction artifact.

Tính lại view_signature độc lập từ metadata train+calibration và nhãn sáu chữ số/LF của Colab:
`d4e75ccc65d3e3665ce5cd6fbe03c4788f5b9b69daba60ae515b31f2b69d6bd1`, khớp cả hai manifest.
`partitions_used` chỉ ghi train=1792, val=calibration; args không có đường test.
Các bằng chứng này nhất quán với không trộn test. **Không thể khẳng định tuyệt đối lịch sử Colab
chưa từng đọc test** khi thiếu source provenance, data.yaml gốc và train.log.

### C.6. Recipe, tài nguyên và phát hiện còn chặn bàn giao

Diff hai args.yaml chỉ khác `model`, `data`, `project`, `save_dir`; ba mục sau là đường chứa model_id.
Sau chuẩn hóa identity path, recipe giống nhau; cả hai batch=16, AdamW, lr0=.001, imgsz=640,
seed=42, patience=20, deterministic=true. Hai hash pretrained có ghi, nhưng file pretrained không
có trong gói và nguồn chỉ là `ultralytics-assets(github)`, thiếu URL/release cụ thể để xác minh nguồn.

| Trường | B01 | B02 |
|---|---|---|
| epochs_run trong manifest | 100 | 100 |
| Số epoch thực trong results.csv | **33** | **43** |
| Epoch có fitness tốt nhất theo CSV (đếm từ 1; fitness bản cài=mAP50-95) | **13** | **23** |
| Thời gian tích lũy dòng CSV cuối | **1260.46 s** | **1695.29 s** |
| train_time_s / best_epoch trong manifest | null / null | null / null |
| VRAM ghi nhận / method | 2160.8 MiB / torch.cuda.max_memory_allocated | 3858.7 MiB / cùng method |
| RAM ghi nhận | 3596.1 MiB | 3670.9 MiB |

CSV time là bằng chứng thời gian tích lũy train, không tự gọi là wall time end-to-end.
RAM method ghi “peak RSS” nhưng code B đã chỉ ra đo RSS một thời điểm; chưa chứng minh peak thật.
Model card có mục đích/dữ liệu/cấu hình/calibration/giới hạn N=2/không test, không khẳng định model khác,
nhưng version và config_hash vẫn `n/a`.

| ID / mức | Vị trí | Bằng chứng và việc cần xử lý |
|---|---|---|
| C-C01 **Critical** | Cả hai artifact.json/run_manifest.json | created_by_git_commit, git_commit, git_dirty đều null. Không xác minh được commit trên step3/impl hay source sạch. Cần gói source bất biến/hash gắn với run và quyết định provenance theo hợp đồng; không điền commit hiện tại hồi tố. |
| C-M01 **Major** | B01 calibration report; `train.py:_validate`; Ultralytics val defaults | AP pin_hole vượt 0.02; đã cô lập khác biệt validation 672×672 vs prediction 640×640. Khóa cùng preprocessing/infer rồi tái xuất calibration report/preds và kiểm AP lại, giữ nguyên ngưỡng. B02 có cùng rủi ro cấu hình dù AP nằm trong ngưỡng. |
| C-M02 **Major** | Cả hai calibration_per_class.json | pr_conf và toàn bộ n_gt_boxes null do lấy sai field/API. GT thực 1.544 đã xác minh ba cách; cần report đúng API và đúng pipeline. |
| C-M03 **Major** | Cả hai run_manifest.json/results.csv | epochs_run=100 trái CSV33/43; best_epoch/train_time_s thiếu. Ghi số đo/định nghĩa thật, phân biệt số epoch dự kiến/thực chạy và epoch 0/1-based. |
| C-M04 **Major** | Gói bàn giao runs/data_refs/reports | Không có train.log, data.yaml gốc, report validation yêu cầu. Không audit được toàn bộ đầu vào/log của run gốc; bổ sung bản gốc hoặc khai báo không thể phục hồi. |
| C-M05 **Major** | last.pt của cả hai | File nạp được nhưng epoch=-1, optimizer=None; không chứng minh checkpoint phục hồi đầy đủ optimizer/epoch theo PLAN. Giữ checkpoint resumable riêng nếu yêu cầu phục hồi trạng thái. |
| C-m01 **Minor** | Hai model_card.md; pretrained.source | Version/config_hash n/a, nguồn pretrained thiếu release/URL; hoàn thiện từ bằng chứng có thật. |
| C-m02 **Minor** | runs/yolo/*/seed42/weights/yolo26n.pt | Có checkpoint phụ ở cả hai run, phù hợp đường AMP helper đã nêu ở B. Thiếu log để xác nhận được cấp sẵn hay tải ngầm; không coi metadata hiện có là bằng chứng offline. |

### C.7. Kết luận và điều kiện kiểm lại

- **B01: FAIL** — runtime/checksum/predictions PASS; provenance/report chưa đủ và AP per-class
  vượt ngưỡng do validation không cùng kích thước tensor với pipeline inference.
- **B02: FAIL** — runtime/checksum/predictions và AP trong ngưỡng; vẫn thiếu provenance và
  metadata/report bắt buộc, chưa đủ điều kiện đóng C theo hợp đồng.
- Chưa đánh dấu Step 3 hoàn thành, chưa cấp PASS cho dùng chính thức ở Bước 4–6. Không cần suy
  diễn phải train lại chỉ từ các lỗi này: trước hết phục hồi bằng chứng source/run và tái đánh giá/
  xuất metadata nhất quán; Verifier sẽ quyết định phần nào cần kiểm/train lại từ bằng chứng đó.

Chỉ commit test/helper và handoff. Các artifact người dùng tải về giữ nguyên và không stage;
không tự merge. Diff WORKFLOW/contract so với main rỗng, source/config so với step3/impl rỗng.
