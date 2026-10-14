# Bàn giao IMPLEMENTER — Step 1

## Phase 0

### Căn cứ và mục tiêu

- Branch đã kiểm tra: `step1/impl`. Không có AGENTS.md trong workspace hoặc các thư mục cha; AGENTS.md người dùng đọc được nhưng rỗng.
- Đã đọc `PLAN.md` §1, §4.1–4.2, §5 Bước 0–1, §6, §11; `docs/DECISIONS.md`, `docs/TRIAGE.md`, `docs/handoff/step1-contract.md`. Không dùng archive, không cần đọc Review.
- Sai khác đường dẫn nhỏ: repo không có `docs/plan.md`; `PLAN.md` tại gốc là bản v3 có đủ các mục được yêu cầu. TRIAGE xác nhận tài liệu gốc ở root. Dùng file này làm căn cứ, giữ nguyên nội dung. Hợp đồng hiện có escape Markdown (`\\_`, `&#x20;`); đọc theo nghĩa hiển thị, không sửa hợp đồng.
- Bước 1 xác minh bản canonical có thể dùng cho thí nghiệm: ảnh/nhãn/hash đúng, số lượng và membership khớp manifest/holdout, không giao nhóm nguồn/cặp/hash giữa bất kỳ hai partition. EDA mô tả phân phối lớp, nhóm, số lỗi, kích thước box và rủi ro shortcut; quan sát overlay train/calibration đủ sáu lớp.
- Đầu vào: dataset chỉ đọc; manifest, holdout, classes, protocol, release đã khóa, các đường dẫn ảnh/nhãn được manifest chỉ định. Không tự ghép cặp hoặc chia lại tập.
- Đầu ra: `reports/data_audit.json`, `reports/data_profile.md` tiếng Việt, `reports/overlays/<partition>/<class_name>/*.png`, API và CLI đúng hợp đồng.
- Hoàn thành triển khai khi đã chạy thật, báo mọi FAIL/WARN và mọi sai khác với PLAN §1.2–1.3, kiểm tra tính tất định, xem overlay đủ sáu lớp. Điều kiện nghiệm thu dữ liệu của plan còn yêu cầu giải quyết lỗi nghiêm trọng và không rò rỉ; nếu audit FAIL phải báo nguyên trạng, không sửa dataset để làm PASS.

### Metadata thật đã đọc

`DATASET_ROOT` chưa được đặt trong process/user/machine. Dataset tồn tại ở mặc định hợp đồng `..\DatasetVer4_Public`. Git có tại `<ProgramFiles>/Git/cmd/git.exe` nhưng không nằm trên PATH. Python cũng chưa có trên PATH; môi trường cũ được README nhắc tới không chạy được trong sandbox, không sửa môi trường đó.

Đã đọc README.md, DATASET_CARD.md và các metadata canonical bên dưới `benchmarks/deeppcb/`:

- `manifests/samples.jsonl`: mỗi dòng một object; `sample_id`, `split`, `source_group`, `pair_ids` (list), `is_defect` (bool), `image`, `label`, `width`, `height`, `sha256`, `label_sha256`, `pixel_sha256`, `boxes`. Box gồm `source_class_id`, `class_id`, `class_name`, `xyxy` pixel. Có metadata provenance/quality/transform và danh tính bo/design để null.
- `configs/classes.json`: `names` là list theo ID YOLO 0-based; `source_to_class_id` ánh xạ nguồn 1-based. Sáu tên: open_circuit, short, mouse_bite, spur, spurious_copper, pin_hole. Không gộp pin_hole với missing_hole.
- Nhãn `.txt` thực tế: mỗi dòng `class_id cx cy w h`, chuẩn hóa theo kích thước ảnh, 10 chữ số thập phân. Good dùng file rỗng. Đối chiếu với box manifest, không chỉ tin một nguồn.
- `splits/holdout.json`: `partitions`, `groups`, `samples`, `pair_ids`, `actual_pair_counts`, thông tin lựa chọn và component groups. Danh sách sample là chuẩn đối chiếu membership, không chỉ đối chiếu tổng. Bốn partition train/calibration/fusion/test; số lượng metadata lần lượt 1792/460/306/440 và 897/230/153/220 pair; đây chưa phải kết quả audit.
- Một canonical good có thể đại diện nhiều pair. Dataset card giải thích ba template trắng trùng pixel được giữ một đại diện, hai alias; template lẻ 90100034 không xuất canonical. Không giả định mỗi pair phải có đúng hai canonical image.
- `configs/protocol.json`: good train cho AD; calibration chọn ngưỡng; fusion của nguồn dành cho fit head; test cuối; không pixel mask; source_group không phải design/bo đã xác minh; giữ good nền trắng.
- `release.json`: `files` là map đường dẫn tương đối dataset root → SHA-256 (18 entry lúc đọc); có dataset_signature, tham chiếu inventory/manifest. Sẽ tính n_locked từ map, không hard-code 18.
- Đã đọc đoạn định nghĩa hash trong script kiểm chứng nguồn chỉ để xác định schema: pixel hash = SHA-256 của byte RGB liên tiếp theo hàng, không gồm header/kích thước. Không chạy script nguồn hay ghi dataset.

### Quyết định ảnh hưởng phạm vi

- R1: giữ split source-group, không LOGO-CV; báo số nhóm, giới hạn độc lập thống kê; không gọi source_group là bo đã xác minh. PLAN v3 nói bootstrap không giải quyết domain shift.
- R7/S2/V2: đo box nhỏ thực tế, báo tác động hình học của resize 640→256 (hệ số 0.4), không khẳng định lỗi biến mất chỉ từ bbox. Tile native 256 chưa được implement ở Step 1, không mô phỏng model.
- R2 và PLAN §6: giữ calibration riêng; giữ tên partition fusion trong báo cáo dù dự án sau dùng vai trò development_selection. Không viết protocol.yaml ở bước này.
- S5 và hợp đồng: chỉ CPU; không torch/ultralytics/anomalib, không Gate 0 hoặc training. R3/V1, R4/H2, R5/R6/R8–R10, S1/S3/S4 thuộc bước sau, không chặn audit.
- Không có pixel ground truth: không tạo mask hay metric segmentation. Good trắng vẫn được đếm; báo dấu hiệu shortcut tên file, template đã làm sạch, nguồn grayscale→RGB và lỗi tổng hợp mà không suy ra model đã học shortcut.

### Giả định nhỏ đã chốt trước code

- CLI ưu tiên `--dataset-root` > biến môi trường DATASET_ROOT > YAML mặc định. Đường dẫn YAML tương đối repo; đường dẫn trong manifest/release tương đối dataset root. Từ chối output nằm trong dataset và đường dẫn input thoát dataset.
- File metadata bắt buộc thiếu/JSON không đọc được hoặc JSONL sai cú pháp là lỗi không khởi chạy (CLI 2). Bản ghi mẫu hỏng nhưng JSON còn đọc được và ảnh/nhãn thiếu/hỏng: ghi errors, FAIL check liên quan và tiếp tục (CLI 1).
- Hash leakage tính lại từ file/pixel; thiếu khả năng tính thì check leakage tương ứng FAIL do chưa thể chứng minh tách biệt. Giao pair là giao tập tất cả pair_ids, không phải sample_id.
- EDA chi tiết kích thước và lựa chọn overlay chỉ train/calibration. Các bảng số đếm/membership lớp của test phục vụ kiểm toán, không dùng chọn ngưỡng. Ngưỡng cạnh nhỏ là `min(width,height) < 16` và `< 32` px tại ảnh gốc: tương ứng <6.4/<12.8 px sau resize-256; chỉ là dấu hiệu rủi ro hình học.
- `per_class=5` nghĩa là tối đa 5 ảnh mỗi lớp mỗi partition được cho phép, thứ tự lựa chọn cố định ưu tiên box nhỏ/vùng biên; báo thiếu mẫu hoặc lỗi render, không bỏ lỗi im lặng.
- Số lượng ảnh/lớp/nhóm và số khóa đều tính từ dữ liệu. Kích thước kỳ vọng 640 là yêu cầu check `image_size_640` trong hợp đồng.
- Không có mơ hồ đang chặn Phase 1. Nếu phát hiện xung đột contract sẽ ghi questions-impl.md và dừng phần bị ảnh hưởng.

### File dự định tạo/sửa và dependency

- Tạo `pyproject.toml`, `requirements/step1.txt`, `configs/dataset.yaml`, `src/pcb_lab/__init__.py`, `src/pcb_lab/data/{__init__,manifest,audit,eda}.py`, `scripts/audit_dataset.py`.
- Cập nhật chính handoff này; tạo báo cáo JSON/Markdown; overlay và môi trường riêng không commit. Bổ sung `.gitignore` để loại môi trường/cache/output tạm (file này đã tồn tại untracked trước phiên).
- Không sửa tests/, PLAN/DECISIONS/TRIAGE/contract; không cấu hình pytest; không thao tác branch khác.
- Dự kiến dependency runtime chỉ Pillow (decode RGB/render overlay) và PyYAML (đọc cấu hình); dùng thư viện chuẩn cho hash, bảng, percentile để tránh thêm numpy/matplotlib. Setuptools chỉ phục vụ đóng gói. Phiên bản sẽ khóa theo môi trường riêng đã cài và chạy thật ở Phase 2.

## Phase 1

Đã triển khai đúng hai API và CLI của hợp đồng, không đổi tên/kiểu/đường dẫn đã chốt:

- `manifest.py`: đọc JSONL/holdout/schema/protocol/release, phân biệt lỗi metadata không chạy được với bản ghi mẫu hỏng, chặn đường dẫn thoát dataset và output ghi vào dataset (kể cả symlink/junction).
- `audit.py`: kiểm tra toàn bộ canonical ảnh/nhãn, SHA-256 release/file/label/pixel RGB, kích thước/mode, ID lớp/tọa độ cả hai dạng annotation, đối chiếu nhãn↔manifest, uniqueness, membership/group/pair aliases với holdout; kiểm tra đủ sáu cặp partition bằng hash tính lại. Lỗi mẫu được ghi `errors` và FAIL, các mẫu sau vẫn được xử lý.
- `eda.py`: báo cáo tiếng Việt, bảng ảnh/box/lớp/partition/nhóm nguồn, histogram lỗi mỗi ảnh; percentile kích thước và mục riêng pin_hole/mouse_bite. Lựa chọn overlay theo metadata train/calibration, kiểm tra holdout và từ chối ảnh còn được tham chiếu ở test/fusion trước khi mở ảnh.
- CLI có `--no-overlays`; mã thoát 0/1/2 đã kiểm tra. `run_audit` chỉ sinh JSON/Markdown. CLI bổ sung check overlay vào report khi render.
- Khóa runtime **Pillow 12.1.1** (decode RGB và vẽ bbox), **PyYAML 6.0.3** (config YAML); **setuptools 80.10.2** cho build/editable install. Đã cài vào `.venv` riêng, chạy **CPython 3.14.7 Windows x86_64 / CPU**, `uv pip check` PASS. Không thêm numpy/matplotlib hoặc framework ML; không sửa môi trường cũ.
- Thêm khóa report được hợp đồng cho phép: `class_image_counts`, `eda`, `overlays`; thêm các check schema/uniqueness/hash/annotation và shortcut. Giữ nguyên tất cả khóa/check bắt buộc.
- Thêm `overlay_index.json`, `overlay_errors.json`, `overlay_warnings.json` để không bỏ mất lỗi render. Thiếu số ví dụ yêu cầu là WARN; ảnh/nhãn render hỏng là lỗi và FAIL. Không coi fixture ít hơn 5 mẫu/lớp là dữ liệu hỏng.
- Sai số đối chiếu bbox chuẩn hóa là 1e-9, tương thích nhãn ghi 10 chữ số thập phân. Đối chiếu multiset bằng ghép một-một trong sai số, không sắp theo float rồi zip: lượt tự kiểm đầu phát hiện cách sort đó báo sai 3 mẫu train có chung x1. Đã sửa lỗi implementation, không nới tolerance, không chỉnh dataset và chạy lại toàn bộ.

Các commit đã có trước bàn giao:

- `1e5aedb`: Phase 0 và cam kết trước khi viết code.
- `4756803`: môi trường CPU, skeleton/config và manifest adapter.
- `5887319`: EDA tiếng Việt và overlay có kiểm tra partition.
- `10cce10`: audit toàn bộ canonical và CLI.

Không sửa tests/, không đăng ký pytest, không sửa PLAN/DECISIONS/TRIAGE/contract, không chuyển branch/merge/push. `docs/handoff/step1-contract.md` đã untracked từ đầu và được giữ nguyên, không đưa vào commit của IMPLEMENTER.

## Phase 2

### Kết quả chạy thật

Đã chạy đúng lệnh audit yêu cầu trong môi trường riêng, rồi chạy lại vào `reports/repeat`:

```powershell
$env:DATASET_ROOT = '..\DatasetVer4_Public'
$env:PATH = (Join-Path (Get-Location) '.venv\Scripts') + ';' + $env:PATH
python scripts/audit_dataset.py --dataset-root $env:DATASET_ROOT --out-dir reports
python scripts/audit_dataset.py --dataset-root $env:DATASET_ROOT --out-dir reports/repeat
```

- **overall = PASS**, CLI exit **0**, **0 errors**, **22 check PASS**, **1 check WARN**, **0 check FAIL**.
- WARN duy nhất: `shortcut_filename`, `n_violations=2252`. Toàn bộ 2252 tên ảnh train/calibration chứa `_good` hoặc `_defect`; chỉ dùng tên để quản lý mẫu, không làm feature model. Đây không phải bằng chứng model đã học shortcut.
- 2998 ảnh được decode/hash; release **18/18** khớp SHA-256. Good trắng canonical vẫn được giữ (EDA thấy 1 good nền đồng nhất ở train/calibration).
- Cả sáu cặp train/calibration, train/fusion, train/test, calibration/fusion, calibration/test, fusion/test đều có giao **source_group=0, pair=0, image_hash=0, pixel_hash=0**.
- Đã kiểm tra schema JSON, đủ ID check bắt buộc, kiểu số nguyên, trạng thái, sáu cặp partition, errors đúng cấu trúc. Đã đọc đầy đủ Markdown, bảng tiếng Việt hiển thị đúng UTF-8.
- Hai lượt chạy cho **65 tệp giống hệt từng byte**: data_audit.json, data_profile.md, ba JSON overlay và 60 PNG. Không timestamp; out_dir không đưa vào nội dung JSON/Markdown.
- SHA-256 `data_audit.json`: `c0f52b184b306bc75b5b529df624a329e7ec0edc56e1ca81bf8535305546ecbb`.
- SHA-256 `data_profile.md`: `626f0fb1c1a496f9bca6887fb40b1a05ef13f02d011d8b51ec99072cfd2889d4`.

### Đối chiếu MỌI số đếm PLAN §1.2–1.3

**Không có số đếm nào khác plan.** Bảng dưới ghi số thực đo; mọi ô bằng giá trị tương ứng trong PLAN, không lấy giá trị kỳ vọng này đưa vào code.

| Partition | Good | Defect | Ảnh | Nhóm nguồn |
| --- | ---: | ---: | ---: | ---: |
| train | 895 | 897 | 1792 | 5 |
| calibration | 230 | 230 | 460 | 2 |
| fusion | 153 | 153 | 306 | 2 |
| test | 220 | 220 | 440 | 2 |
| Tổng | 1498 | 1500 | 2998 | 11 |

| ID | Lớp | Box train | Box test |
| --- | --- | ---: | ---: |
| 0 | open_circuit | 1216 | 298 |
| 1 | short | 973 | 215 |
| 2 | mouse_bite | 1228 | 250 |
| 3 | spur | 981 | 228 |
| 4 | spurious_copper | 884 | 194 |
| 5 | pin_hole | 865 | 222 |

Sáu lớp và ID khớp §1.3; 895 good train cho AD khớp §1.2. Số pair tính từ mọi alias cũng khớp holdout: train 897, calibration 230, fusion 153, test 220. Không giả định canonical good có quan hệ 1:1 với pair.

### Overlay và EDA lỗi nhỏ

Đã tạo **60 PNG**: 5/lớp/partition × 6 lớp × 2 partition. `overlay_errors.json=[]`, `overlay_warnings.json=[]`. Ảnh test/fusion không được xuất hoặc xem. Đã mở và xem trực tiếp sáu overlay dưới đây; bbox/nhãn hiển thị đúng, có lỗi nhỏ và box gần biên ảnh:

| Lớp | Partition | Sample đã xem |
| --- | --- | --- |
| mouse_bite | train | deeppcb_20085005_defect |
| open_circuit | calibration | deeppcb_13000017_defect |
| pin_hole | train | deeppcb_20085003_defect |
| short | calibration | deeppcb_13000007_defect |
| spur | train | deeppcb_20085000_defect |
| spurious_copper | calibration | deeppcb_13000019_defect |

Trên train+calibration:

- pin_hole: 1130 box, nhỏ nhất theo diện tích **21×23 px**; cạnh ngắn min **20 px**, P5/50/95 = **24/30/42 px**; <16 px **0%**, <32 px **63.5398%**.
- mouse_bite: 1567 box, nhỏ nhất **20×20 px**; cạnh ngắn min **20 px**, P5/50/95 = **24/28/37 px**; <16 px **0%**, <32 px **77.1538%**.
- Min cạnh sau phép tính resize-256 là 8 px ở cả hai lớp; chưa có cơ sở nói lỗi biến mất. Không dùng test để đưa ra ngưỡng mô tả này.

### Tự kiểm chứng lỗi và giới hạn

12 nhóm smoke check tự tạo trong `reports/verification/` (ignored, không thuộc tests của VERIFIER) đều PASS: fixture hợp lệ/đủ sáu cặp/tất định; chặn partition overlay trước I/O; spy Image.open chứng minh overlay chỉ mở train/calibration; ảnh hỏng, class sai, NaN/box âm và nhãn mất không làm ngừng mẫu sau; cả bốn loại leakage trên đủ sáu cặp bằng byte ảnh thật; release/membership sai; record JSON có schema hỏng và path thoát root; chặn output vào dataset; CLI 0 khi chỉ WARN; CLI 0 khi thiếu số ví dụ overlay; CLI 1 khi dữ liệu lỗi; CLI 2 khi metadata thiếu/JSONL hỏng. `compileall` và `uv pip check` PASS.

Tệp kiểm chứng tạm: `reports/verification/self_check.py`, `self_check_results.json`, `final_check_results.json`. Không đưa fixtures hay ảnh tự sinh vào commit. Kiểm thử độc lập của VERIFIER vẫn là việc chưa làm; chưa nhận findings, chưa yêu cầu đổi kỳ vọng test. Khi nhận sẽ đọc `git show step1/tests:docs/handoff/step1-verify.md` và sửa/phản biện trên branch này.

### Chạy lại / cài môi trường

Từ repo root, có thể tái tạo môi trường bằng uv và Python đúng phiên bản:

```powershell
uv --cache-dir .cache/uv venv --python 3.14.7 .venv
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements/step1.txt
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe --no-build-isolation --no-deps -e .
$env:DATASET_ROOT = '..\DatasetVer4_Public'
& .venv\Scripts\python.exe scripts/audit_dataset.py --dataset-root $env:DATASET_ROOT --out-dir reports
```

CLI trực tiếp cũng chạy được khi chưa editable install vì tự thêm `src` theo vị trí script. API import dùng editable install. Không có cấu hình pytest. Có thể thêm `--no-overlays` để chỉ kiểm toán/EDA; các khóa overlay không xuất hiện trong lần chạy đó.

### Đã làm / chưa làm / rủi ro / câu hỏi mở

- **Đã làm:** toàn bộ phạm vi Step 1 được giao và phần tối thiểu Step 0 để chạy CPU; đã chạy dataset thật, xem overlay phát triển và đối chiếu plan.
- **Chưa làm theo phạm vi:** train/inference, TileManager, protocol mới, GPU Gate 0, experiment register, test độc lập/merge. Không tuyên bố Step 0 đầy đủ hoặc model đã được kiểm chứng.
- **Rủi ro/giới hạn:** hash exact không loại trừ near-duplicate; source_group không chứng minh danh tính bo/design. Kiểm tra shortcut là dấu hiệu metadata/quan sát, không thay thử nghiệm model. Bbox chỉ giới hạn hình học của resize/tile, không phải bằng chứng về pixel lỗi còn lại. Nền good được nguồn làm sạch và dữ liệu lỗi có phần tổng hợp vẫn là giới hạn domain.
- Report đối chiếu canonical và các file release khóa, không chạy lại toàn bộ raw/export audit nguồn; không xác nhận giấy phép/chất lượng expert của mọi annotation.
- YAML liệt kê các đường dẫn contract cố định; thay dataset root được hỗ trợ nhưng không dùng YAML để đổi sang manifest khác. Với metadata bắt buộc không thể đọc/schema cấu trúc hỏng, API raise DatasetInputError và CLI 2; không giả tạo audit PASS.
- **Câu hỏi chặn triển khai:** không có. H1/H2/H3 và các thí nghiệm model ở DECISIONS vẫn thuộc bước sau; không tự quyết thay người dùng.
- Git status đã được kiểm tra trước mỗi commit. Không stage dataset, overlay, môi trường/cache, key/token. Dataset chỉ đọc; không merge hoặc đụng branch khác. Dừng bàn giao để người dùng/VERIFIER review.
