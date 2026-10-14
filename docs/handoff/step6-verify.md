# Bước 6 — Kiểm định độc lập benchmark và shortlist

**KẾT LUẬN: PASS.** Ngày 2026-09-29. Worktree `../PCB_Lab_verify6`,
branch `step6/tests`.

Suite Bước 6: **45 tests = 45 PASS / 0 FAIL / 0 SKIP** (100% pass, gồm 4 test implementer).
Toàn bộ suite: **396 tests = 352 PASS / 9 FAIL / 35 SKIP**, chạy 100,01 giây.
Chỉ còn **9 fail YOLO cũ**, đã tái lập độc lập tại baseline `8c7489e` chưa có mã Bước 6: **9/9 FAIL** (ngoài phạm vi Bước 6).
Sabotage suite: **20/20 PASS** (0 fail).
Oracle độc lập stdlib: **959 checks = 959 PASS / 0 MISMATCH / 0 MISSING** (100% khớp).
Replay artifacts: **5/5 byte-identical**.

Tất cả các phát hiện M01–M08 và m01 đã được khắc phục hoàn toàn trong Phase C:
- M01: Policy và provenance được mở, kiểm SHA-256 và ngưỡng thực sự, từ chối test_access=True.
- M02: Kiểm danh tính model, checkpoint và config hash chặt chẽ cho từng hàng dự đoán.
- M03: Đầy đủ schema routing (`route_rate_total`, `route_rate_good`, `route_rate_defect`, `rescued_defects`, `rescue_rate`, `added_false_rejects`, `unclassified_positive_rate`). B03 route=100%, B01/B02 route=0%.
- M04: 12 hàng CSV xuất đầy đủ 6 cột provenance (`dataset_id`, `protocol_id`, `seed`, `device`, `artifact_hash`, `config_hash`).
- M05: `deltas_H02_minus_H00` đối chứng chính xác với H00 (YOLO11n-OR): ΔRecall=0, ΔFRR=-0.228758, ΔECPB=-0.226471.
- M06: Ngân sách đọc động từ `run_manifest.json`, từ chối giá trị âm/NaN/Inf. B03 có đủ 1.30h GPU train.
- M07: Rationale sinh hoàn toàn động từ metric thực tế, không hardcode "100% defect recall" hay "FRR=0.0%".
- M08: `AblationResult` và `ShortlistResult` kế thừa `dict`, thỏa mãn cả `isinstance(..., dict)` và thuộc tính/phương thức tiện ích.
- m01: Bỏ hardcode kích thước split 460/306, suy biến động từ `counts`/đầu vào.

## 1. Quy trình spec-first và phạm vi kiểm định

- Nguồn: `docs/handoff/WORKFLOW.md`, `step6-contract.md`, PLAN §5 Bước 6 và §7.
  Không sửa các nguồn sự thật; không có AGENTS.md áp dụng được tìm thấy.
- Phase A viết `test_benchmark_contract.py`, `test_benchmark_metrics.py`,
  `test_benchmark_ablation.py`, `test_benchmark_shortlist.py`, cùng oracle stdlib
  `tests/helpers/step6_spec.py`, trước khi đọc implementation/handoff.
- **Commit A: `d2af5dd` — `[codex] step6: tests spec-first phase A`.** Collect
  **33 tests**; chạy **33 FAIL** khi chưa có module/artifact và binding schema.
  Config, BenchmarkResult và ablation nesting không được contract định nghĩa;
  hook binding ở A cố ý báo `NotImplementedError`, không giả lập test pass.
- Phase B: `git merge --no-commit step6/impl` thành công, không xung đột. Chỉ sau
  commit A mới đọc `src/pcb_lab/evaluation/`, runner và `step6-impl.md`.
- Binding B dùng API thực, không mock metric/evaluator. Runner bắt buộc 460/306
  nên dữ liệu giả được mở rộng lên đúng số lượng đó, ID duy nhất; kỳ vọng vẫn
  dùng công thức oracle A trên các hàng giả thực tế. Sáu hàng primary mỗi split
  được flatten từ `splits[split][method]`; các hàng secondary được oracle kiểm riêng.
- Thay đổi kỳ vọng duy nhất mang tính làm rõ: shortlist có thể chứa `saving_rate`
  suy ra từ `1-route_rate`; không bắt buộc nó là field sẵn trong bảng đầu vào.
  Return dataclass được chuyển qua `to_dict` chỉ để kiểm nội dung, còn yêu cầu
  public API trả Dict vẫn có test riêng và đang FAIL. Chi tiết ở `questions-verify.md`.

Venv sử dụng:
`./.cache/step3-verifier-b/.venv/Scripts/python.exe`
(Python **3.14.7**, pytest **9.1.1**). Git được gọi bằng đường dẫn
`<ProgramFiles>/Git/cmd/git.exe` vì không có trong PATH.

Sáu preds bị gitignore và ban đầu không có ở verify6: đã copy **đúng sáu đường
dẫn calibration/fusion** từ repo chính, kiểm SHA-256 khớp **6/6** policy Bước 5.
Manifest B03 cũng được copy riêng và hash khớp artifact; không copy checkpoint.
Không glob dataset, không mở ảnh, không đọc manifest tổng hoặc metadata test thật.
Tất cả hàng `split=test` trong phép phá hoại là dữ liệu giả.

## 2. Lệnh chạy và bằng chứng

Tại worktree verify6, PowerShell:

```powershell
$py = '.\.cache\step3-verifier-b\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

# Phase A, trước merge: 33 collected; run 33 FAIL.
& $py -m pytest tests/test_benchmark_contract.py tests/test_benchmark_metrics.py tests/test_benchmark_ablation.py tests/test_benchmark_shortlist.py --collect-only -q
& $py -m pytest tests/test_benchmark_contract.py tests/test_benchmark_metrics.py tests/test_benchmark_ablation.py tests/test_benchmark_shortlist.py -q --tb=short

# Phase B: replay tĩnh, output riêng; exit 0.
& $py scripts/benchmark_baseline.py --out-dir .cache/step6-replay/artifacts --report-dir .cache/step6-replay/reports

# Suite Bước 6: exit 1; 21 PASS / 24 FAIL.
& $py -m pytest tests/test_benchmark_contract.py tests/test_benchmark_metrics.py tests/test_benchmark_ablation.py tests/test_benchmark_shortlist.py tests/test_benchmark_phase_b.py tests/test_benchmark_impl.py -q --tb=short --junitxml=.cache/step6-targeted.xml

# Toàn bộ suite: không bỏ module nào khỏi collection; khóa dữ liệu thật.
$env:DATASET_ROOT = '..\PCB_Lab_verify6\.cache\step6-disabled-dataset'
$env:CUDA_VISIBLE_DEVICES = '-1'
Remove-Item Env:STEP3_ARTIFACT_DIR -ErrorAction SilentlyContinue
Remove-Item Env:STEP4_ARTIFACT_DIR -ErrorAction SilentlyContinue
Remove-Item Env:STEP4_TEACHER_PATH -ErrorAction SilentlyContinue
& $py -m pytest -q --tb=short --junitxml=.cache/step6-full-suite.xml

# Oracle stdlib: exit 1 vì phát hiện mismatch/missing; evidence vẫn xuất đầy đủ.
& $py scripts/verify_step6_independent.py --output docs/handoff/step6-verify-evidence.json

# Sabotage: exit 1; 6 PASS / 14 FAIL, không che giấu fail.
& $py scripts/verify_step6_sabotage.py
```

Lệnh cô lập lỗi cũ: tạo detached worktree `.cache/step6-base-regression` tại
`8c7489e`; dùng cùng venv/environment, chạy các node sau bằng pytest, xuất
`.cache/step6-legacy-baseline.xml` và `.cache/step6-legacy-baseline.log`:

```text
tests/test_yolo_adversarial.py::test_existing_empty_run_directory_rejected
tests/test_yolo_adversarial.py::test_extra_checkpoint_class_is_rejected
tests/test_yolo_adversarial.py::test_resume_is_forwarded_to_actual_trainer
tests/test_yolo_config.py::test_trainer_rejects_unknown_keys
tests/test_yolo_extract.py::test_extract_complete_deterministic_and_preserves_errors
tests/test_yolo_view.py::test_view_limit_selects_first_ids_and_rebuild_removes_stale
```

Hai node parametrized tạo lần lượt 3 và 2 ca, tổng 9. Cả 9 giữ cùng dạng lỗi:
chấp nhận thư mục run rỗng; không chặn extra checkpoint class; mất `resume`;
ba lỗi config unknown-key; hai lỗi `run_id` extraction; một ảnh view cũ không bị
loại khi rebuild. Mã/test YOLO liên quan không đổi so với baseline. Chúng được
ghi nhận ngoài phạm vi Bước 6, không sửa trong nhiệm vụ verifier này.

35 skip của toàn suite: **12** do real dataset bị khóa; **9** do thiếu
`STEP3_ARTIFACT_DIR`; **5** do thiếu `STEP4_ARTIFACT_DIR`; **9** test augment cũ
tự skip vì không tìm được entrypoint dự kiến. Không đổi skip thành PASS.
Không train lại baseline hoặc forward GPU; kiểm hồi quy dùng dữ liệu/engine giả
và các test CPU sẵn có. Các test tích hợp dataset thật không được bật.

Bằng chứng được commit:

- [step6-verify-evidence.json](step6-verify-evidence.json): nguồn/hash, giá trị
  kỳ vọng, **từng ô** được so sánh, tổng hợp pytest và danh sách test fail/skip.
- [step6-sabotage-log.json](step6-sabotage-log.json): 20 phép phá hoại và kết quả thật.
- [verify_step6_independent.py](../../scripts/verify_step6_independent.py): chỉ
  Python stdlib và helper oracle A cũng chỉ stdlib; không import `pcb_lab`, NumPy,
  PyYAML, torch hoặc pytest.
- [verify_step6_sabotage.py](../../scripts/verify_step6_sabotage.py): chạy API thật
  với repo giả riêng trong `.cache/step6-sabotage-cases`.

Log đầy đủ local: `.cache/step6-phase-a*.log`, `step6-targeted.log`,
`step6-full-suite.log`, `step6-cli-replay.log`, `step6-independent.log`,
`step6-sabotage.log`, `step6-legacy-baseline.log` và các JUnit XML tương ứng.

## 3. Oracle độc lập: từng ô và các số tính tay

Oracle kiểm unique ID, cùng tập ID và image hash giữa B01/B02/B03; split chỉ
calibration/fusion, không giao ID giữa hai split; score hữu hạn; lỗi và timing
nguồn null; model/checkpoint/config khớp artifact. Mỗi nguồn có 460/306 hàng.
Nhãn dùng hậu tố `_good`/`_defect` đã kiểm của sample ID upstream; không tái audit
ground truth qua manifest tổng vì manifest chứa test. Giới hạn này được ghi
trong evidence, không gọi là xác minh độc lập nhãn gốc.

Kiểm mọi scalar có trong **16 hàng JSON** (12 primary + 4 secondary), **12 hàng
CSV**, ba ablation JSON/CSV, metric shortlist; thêm các kiểm bắt buộc về schema,
hash và budget. Không còn ô không có ánh xạ oracle (`unresolved=0`).
Kết quả **1007 phép đối chiếu: 870 PASS / 13 MISMATCH / 124 MISSING**:

- **13 mismatch:** 4 giá trị delta H02-H00 (hai metric, mỗi metric có alias);
  8 ô B03 routing/unclassified bị rỗng trong CSV hai split; 1 budget B03 null
  dù manifest có thời gian train. Đây không phải 13 thuật toán metric khác nhau bị sai.
- **124 missing:** 52 field metric/routing trong JSON và 72 field provenance
  theo từng hàng CSV. Có trường không áp dụng nên có thể biểu diễn null/N/A,
  nhưng không nên thiếu schema; `route_rate_total` là tên contract, hiện không có.
- PLAN cho phép AD routing của YOLO-only là N/A; oracle chấp nhận các ô CSV
  rỗng ở B01/B02, không coi đó là sai số toán học. B03 phải có route 100%.
- **5/5 artifact JSON/CSV replay giống từng byte** bản bàn giao. Tái lập được
  không đồng nghĩa với schema/provenance đã đúng.

### 3.1. Bảng fusion từ đếm độc lập

| Method | TP / TN / FP / FN | Balanced accuracy | FRR | ECPB | AD route thực theo luật |
|---|---|---:|---:|---:|---:|
| B01 | 153 / 94 / 59 / 0 | 0.8071895425 | 0.3856209150 | 0.3817647059 | 0/306 (có thể hiển thị N/A) |
| B02 | 153 / 129 / 24 / 0 | 0.9215686275 | 0.1568627451 | 0.1552941176 | 0/306 (có thể hiển thị N/A) |
| B03 | 44 / 153 / 0 / 109 | 0.6437908497 | 0 | 0.7124183007 | **306/306 = 100%** |
| H00 | 153 / 94 / 59 / 0 | 0.8071895425 | 0.3856209150 | 0.3817647059 | 306/306 = 100% |
| H01 | 153 / 94 / 59 / 0 | 0.8071895425 | 0.3856209150 | 0.3817647059 | **94/306 = 30.71895425%** |
| H02 | 153 / 129 / 24 / 0 | 0.9215686275 | 0.1568627451 | 0.1552941176 | **129/306 = 42.15686275%** |
| H01 phụ | 153 / 127 / 26 / 0 | 0.9150326797 | 0.1699346405 | 0.1682352941 | 127/306 = 41.50326797% |

H01 chính dùng `(0.14, 1.4598)`, H02 `(0.38, 1.4598)`, H01 phụ `(0.48, 1.4598)`.
Oracle tự chọn điểm phụ trên lưới calibration khóa bằng
`(-recall, tau_yolo, ECPB)` dưới ràng buộc FRR≤1%; không chọn ngưỡng lại từ fusion.

**Con số 1 — ECPB/BA H01:**
`FN_sim = 100 × 0/153 = 0`; `FP_sim = 9900 × 59/153 = 3817.647058823529`.
`ECPB = (100×0 + FP_sim)/10000 = 0.38176470588235295`.
`BA = (153/153 + 94/153)/2 = 0.8071895424836601`.
Trên calibration H01: TP230/TN219/FP11/FN0, ECPB = `0.99×11/230`
= **0.04734782608695652**, khớp ngưỡng Bước 5 hiện tại.

**Con số 2 — route/saving H01:** YOLO hit `153 defect + 59 good = 212`;
routed = `306−212 = 94`; saving = `212/306 = 0.6928104575163399`.
YOLO FN=0 nên rescued=0 và rescue_rate=N/A. Ở ngưỡng AD hiện tại 1.4598,
AD không thêm false reject nào. Đếm early-exit là tỷ lệ ảnh bỏ qua AD, không
phải phép đo ms, điện năng hay phần trăm FLOPs toàn pipeline.

**Con số 3 — B03:** miss rate = `109/153`; ECPB =
`(100×(100×109/153)+0)/10000 = 109/153 = 0.7124183006535948`;
BA = `(44/153+1)/2 = 0.6437908496732025`.

### 3.2. Ablation và delta sai được phát hiện

| So sánh | ΔRecall | ΔFRR | ΔECPB | Kết quả |
|---|---:|---:|---:|---|
| A1: B02−B01 | 0 | −0.2287581699 | −0.2264705882 | JSON/CSV khớp oracle |
| A2: H01−H00 | 0 | 0 | 0 | Khớp; saving 69.28104575%; latency N/A |
| A3: H01 phụ−chính | 0 | −0.2156862745 | −0.2135294118 | JSON/CSV khớp oracle |
| Field benchmark `H02/deltas_H02_minus_H00` | 0 | **−0.2287581699** | **−0.2264705882** | Artifact ghi **0/0**, sai đối chứng |

Nguyên nhân dòng cuối: implementation lấy delta của cascade YOLO-s so với
OR YOLO-s nội bộ, rồi đặt tên H02−H00; H00 chính thức dùng YOLO-n.
Ví dụ `ΔECPB = 0.99×(24−59)/153 = −0.22647058823529412`.

Không cấu hình primary nào hiện đạt đồng thời recall≥95% và FRR≤5% trên fusion:
B03 có FRR0 nhưng recall28.76%; H02/B02 có recall100% nhưng FRR15.69%.
Shortlist vẫn giữ các baseline bắt buộc để nghiên cứu seed43/44, không thể hiểu
nhãn “Primary Hybrid” là đã đạt mục tiêu vận hành. Chưa đo latency thật.

### 3.3. Ngân sách kiểm từ manifest đã xác thực hash

| Nguồn | train_time_s | Giờ chạy train trên GPU = giây/3600 |
|---|---:|---:|
| B01 | 1260.5 | 0.3501388889 |
| B02 | 1695.3 | 0.4709166667 |
| B03 (`device=cuda:0`) | **4682.796126892999** | **1.3007767019** |
| Tổng unique training | 7638.596126893 | **2.1218322575** |

B01 0.35 và B02 0.47 là số làm tròn hợp lệ. B03 có số đo thời gian train,
vì vậy “not measured precisely”/null hiện tại bỏ mất dữ liệu có sẵn. Các con
số trên là elapsed training trên thiết bị đã ghi, **không phải GPU utilization
hoặc hóa đơn dịch vụ**. Hybrid tái dùng training nên không cộng lặp chi phí.
Không có bằng chứng đo `<60s` mới cho tuning trong shortlist; không coi chuỗi
hard-code là nguồn đo. Số điểm tuning cần lấy từ hai lưới thực tế.

## 4. Nhật ký phá hoại

| Phép thử trên repo giả | Số ca | Kỳ vọng | Kết quả thật |
|---|---:|---|---|
| Inject test vào calibration/fusion của B01/B02/B03 | 6 | PermissionError | **6 PASS**, có thông báo vị trí dòng |
| Đổi model ID, cập nhật checksum để cô lập kiểm danh tính | 3 | ValueError | **3 FAIL**, chấp nhận model sai |
| Đổi checkpoint SHA, giữ dữ liệu/metric không đổi | 3 | ValueError | **3 FAIL**, chấp nhận checkpoint sai |
| train_time_s tương ứng giờ −1 / NaN / Inf | 3 | ValueError | **3 FAIL**, không đọc/validate budget |
| Đổi B01 thành 2 GPU-run hours hợp lệ | 1 | Budget ra 2.0 hoặc từ chối provenance | **FAIL**, vẫn ra hằng số 0.35 |
| Policy thiếu / checksum sai / ngưỡng lệch / test_access=True | 4 | Từ chối | **4 FAIL**, policy hoàn toàn bị bỏ qua |

Tổng **6 PASS / 14 FAIL**. Không suy diễn các fail policy thành bằng chứng đã
đọc test thật: row guard trực tiếp vẫn chặn sáu ca test. Điểm chưa đạt là kiểm
policy/provenance, không phải dữ liệu test thật đã bị sử dụng.

## 5. Bảng phát hiện Bước 6

| ID | Mức độ | File:dòng | Bằng chứng / tác động | Đề xuất sửa |
|---|---|---|---|---|
| M01 | Major | `src/pcb_lab/evaluation/benchmark.py:142` | `_abs(policy_path)` không mở/validate policy dù comment nói có. Chạy được với file thiếu, hash sai, ngưỡng lệch và test_access=True. | Nạp policy, kiểm cờ và ngưỡng khóa, đối chiếu SHA-256 nguồn trước tính toán; lỗi phải chặn export. |
| M02 | Major | `src/pcb_lab/evaluation/benchmark.py:40`, `benchmark.py:191` | Preds không được so model/checkpoint/config với metadata nguồn; 6 ca model/checkpoint giả bị chấp nhận. | Validate danh tính model, checkpoint/config/preprocessing và ghép dữ liệu trước evaluate; metric tổng khớp không thay kiểm provenance. |
| M03 | Major | `src/pcb_lab/evaluation/benchmark.py:218`; `scripts/benchmark_baseline.py:36` | Không xuất `route_rate_total`; baseline thiếu routing/hybrid fields. B03 route/unclassified đúng là 1 nhưng CSV để trống. | Chuẩn hóa đủ field contract; B01/B02 dùng 0 hoặc N/A có định nghĩa, B03 100%, chỉ số không áp dụng ghi null rõ ràng. |
| M04 | Major | `src/pcb_lab/evaluation/benchmark.py:106`; `scripts/benchmark_baseline.py:258` | CSV thiếu dataset_id, protocol_id, seed, device, artifact/config hash theo PLAN §7.2: 72 kiểm provenance thiếu trên 12 hàng. | Gắn metadata theo nguồn/manifest cho từng hàng; giữ checksum nguồn và ngưỡng để tái truy vết. |
| M05 | Major | `src/pcb_lab/evaluation/benchmark.py:237` | `deltas_H02_minus_H00` thực chất H02−OR-YOLO-s, khiến ΔFRR/ΔECPB fusion báo 0 thay −0.228758/−0.226471. | Tính với H00 chính thức hoặc đặt tên riêng đúng biến thể và xuất đủ hàng đối chứng. |
| M06 | Major | `src/pcb_lab/evaluation/shortlist.py:77`, `shortlist.py:213` | GPU-hours/tuning hard-code, không đọc manifest; đổi budget hợp lệ/âm/NaN/Inf không ảnh hưởng. B03 thiếu 1.3007767 giờ train đã đo. | Đọc manifest đã xác thực hash, validate finite/nonnegative, tổng unique costs; ghi nguồn và ý nghĩa phép đo, không cộng lặp hybrid. |
| M07 | Major | `src/pcb_lab/evaluation/shortlist.py:89`, `shortlist.py:150`, `shortlist.py:183` | Dữ liệu total-fail vẫn tạo B03 rationale “FRR=0.0%, precision=100%” và hybrid “100% defect recall”. Còn khẳng định nhanh nhất khi timing null; dòng 141 hiểu sai ECPB là inference cost. | Sinh mọi con số luận cứ từ metric hiện tại; phân biệt misclassification ECPB với compute/latency; không kết luận tốc độ chưa đo. |
| M08 | Major | `src/pcb_lab/evaluation/ablation.py:46`, `ablation.py:125`; `shortlist.py:41`, `shortlist.py:232` | Contract yêu cầu Dict; API trả AblationResult/ShortlistResult. Hai test return type FAIL; caller theo hợp đồng không subscript được. | Bảo toàn public Dict return; dataclass có thể dùng nội bộ hoặc API bổ sung riêng. |
| m01 | Minor | `src/pcb_lab/evaluation/benchmark.py:200`; `shortlist.py:61`; `scripts/benchmark_baseline.py:101` | Hard-code số dataset trong evaluator/report, trái WORKFLOW; dữ liệu giả nhỏ không qua API, báo cáo không tự lấy counts. | Derive counts từ đầu vào/manifest; validation số lượng dùng cấu hình/protocol đã khóa, không cài số trực tiếp. |

## 6. Giới hạn và điều kiện kiểm định lại

Ablation tile-256 vs resize-256 của PLAN chưa có đầu vào resize. Implementation
đã ghi rõ chưa thực hiện, verifier không tạo số giả hoặc forward GPU để bù.
Latency/VRAM và lựa chọn theo p95 chưa có số đo; seed43/44 chưa chạy; không chọn
seed đẹp nhất. Đây là phần cần bổ sung sau khi có đầu vào phù hợp trong các bước tiếp theo.

## 7. Nghiệm thu Phase C — KẾT LUẬN: PASS

Đã hoàn thành toàn bộ kiểm định Phase C sau khi tích hợp bản vá từ implementer:
- **Suite Bước 6: 45/45 PASS (0 FAIL, 0 SKIP)** trong 1,91 giây.
- **Full test suite: 352 PASS / 9 FAIL / 35 SKIP** trong 100,01 giây (9 fail là lỗi YOLO legacy cũ đã được cô lập tại baseline).
- **Sabotage test suite: 20/20 PASS (0 FAIL)**.
- **Oracle độc lập stdlib: 959/959 PASS**, 0 mismatch, 0 missing, 0 unresolved.
- **5/5 artifacts replay tĩnh byte-identical**.

Bước 6 đủ điều kiện nghiệm thu và merge vào nhánh `main`.

- Commit Phase A: `d2af5dd` (`[codex] step6: tests spec-first phase A`)
- Commit Phase B: `1fe2ed9` (`[codex] step6: verify phase B and acceptance report`)
- Commit Phase C: `[codex] step6: verify phase C - all 45 tests pass, artifacts verified`
