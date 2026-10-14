# Step 4 — EfficientAD-S (B03), kiểm định Giai đoạn A

Ngày: 2026-09-27. Verifier: Codex. Workspace `../PCB_Lab_verify4`,
nhánh `step4/tests`, nền `e1f37e1`; điều phối cập nhật contract `a385e16`
trong khi viết test. Đã đọc bổ sung và áp dụng Engine/normalization schema này.

## Phạm vi và tính độc lập

Đọc `WORKFLOW.md`, `step4-contract.md`, `PLAN.md` (§4.4, §5 Bước 4, §7),
`DECISIONS.md`, `TRIAGE.md`; thêm contract Step 2 và test/helper verifier đã nghiệm
thu để dùng đúng API loader/TileManager. Không đọc source EfficientAD,
`step4/impl` hoặc `docs/handoff/step4-impl.md`. Không sửa production/config/spec,
không huấn luyện, không tải teacher, không mở ảnh test thật.

Import implementation nằm trong test/fixture/callable chạy lúc thực thi. Không dùng
`importorskip` để giấu module EfficientAD chưa có. Chỉ các phép nghiệm thu artifact
thật mới skip khi chưa đặt `STEP4_ARTIFACT_DIR`; đường dẫn được cung cấp mà hỏng sẽ
fail. Mạng bị chặn trong các test mới để tránh download ngoài ý muốn.

## Các bộ test và tiêu chí đối soát

| File | Nội dung và oracle độc lập |
| --- | --- |
| `tests/test_efficientad_contract.py` | Config khóa và khóa lạ ở root/train/normalization/infer; ConfigError khi null chưa duyệt; từ chối thay recipe; test split bị PermissionError ngay cả root chưa tồn tại; tiêm defect vào AdTrainSet, yêu cầu DataLeakError/ValueError trước decode; train good-only, eval đủ good/defect; cấu trúc artifact/run/report, SHA-256 thật, schema manifest/artifact/calibration. |
| `tests/test_efficientad_tiling.py` | 9 tensor tiles native từ dataset, tọa độ 0/224/448; pixel không bị resize; hằng số và affine field phải tái tạo với sai số tuyệt đối ≤1e-6; overlap 32px tăng tuyến tính, đơn điệu ở cả hai trục, không nhảy biên; padding không lọt vào score; anomaly_score đúng float(max(map)), không mean. |
| `tests/test_efficientad_heatmap_bbox.py` | R8 qua predict thật với FakeEngine theo a385e16: threshold, mở 3×3 loại vệt mảnh và tách cầu 1px; area 18/20 bị loại, 21 được giữ; hai hình vuông chồng nhau có area 17 nhưng bbox area 25 phải bị loại; peak thuộc component, không lấy peak component khác trong bbox; nested rectangles có IoU .36/.49 để bắt thiếu NMS; Step 2 NMS giữ IoU=.45, loại khi lớn hơn; class anomaly_unclassified/class_id=None, tọa độ global và clip biên/seam. |
| `tests/test_efficientad_reload.py` | ArtifactMismatchError/SmokeArtifactError rõ ràng trước deserialize; describe đối chiếu metadata; deserialize toàn payload → torch.save → from_artifact → predict 5 ảnh calibration gồm good và defect, score/map sai số tuyệt đối ≤1e-5, rtol=0. |
| `tests/test_efficientad_extract.py` | Chặn toàn bộ partitions sai trước load, kể cả request trộn calibration/test; dataset giả đúng 460/306 dòng, hai lần chạy byte-identical; không lặp/drop; lỗi decode/inference giữ dòng, error_reason và score null; good hợp lệ score 0 phân biệt với lỗi; hashes/metadata/timing; snapshot dataset không đổi; nghiệm thu predictions thật đối chiếu manifest và không giao test qua ID/hash/group. |
| `tests/helpers/step4_spec.py`, `step4_fixtures.py` | Schema/assertions và PNG giả do verifier sở hữu; không tái sử dụng thuật toán implementation làm oracle. Plugin đăng ký trong `tests/conftest.py`. |
| `tests/test_yolo_metadata_refresh.py` | Chuyển import runner ở đầu module vào fixture `runner`; giữ nguyên sáu test/assertion YOLO. Sửa lỗi collection đã quan sát do eager import kéo theo torch.dll; không đổi production. |

Số 895/460/306 và 230/230 chỉ nằm trong test nghiệm thu từ spec; không thêm hằng số
vào production. Dataset giả có test metadata nhưng **không tạo file ảnh test**.
Extraction dùng fake detector qua API công khai để cô lập logic ghi dòng; việc
nạp checkpoint/hash được kiểm riêng, không xem fake checkpoint là model thật.

## Giới hạn được ghi trước khi xem implementation

- Engine protocol và normalization schema đã chốt tại `a385e16`: các test số học
  của adapter/R8 dùng FakeEngine độc lập, không skip vì thiếu artifact. Kiểm thêm
  engine nhận đúng RGB uint8, không resize/đổi kênh/chuẩn hóa hai lần. Các test
  TileManager độc lập chạy ngay trên nền Step 2. Artifact/reload thật vẫn opt-in.
- Chưa xác minh runtime nguồn teacher statistics/map quantiles hay fit→save lần
  đầu. Đã ghi phần bị chặn và giao diện cần bổ sung trong `questions-verify.md`.
  Config split đúng và normal_only_verified trong manifest không đủ chứng minh
  nguồn thống kê đúng; không dùng chúng để tuyên bố nghiệm thu B03.
- Chỉ R8 candidate boxes trong Bước 4; không chọn tau_ad, không benchmark
  segmentation/mAP sáu lớp cho AD, không cascade. R2 cần đủ defect calibration
  trong extraction, không lẫn với calibration good-only dùng normalize map.
- CPU theo S5; không yêu cầu CUDA hay thay recipe để khớp kết quả.

## Lệnh và kết quả thực chạy

Môi trường dùng đúng venv hiện có:

```powershell
Set-Location ../PCB_Lab_verify4
$py = './.cache/step3-verifier-b/.venv/Scripts/python.exe'
$files = @(
  'tests/test_efficientad_contract.py',
  'tests/test_efficientad_tiling.py',
  'tests/test_efficientad_heatmap_bbox.py',
  'tests/test_efficientad_reload.py',
  'tests/test_efficientad_extract.py'
)
& $py -B -m pytest @files --collect-only -q -p no:cacheprovider
& $py -B -m pytest --collect-only -q -p no:cacheprovider
& $py -B -m pytest @files -q --tb=short -p no:cacheprovider
```

- Metadata môi trường: Python **3.14.7**, torch **2.14.0**, pytest **9.1.1**,
  NumPy **2.5.3**, PyYAML **6.0.3**. Import NumPy/pytest thành công.
- Collection riêng Step 4: **75 tests collected, exit 0**, 0.08s.
- Collection toàn repo lần đầu: **225 collected, 1 collection error, exit 2**,
  do `test_yolo_metadata_refresh.py` import runner kéo theo torch.dll. Sau sửa lazy
  import: **231 tests collected, exit 0**, 0.24s; không bỏ qua module/test nào.
- Chạy toàn bộ 75 test Step 4: **11 passed, 59 failed, 5 skipped**, 1.09s,
  exit 1. **58 FAIL là ModuleNotFoundError cho EfficientAD**, **1 FAIL là config
  efficientad_s.yaml chưa tồn tại** trên nhánh chỉ có spec. Đây là kết quả đỏ
  nguyên trạng ở A, không phải đánh giá code Agent 1.
- **5 SKIP**: artifact layout/hash, calibration schema, describe thật, checkpoint
  save/reload thật, predictions thật; chưa đặt `STEP4_ARTIFACT_DIR`. Các test số học
  dùng FakeEngine **không skip**, fail rõ vì implementation chưa merge.
- Kiểm fixture độc lập đã chạy: train 4 (2 good/2 defect), calibration 460
  (230/230), fusion 306 (153/153); loader Step 2 nhận đúng số mẫu, AdTrainSet chỉ
  có 2 good, decode mẫu development hợp lệ. Không tạo/mở file test. PASS.
- `git diff --check`: PASS. Các import EfficientAD đều nằm trong test/callable;
  không thêm import torch lúc collection.

| Bộ test | Tổng | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: | ---: |
| contract | 29 | 0 | 27 | 2 |
| tiling | 12 | 8 | 4 | 0 |
| heatmap_bbox | 19 | 3 | 16 | 0 |
| reload | 5 | 0 | 3 | 2 |
| extract | 10 | 0 | 9 | 1 |
| **Tổng** | **75** | **11** | **59** | **5** |

Lần chạy có thêm `--basetemp` và `--junitxml` trỏ vào cache verifier để giữ bằng
chứng ngoài git. Log tại
`./.cache/step4-phase-a/`:
`phase-a.log`, `phase-a.xml`, `collect-all.log` (lần đầu),
`collect-all-final.log`. Script kiểm fixture một lần là `check_fixture.py` trong
cùng cache; không phải implementation EfficientAD hay test acceptance giả.

Lần đầu ghi log từ sandbox bị AccessDenied nên chưa chạy pytest; đã chạy lại ngoài
sandbox. Lỗi nạp torch DLL vẫn tái hiện cả hai môi trường và được báo nguyên trạng.

## Kết luận và bảng phát hiện

**Giai đoạn A: PASS-WITH-CONDITIONS** — đã viết và thực chạy suite spec-first,
collection toàn repo PASS; test chưa có implementation giữ đỏ đúng yêu cầu.
**Không kết luận B03 PASS**, chưa bước sang B, chưa đọc implementation.

| Mức độ | Vị trí / bằng chứng | Trạng thái và bước tiếp theo |
| --- | --- | --- |
| Major — dự kiến ở A | `tests/test_efficientad_contract.py:17`; phase-a.xml: 1 thiếu config + 58 thiếu module | Giữ nguyên expectation. Sau commit A, điều phối merge implementation rồi chạy B. |
| Major — môi trường | venv `Lib/site-packages/torch/__init__.py:292`; WinError 4551, torch.dll bị Application Control chặn | Chưa thể xác minh runtime PyTorch. Cần môi trường CPU được Windows cho phép nạp DLL; không đổi hay tắt chính sách từ verifier. |
| Major — còn mở ở B | `questions-verify.md`, mục Step 4 số 2; `tests/test_efficientad_reload.py` | Cần chứng minh sample nguồn teacher/quantile không chứa defect và model đang fit nhất quán trước lần save đầu. Test roundtrip hiện kiểm payload artifact đầy đủ, không thay thế phép kiểm fit→save. |
| Minor — đã sửa | `tests/test_yolo_metadata_refresh.py:11` | Import runner trì hoãn bằng fixture. Collection tăng từ 225 + error lên 231 thành công; assertion cũ không đổi. |

Ở Giai đoạn B phải có nhật ký phá hoại và ít nhất hai phép đối soát số học độc lập
theo WORKFLOW. Chưa có huấn luyện, checkpoint hay số liệu chất lượng B03 trong A.

## Giai đoạn B — thẩm định độc lập và đối soát số học (2026-09-27)

**Kết luận: PASS-WITH-CONDITIONS.** Kiểm định implementation trên CPU và smoke
pretrained độc lập đã PASS; chưa nghiệm thu baseline B03 70 epoch. Năm phép kiểm
artifact chính thức vẫn skip vì chưa cung cấp `STEP4_ARTIFACT_DIR`; chưa có bằng
chứng predictions 460/306 từ checkpoint baseline thật. Không phát hiện lỗi
Critical/Major của production trong phạm vi đã thực thi. CUDA chưa chạy thực tế.

### Nền mã và thay đổi của verifier

- Worktree `../PCB_Lab_verify4`, branch `step4/tests`, bắt đầu sạch tại
  `1361c1e`; đã có implementation `fa71409` và commit test A `f13dbd5`.
- Sau A mới đọc source EfficientAD và `step4-impl.md`. Đối chiếu WORKFLOW, contract,
  PLAN §4.4/§5 Bước 4/§7, DECISIONS/TRIAGE. Không sửa production, config hoặc spec.
- Lần chạy đầu nguyên trạng: **67 passed, 3 failed, 5 skipped**, exit 1, 24.55s.
  Ba failure ở extractor đều do FakeDetector thiếu `meta`, trước khi xuất dòng.
  Concrete detector có thuộc tính này; contract chỉ chốt giao thức tiêm **Engine**,
  không chốt một giao thức FakeDetector cho extractor.
- Sửa đúng binding trong `tests/test_efficientad_extract.py:48`: thêm `self.meta`
  từ fixture artifact, lấy `run_id` từ fixture run manifest. Giữ nguyên toàn bộ
  assertion 460/306, thứ tự, hash, lỗi, score, JSONL byte-identical và dataset
  read-only; không sửa implementation hay coi checkpoint stub là model thật.
- Thêm `tests/test_efficientad_phase_b.py`: 12 ca tiêm contamination, hai oracle
  số học nhỏ và một integration test smoke pretrained thật. Smoke này chỉ chạy
  khi có `STEP4_TEACHER_PATH`; đường dẫn đã đặt mà sai phải fail. Không download.

### Lệnh và kết quả thực chạy

Venv do điều phối chỉ định: Python **3.12.10**, torch **2.6.0+cpu**, torchvision
**0.21.0+cpu**, NumPy **2.2.6**, pytest **8.3.5**, OpenCV **4.11.0**, PyYAML
**6.0.3**, psutil **7.0.0**. Import thực tế thành công. Sandbox báo `No Python`
với Python nền WindowsApps trước khi chạy pytest; chạy lại ngoài sandbox thành
công. Không gặp WinError 4551 trong venv này, không thay chính sách Windows.

```powershell
Set-Location ../PCB_Lab_verify4
$py = './scratch/step4-env/Scripts/python.exe'
$evidence = './scratch/step4-phase-b'
$files = @(
  'tests/test_efficientad_contract.py',
  'tests/test_efficientad_tiling.py',
  'tests/test_efficientad_heatmap_bbox.py',
  'tests/test_efficientad_reload.py',
  'tests/test_efficientad_extract.py'
)
# Trước khi sửa binding của FakeDetector:
& $py -B -m pytest @files -q -p no:cacheprovider --basetemp "$evidence/pytest-initial" --junitxml "$evidence/initial.xml"
# Sau khi bổ sung test B:
$env:STEP4_TEACHER_PATH = '../PCB_Lab_impl4/artifacts/pretrained/efficientad_teacher_s.pt'
& $py -B -m pytest @files tests/test_efficientad_phase_b.py -q -s -p no:cacheprovider --basetemp "$evidence/pytest-phase-b" --junitxml "$evidence/phase-b.xml"
& $py -B -m pytest tests/test_efficientad_impl.py tests/test_tile_manager.py tests/test_preprocessing.py -q -p no:cacheprovider --basetemp "$evidence/pytest-regression" --junitxml "$evidence/regression.xml"
& $py -B -m pytest --collect-only -q -p no:cacheprovider
```

| Bộ test cuối | Tổng | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: | ---: |
| contract | 29 | 27 | 0 | 2 |
| tiling | 12 | 12 | 0 | 0 |
| heatmap_bbox | 19 | 19 | 0 | 0 |
| reload | 5 | 3 | 0 | 2 |
| extract | 10 | 9 | 0 | 1 |
| phase_b | 15 | 15 | 0 | 0 |
| **Verifier tổng** | **90** | **85** | **0** | **5** |

- Verifier: exit 0, **268.26s**. 75 test theo yêu cầu có **70 PASS / 5 SKIP**.
- Implementer + TileManager/preprocessing: **75 passed**, exit 0, **20.22s**.
- Collection toàn repo: **274 tests collected**, exit 0, **0.55s**. Đây là
  collection, không tuyên bố đã chạy toàn bộ 274 test (bao gồm YOLO/Step 1–3).
- Năm skip giữ nguyên: layout/hash artifact, calibration schema, describe artifact,
  checkpoint roundtrip baseline, predictions baseline thật. Smoke không thay thế
  các phép nghiệm thu này, không đổi cờ smoke thành baseline để làm test xanh.

### Nhật ký phá hoại

| Phép phá hoại / dữ liệu biên | Bằng chứng thực chạy | Kết quả |
| --- | --- | --- |
| Chèn defect, chèn boxes vào good, đổi partition thành test ở teacher/map statistics | `test_efficientad_phase_b.py:29`; 2 hàm × 3 lỗi × 2 thời điểm | **12 PASS**: DataLeakError trước iteration khi metadata đã sai, hoặc trước tile/model execution khi iterator đưa mẫu sai. Không mở ảnh test. |
| Chèn defect vào AdTrainSet; yêu cầu test/partition lạ, cả request trộn calibration/test | Suite contract/extract | **PASS**: DataLeakError/ValueError hoặc PermissionError trước decode/load. |
| Config khóa lạ/null, tile/stride/batch/split/score/stitch sai | Suite contract | **PASS**: ConfigError, không huấn luyện bằng recipe sai. |
| Checkpoint checksum giả; artifact smoke không opt-in | Suite reload | **PASS**: ArtifactMismatchError/SmokeArtifactError; payload giả không được deserialize thành công. |
| Lỗi decode và lỗi inference tại calibration_0001 | Suite extract, mỗi trường hợp lặp hai lần | **PASS**: vẫn 460/306 dòng, đúng một dòng lỗi score=null; good score=0 hợp lệ được giữ riêng. |
| Overlap 32, affine field, padding có giá trị 10.000, đổi max thành mean sẽ sai | Suite tiling | **PASS**: giữ ảnh native, blending liên tục, crop padding, score=max(map). |
| Vệt mảnh/cầu 1px; area 18/20/21; pixel area khác bbox area; peak lân cận; NMS IoU .36/.49 và biên .45 | Suite heatmap_bbox | **PASS**: morphology, area strict, component peak, global NMS, clip đúng. |
| Hai lần extraction với cùng input; kiểm snapshot dataset giả | Suite extract | **PASS**: JSONL byte-identical, không mất/trùng dòng, hash và nội dung dataset không đổi. |

Đây là fault injection và kiểm biên qua test; không sửa/xóa dữ liệu thật hoặc
checkpoint nguồn. Network bị chặn trong toàn bộ test EfficientAD qua fixture A.

### Đối soát độc lập (không dùng hàm thống kê production làm oracle)

| Đại lượng | Đường độc lập | Kết quả |
| --- | --- | --- |
| Train/calibration/fusion populations | Parse trực tiếp JSONL manifest, đếm good/defect; chỉ metadata của test | train **895 good / 897 defect**, calibration **230/230**, fusion **153/153**; 895 × 9 = **8.055** tile train good. |
| Teacher mean/std, ví dụ nhỏ | Dãy 1..8 và 10..80; mean và population variance tính tay | mean **4.5 / 45.0**, std **sqrt(5.25) / sqrt(525)**; 8 feature pixels/kênh, PASS. |
| Map quantiles, ví dụ nhỏ | Sắp thứ tự rồi nội suy rank `(n-1)*q`, không gọi np.quantile/torch.quantile | hai nhánh q90/q99.5 = **[6.3, 6.965] / [73.0, 79.65]**, PASS. |
| Teacher mean/std, smoke thật | Forward hook thu quan sát kênh 0/191/383; ghép vector float64, variance quanh mean; đối chiếu với thống kê streaming của runner | kênh 0 mean **0.4066927818773325**, std **1.3159968950594205**; max abs error 3 kênh: mean **2.751753891416797e-8**, std **1.0283473850236646e-8**. |
| Map quantiles, smoke thật | Thu đủ raw maps hai nhánh rồi dùng sorted-rank oracle trên **112.896** pixels/nhánh, không subsample | combined q90 **4.611445057904348**, q99.5 **6.611666662944481**; mean **3.226111521019749**, std **1.1468684676717853**; khớp atol=1e-12, rtol=0. |
| Fit → FIRST save → from_artifact → predict | Hook ngay trước `_save(model.pt)`, dùng chính model đang fit, xác nhận file chưa tồn tại; sau đó gọi from_artifact thật | **5/5 ảnh**, cả good và defect; max map/score error **0.0 / 0.0**, ngưỡng ≤1e-5. |

Branch quantiles smoke được tính lại độc lập:
`[[9.208272933959961, 13.209234404563905],
[0.015685973688960075, 0.01990358913317322]]`.
Các số map statistics khớp số artifact smoke Agent 1, dù verifier train lại từ
đầu. Mean loss **77.07076431** cũng trùng; đây là smoke, không là metric chất lượng.

Đã ghi sample ID thực sự decode trong từng stage và đối chiếu manifest: teacher
và optimizer dùng đúng bốn train good `00041000..00041003`; map statistics dùng
bốn calibration good `12000001/12000017/12000038/12000052`. Không có defect hoặc
test trong ba stage này. Báo cáo calibration chỉ đọc defect sau khi stats đã khóa.
Hooks chỉ quan sát, vẫn gọi hàm thống kê, loss, optimizer và serializer production.
Khép lại giới hạn A về runtime nguồn thống kê và so sánh trước lần save đầu.

Smoke verifier dùng CPU, teacher pretrained có sẵn, **1 epoch / 36 optimizer step**,
không download và không tạo output chính thức trong artifacts/. SHA256 checkpoint
verifier: `bcee10df62ba39609c74493930873310dbf043f3f7ff100cd6c05991ae3d014b`.
Đã tính lại SHA256 checkpoint và run_manifest, đối chiếu artifact metadata.
Hash khác Agent 1 không phải lỗi số học: config dùng đường dẫn teacher tuyệt đối
và provenance thuộc verifier; không sửa provenance để giả cùng run.

### Bảng phát hiện và điều kiện còn lại

| Mức độ | File:dòng / bằng chứng | Kết luận / đề xuất |
| --- | --- | --- |
| Minor — verifier, đã sửa | `tests/test_efficientad_extract.py:48`; initial.xml có 3 AttributeError tại `extract.py:18` | Bổ sung binding meta/run_id của FakeDetector; assertion giữ nguyên, cả 3 ca PASS sau sửa. Không phải lỗi production đã chứng minh. |
| Major — nghiệm thu B03 còn thiếu | `tests/helpers/step4_fixtures.py:31`; 5 skip trong phase-b.xml | Sau train baseline chính thức, đặt STEP4_ARTIFACT_DIR và chạy đủ layout/hash/schema/reload/preds. Cần 895 good train, 230 good chuẩn hóa và predictions thật đủ 460/306 dòng. Chưa tuyên bố hoàn thành B03. |
| Minor — phạm vi thiết bị | `src/pcb_lab/models/efficientad/architecture.py:45`; environment.log là torch CPU | CUDA chưa được kiểm thực tế. Test nội suy forward/backward của implementer đã PASS trên CPU; cần smoke CUDA trước run GPU dài. |
| Major ở A — đã đóng cho CPU smoke | `tests/test_efficientad_phase_b.py:112`; numerical-audit.json | Có trace nguồn good-only, oracle độc lập và pre-save probe thật; không còn chặn do thiếu giao diện audit. Teacher số học runtime đối chiếu 3/384 kênh, không tuyên bố kiểm đủ mọi kênh. |

### Evidence ngoài git

Thư mục `./scratch/step4-phase-b/` chứa
`initial.log/xml`, `phase-b.log/xml`, `regression.log/xml`, `environment.log`,
`collect-all.log`. Thư mục
`pytest-phase-b/test_real_cpu_smoke_audits_sou0/` chứa `numerical-audit.json` và
`output/runs/smoke/B03_efficientad_s/seed42/` (checkpoint, manifest, audit, env,
calibration stats, model card, report). Có thể cần quyền ngoài sandbox để đọc
thư mục tạm pytest. Không commit các output này, weights hoặc dữ liệu.
