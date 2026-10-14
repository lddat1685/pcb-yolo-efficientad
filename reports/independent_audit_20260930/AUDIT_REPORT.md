# Biên bản kiểm toán độc lập toàn đề tài — Bước 1–10

**KẾT LUẬN CUỐI: FAIL — chưa đủ điều kiện nghiệm thu toàn bộ đề tài theo PLAN.md trước bảo vệ.**

Kiểm toán ngày 30/09/2026, trên nhánh `main`, commit `42f34fd49bf9149ae652f9ac6b7521ea936e4739`. Đây là kết luận về bằng chứng và khả năng tái lập hiện có, không phải kết luận rằng toàn bộ mô hình hoặc dữ liệu sai. Các số đếm phân loại được xác nhận; các tuyên bố về phép đo latency, tính đóng băng và độ tin cậy của bộ kiểm tra export có lỗi chặn nghiệm thu.

Chỉ thị bổ sung của người dùng: tiếp tục toàn bộ kiểm toán theo `PLAN.md`. Căn cứ: PLAN §1–§2, §5–§9, §11; `docs/handoff/WORKFLOW.md`; các hợp đồng, biên bản, nhật ký; mã nguồn, tests, reports, artifacts và exports hiện có.

Không sửa mã nguồn sản phẩm, test gốc, PLAN, checkpoint, ngưỡng hoặc báo cáo đóng băng. Không train lại baseline hoặc chọn lại ngưỡng. Có chạy smoke train nhỏ do test suite yêu cầu; mọi đầu ra mới nằm trong thư mục kiểm toán. Hai detector được suy luận lại trên test chỉ để xác minh cấu hình đã khóa. Không sử dụng kết quả đó để chọn mô hình mới.

## 1. Kết quả thực thi và phạm vi bằng chứng

| Kiểm tra đã thực chạy | Kết quả | Bằng chứng trong thư mục này |
|---|---|---|
| Script tái lập theo chỉ thị | **60 PASS, 0 FAIL, 0 MISSING**, exit 0 | `reproducibility.log` |
| Audit dataset chạy lại, output riêng | **PASS**, 2.998 ảnh, 0 lỗi, 18/18 release files khớp | `data_audit/data_audit.json`, `dataset_run.log` |
| Oracle Python standard library từ raw predictions và manifest | **3.080 dòng**, đủ 7 × 440; 0 sai số chỉ số tổng hợp được đối chiếu | `recompute.py`, `independent_numbers.json` |
| Calibration/development bằng oracle độc lập | 2.050 điểm H01 khớp; H02 tái tính riêng ra ngưỡng 0,38/1,4598; 6/6 checksum nguồn khớp | `audit_development.py`, `development_provenance.json`, `h02_grid_recomputed.json` |
| Toàn bộ test suite với B01/B03 thật được chỉ định | **427 passed, 11 failed, 10 skipped**, 248,95 giây | `pytest.log`, `pytest.xml` |
| Kiểm thêm artifact B02 + YOLO smoke | **9 passed, 1 failed**; smoke vẫn thiếu `git_commit` trong môi trường chạy | `pytest_b02.log`, `pytest_b02.xml` |
| EfficientAD smoke với teacher có sẵn | **1 passed**, 253,08 giây; chạy lại sau khi quyền đọc ngoài sandbox được cấp | `pytest_teacher_verified.log`, `pytest_teacher_verified.xml` |
| Bù khoảng trống kiểm thử augmentation | 64 seed, đủ 8 phép biến đổi dihedral; ảnh/box nhất quán; 4 partition ngoài train bị từ chối | `supplemental_checks.py`, `supplemental_results.json` |
| Suy luận lại YOLO từ checkpoint khóa | B01 và B02: **440/440 image score khớp chính xác**, 0 decision flip mỗi model | `replay_yolo.py`, `replay_box_metrics.json`, `replay_*_test_boxes.jsonl` |
| Oracle box AP trên raw boxes suy luận lại | 12 AP@0.5 khớp chính xác công thức 11 điểm của Step 7; counts/recall từng lớp khớp | `replay_box_metrics.json`, `replay_metrics.log` |
| CLI ↔ Streamlit AppTest với checkpoint thật | **10/10 ca**, 15 trường logic khớp chính xác, không so timing/run ID | `ui_cli_parity.log` |
| TorchScript đối chiếu trực tiếp, không dùng hàm parity đang kiểm toán | **3/3 mẫu calibration**: map/score delta 0, decision khớp | `runtime_replay.py`, `runtime_direct_parity.json` |
| Phá hoại guard đóng băng và parity | Tái hiện bỏ qua ngưỡng/checksum và PASS sai khi export lỗi | `adversarial_probes.py`, `adversarial_results.json` |
| Tái xuất 5 cặp bảng luận văn + README vào output riêng | 11/11 nội dung khớp bản bàn giao; phát hiện lỗi LaTeX ở 3 bảng | `thesis_regenerated/`, `supplemental_results.json` |

Các script tính toán số học chính `recompute.py` và `audit_development.py` chỉ dùng standard library, không import metric/evaluator của dự án. Trong replay detector, Ultralytics chỉ tạo dự đoán; ghép box, confusion, AP, percentile và ECPB được tính bằng đường độc lập. Replay boxes là **bằng chứng mới**, không giả nhận là raw boxes của lần chạy Step 7 ban đầu.

Không coi 60/60 là kiểm chứng khoa học toàn hệ thống: phần lớn tiêu chí trong `scripts/verify_reproducibility.py` chỉ kiểm file tồn tại. Parity chỉ đọc cờ `passed` đã lưu; script không suy luận lại, không tính metric từ raw predictions, không kiểm đầy đủ mọi checksum trong bảng tài liệu.

## 2. Giai đoạn 1 — Toàn vẹn và môi trường tái lập

- HEAD đúng `main`; working tree không có thay đổi tracked trước kiểm toán. Ba file untracked có sẵn (`PCB_Defect_Detection_Presentation.pptx`, `create_ppt.py`, `docs/diagrams/mermaid-config.json`) nằm ngoài phạm vi nghiệm thu main và được giữ nguyên.
- `main` local đi trước `origin/main` 56 commit; không fetch/push/merge. Kết luận này áp dụng đúng HEAD local nêu trên, không mặc nhiên áp dụng bản remote.
- File frozen giữ nguyên SHA-256 `038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd`.
- Ba checkpoint B01/B02/B03 đúng full SHA trong frozen; hash run manifest của ba artifact cũng khớp sidecar hiện tại.
- Hash danh sách test được tính độc lập bằng SHA-256 của JSON danh sách ID đã sort: `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24`.
- Runtime kiểm toán: Python 3.14.7, torch 2.14.0 CPU, torchvision 0.29.0, Ultralytics 8.4.161, NumPy 2.5.3, Pillow 12.3.0; chi tiết lưu trong `supplemental_results.json`.
- Repo được import qua `PYTHONPATH`/test configuration. Không thực hiện cài mới môi trường từ README, không tuyên bố đã kiểm chứng clean installation. README cài nhiều ML dependency không pin; `pyproject.toml`, requirements Step 3 và Step 4 đang có các pin khác nhau. Cần lock riêng cho từng môi trường train/inference.

Lưu ý về lịch sử: timestamp tự khai trong frozen và lịch sử Git hiện có không tự chứng minh đầy đủ thứ tự mọi lần chạy ngoài repo. Không phát hiện bằng chứng đủ để cáo buộc đã tuning trên test; đồng thời các guard hiện tại không bảo đảm điều đó. Xem M01.

## 3. Giai đoạn 2 — Dữ liệu, loader, training và isolation

| Partition | Good | Defect | Tổng | Nhóm nguồn |
|---|---:|---:|---:|---:|
| train | 895 | 897 | 1.792 | 5 |
| calibration | 230 | 230 | 460 | 2 |
| fusion | 153 | 153 | 306 | 2 |
| test | 220 | 220 | 440 | 2 |

Audit thực đọc/decode, kiểm nhãn, hash ảnh/pixel và 18 release files; 0 lỗi. Oracle riêng kiểm ID/group/pair/hash trên cả 6 cặp partition: không giao nhau. Số box test của sáu lớp lần lượt **298 / 215 / 250 / 228 / 194 / 222**, tổng **1.407**.

Đối chiếu ID trong normal-only audit với manifest: **895/895 train good**, không defect hoặc test. Danh sách chuẩn hóa AD: **230/230 calibration good**. Nguồn thống kê teacher là train good theo run manifest và các tests; không coi quan sát metadata này là việc phát lại toàn bộ training GPU. Smoke EfficientAD bổ sung đã kiểm nguồn mẫu, thống kê và lần save đầu qua test thật.

YOLO B01 ghi 33 epoch, best 13, 1.260,5 giây; B02 43 epoch, best 23, 1.695,3 giây. AD ghi 70.000 optimizer steps, 9 epoch thực, 4.682,796 giây. Các trường hiện tại đã được refresh; không lặp lại cáo buộc cũ rằng chúng còn null/100 epoch.

Checkpoint YOLO `last.pt` của cả hai model có **`epoch=-1`, `optimizer=None`** khi nạp thực. Chúng dùng được cho suy luận, nhưng không đáp ứng yêu cầu lưu checkpoint phục hồi đầy đủ trạng thái training trong PLAN Bước 3. Test resume cũng thất bại độc lập; xem M03.

TileManager, preprocessing, heatmap/bbox và loaders có nhiều kiểm thử chạy xanh. Chín test augmentation trong suite gốc bị skip do helper không nhận ra `apply_augmentation`, không phải module chưa có. Kiểm tra riêng 64 seed đã xác nhận chức năng cơ bản; helper/test gốc vẫn cần sửa để CI không bỏ qua kiểm thử.

## 4. Giai đoạn 3 — Hiệu chỉnh và tính lại raw predictions

Từ calibration raw scores: AD defect mean **1,4598158206628717**; grid dùng sample standard deviation (`ddof=1`). Oracle tính mean/std và trục lưới độc lập, xác nhận 41 × 50 = **2.050** điểm H01. Mọi confusion/metric được đối chiếu trong lưới lưu đều khớp.

Điểm chính H01 **0,14 / 1,4598** đạt recall 100%, FRR **11/230 = 4,7826%** trên calibration. Điểm phụ **0,48 / 1,4598** hợp tiêu chí FRR ≤1%. Có nhiều điểm hòa, nên đây không phải một nghiệm tối ưu duy nhất được dữ liệu chứng minh. H02 tái quét từ raw B02/AD cho ra điểm chính **0,38 / 1,4598**. Repo chỉ lưu toàn bộ lưới H01; `scripts/calibrate_fusion.py:234` không lưu toàn bộ lưới H02 dù đã quét.

Tất cả 3.080 dòng test có đúng ID canonical, nhãn và group đúng manifest, không thiếu/trùng mẫu, không status lạ. Quyết định lưu khớp phép so score với ngưỡng frozen hiện tại. Kết quả độc lập:

| Model | TP | FN | FP | TN | BalAcc | Recall | FRR | ECPB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B01 | 220 | 0 | 22 | 198 | 95,0000% | 100% | 10% | 0,099000 |
| B02 | 220 | 0 | 77 | 143 | 82,5000% | 100% | 35% | 0,346500 |
| B03 | 80 | 140 | 0 | 220 | 68,1818% | 36,3636% | 0% | 0,636364 |
| H00 | 220 | 0 | 22 | 198 | 95,0000% | 100% | 10% | 0,099000 |
| H01 | 220 | 0 | 22 | 198 | 95,0000% | 100% | 10% | 0,099000 |
| H02 | 220 | 0 | 77 | 143 | 82,5000% | 100% | 35% | 0,346500 |
| H01_sec | 219 | 1 | 11 | 209 | 97,2727% | 99,5455% | 5% | 0,054045 |

ECPB được tính từ số đếm thực, không đọc lại ô ECPB: với prevalence mô phỏng 1% và penalty 100:1 trong PLAN, `ECPB = miss_rate + 0.99 × FRR`. Đây là chi phí theo giả định mô phỏng, không phải đo chi phí nhà máy thực.

H01: **198/440** ảnh được định tuyến theo luật; good **198/220**, defect **0/220**. Early-exit theo luật **242/440**. `rescued_defects=0`, `added_false_rejects=0`, rescue rate **N/A** vì YOLO có 0 FN tại ngưỡng chính. H01_sec có 1 FN YOLO và cứu **0/1**. Trên 440 ảnh, H01_sec hoàn toàn trùng với YOLO11n đơn ở cùng ngưỡng 0,48; cải thiện so với B01 ngưỡng 0,14 không phải bằng chứng AD cứu lỗi.

| Model | Worst-group recall | Worst-group FRR |
|---|---:|---:|
| B01/H00/H01 | 100% | 16% |
| B02/H02 | 100% | 50,8333% |
| B03 | 33,3333% | 0% |
| H01_sec | 99,1667% | 9% |

Nhóm 44000: 100 good/100 defect; nhóm 92000: 120 good/120 defect. Các chỉ số group đối chiếu khớp. H01 không đạt mốc FRR ≤5% trên test; H01_sec đạt mức tổng thể nhưng không đạt ở group xấu nhất. Mốc test là mục tiêu nghiên cứu, không được sửa ngưỡng để ép đạt.

## 5. Giai đoạn 4 — Kiểm chứng giả thuyết và ma trận thí nghiệm

| Giả thuyết | Kết luận kiểm toán | Bằng chứng/giới hạn |
|---|---|---|
| H1: AD cứu một phần YOLO bỏ sót | **Chưa được chứng minh tại các điểm khóa** | Điểm chính không có FN YOLO; secondary test có 1 FN nhưng AD không cứu. Không suy diễn AD vô dụng ở mọi operating point. |
| H2: lợi ích hybrid đáng báo nhầm và chi phí | **Không được ủng hộ bởi thí nghiệm hiện có** | B01/H01/H00 có cùng 440 quyết định; H01 thêm suy luận AD mà không thêm TP. Calibration/fusion cũng cứu 0 ảnh tại điểm chính. |
| H3: ảnh hưởng resolution/tiling tới lỗi nhỏ | **Chưa kiểm chứng** | Đã thống kê bbox nhỏ, nhưng thiếu ablation bắt buộc tile-256 vs resize-256. Thống kê kích thước không thay thí nghiệm này. |
| H4: cân bằng trên group mới và nhiều seed | **Chưa đủ bằng chứng** | Chỉ seed42; FRR H01 tăng 4,78% calibration → 38,56% fusion → 10% test; worst test group 16%. N=2 không chứng minh domain shift bất kỳ. |
| H5: cascade giữ recall của OR, giảm chi phí | **Đúng phần quyết định/routing; chưa đạt bằng chứng latency test theo protocol** | Mười ca CLI/UI xác nhận early-exit thật; latency Step 7 là tổng số đo nhánh, Step 9 OR chỉ 20 mẫu. Không dùng chúng để chứng nhận mức tiết kiệm E2E trên cùng 440 ảnh. |

Một điểm cần sửa cả trong diễn giải PLAN: với cùng checkpoint, input, ngưỡng và các nhánh tất định không lỗi, cascade cho quyết định `Y OR A`, đúng bằng OR. Vì thế **recall và FRR của H00/H01 phải bằng nhau**; early-exit tự nó không giảm FRR. Lợi ích có thể nằm ở lượng tính toán/latency. Không lấy sự bằng nhau về accuracy làm bằng chứng thực nghiệm rằng cascade tự cải thiện khả năng phát hiện.

PLAN §11.2 cho phép ba seed **hoặc giải thích rõ giới hạn nếu chưa đủ**. Vì vậy thiếu seed43/44 không tự động được coi là lỗi đủ để đánh trượt riêng lẻ. Tuy nhiên, H4 không thể được xác nhận; cần ghi rõ đây là benchmark một seed trong báo cáo cuối. Thiếu tile-vs-resize và iso-FRR/DET vẫn là khoảng trống thí nghiệm đã được yêu cầu ở PLAN §5 Bước 6, §7.4, §9.

Không có bằng chứng cho phép gọi H01 là tối ưu. Câu khẳng định tại `reports/step7_final_test_evaluation_report.md:77` trái số liệu và tinh thần chấp nhận kết quả âm của PLAN. `docs/LIMITATIONS.md:83` đã thừa nhận baseline có thể phù hợp hơn; báo cáo chính và shortlist cần nhất quán với điều này. Không được dùng chính lần audit này để chọn cấu hình mới rồi gọi là lựa chọn trước test.

## 6. Giai đoạn 5 — Các lỗi chặn nghiệm thu và kiểm thử thực thi

| ID | Mức | Vị trí | Phát hiện và bằng chứng | Điều kiện khắc phục |
|---|---|---|---|---|
| C01 | **Critical** | `src/pcb_lab/evaluation/test_eval.py:254`, `:267`, `:355`, `:370`, `:384`; PLAN §5 Bước 7, §7.5 | Decode nằm ngoài timer; AD chạy cho mọi ảnh trước khi dựng hybrid; latency H00/H01/H02/H01_sec là phép cộng nhánh. **1.760/1.760** giá trị hybrid đúng bằng tổng tương ứng, residual 0. `route_ad` là tuyến giả lập từ score, không phải log thực thi cascade của lần chạy này. | Dùng pipeline chung để đo E2E thật, warmup từng model, cùng tập/thiết bị/precision; tách cold start, đo các lượt timing theo protocol. Giữ kết quả cũ và ghi nhãn chúng là latency tổng hợp/ước tính, không sửa âm thầm. |
| C02 | **Critical** | `scripts/export_runtime.py:163`, `:248`, `:293` | Lỗi load/inference export bị chuyển thành delta=0 và chép decision gốc. Hai fault injections đều trả `passed=true`, matches 1/1 khi không có kết quả export hợp lệ. Điều kiện PASS cũng không xét đầy đủ map/box delta hoặc số mẫu thực đo. | Lỗi phải là FAIL/UNVERIFIED; ghi exception, kiểm đủ coverage và các tolerance đã đăng ký; nạp đúng file export được yêu cầu. Chỉ ghi PASS khi backend đích thực sự chạy. |
| M01 | **Major** | `src/pcb_lab/evaluation/test_eval.py:121`, `:204`, `:311`; `scripts/evaluate_test.py:32` | Evaluator dùng `OPERATING_POINTS` global thay frozen. Synthetic frozen B01=0,99, score=0,5 vẫn cho **440 DEFECT thay vì 0**. Status=DRAFT, test hash sai, sample_count sai và checkpoint SHA sai đều qua tới loader. Lệnh mặc định còn tạo lại frozen và ghi đè output cũ. | Validate full freeze và dataset/checkpoint hashes trước inference, lấy mọi tham số từ freeze, từ chối overwrite hoặc tạo version riêng; thêm test phá hoại trực tiếp evaluator. Hiện freeze và constants trùng nhau nên lỗi này không tự chứng minh số đếm công bố sai hoặc đã có test leakage. |
| M02 | **Major** | `src/pcb_lab/evaluation/test_eval.py:386`, `:541`, `:596`; `artifacts/test/predictions_test.jsonl` | Raw test lưu score/decision nhưng bỏ detections, checkpoint/config/image hash, timing stages… nên không thể tính lại AP của run gốc chỉ từ file này. Step 7 tự viết evaluator AP 11 điểm và không xuất mAP@0.5:0.95 theo hợp đồng/PLAN; không dùng evaluator thống nhất. | Lưu raw boxes/candidates và đủ provenance; pin định nghĩa matching/AP, xuất AP50/50:95 và recall tại ngưỡng riêng; thống nhất evaluator. Replay audit xác nhận số AP11 hiện tại, nhưng không thay yêu cầu raw evidence của run gốc. |
| M03 | **Major** | `tests/test_yolo_adversarial.py:53`, `:121`, `:207`; `tests/test_yolo_config.py:43`; `tests/test_yolo_extract.py:65`; `tests/test_yolo_view.py:72`; YOLO `last.pt` | Các regression thực: thiếu guard run directory, chấp nhận lớp thứ 7, resume không chuyển xuống trainer, config lạ không bị chặn, run_id sai, stale view. `last.pt` cả hai model không còn optimizer/epoch phục hồi. Full suite chưa xanh. | Sửa production và tests theo hợp đồng, lưu checkpoint resumable đúng nghĩa, chạy lại regression. Không miễn trừ lỗi chỉ vì xuất hiện ở bước trước. |
| M04 | **Major** | `reports/step7_final_test_evaluation_report.md:77`; `artifacts/benchmark/shortlist.json`; PLAN §7.6 | Khuyến nghị H01 “tối ưu” không được raw predictions ủng hộ; không cứu lỗi, cùng decision với B01; FRR development chính 38,56%. H01_sec hơn B01 ở ngưỡng khác cũng trùng YOLO-only cùng ngưỡng. | Viết lại kết luận về kết quả âm/trade-off và lựa chọn đã đăng ký. Không quảng bá giá trị cứu lỗi chưa quan sát; nếu nghiên cứu thêm sau test cần protocol/version phù hợp. |
| M05 | **Major** | `reports/step6_baseline_benchmark_report.md:53`; `src/pcb_lab/evaluation/ablation.py:153`; PLAN §7.4/§9 | Chưa có tile-vs-resize, chưa có bảng Recall@FRR={1%,3%,5%}/DET của các baseline bắt buộc. Một seed chỉ cho phép kết luận giới hạn. Lưới đầy đủ H02 không được lưu. | Bổ sung thí nghiệm phát triển và bản ghi có version; mô tả rõ phần chưa làm, không gọi toàn bộ ma trận đã hoàn tất. Không điều chỉnh test đã mở để tạo kết quả thuận lợi. |
| m01 | **Minor** | `README.md:99`, `:105`, `:211`; `docs/REPRODUCIBILITY.md:13`, `:149`; `pyproject.toml:10` | Đường dẫn worktree cũ; dependency ML không pin; tuyên bố script hash “mọi artifact/toàn repo” rộng hơn code thật. `--out-dir` relative của exporter gây ValueError; absolute path trong repo chạy được. | Dùng repo-root chung, pin môi trường, mô tả đúng phạm vi verifier, xử lý đường dẫn CLI. Kiểm clean install riêng trước giao nộp. |
| m02 | **Minor** | `exports/thesis/overall_comparison.tex:12`, `per_group_breakdown.tex:18`, `confusion_counts.tex:7` | `H01_sec` không escape `_` trong text-mode tabular; `.tex` overall thiếu cột p95 có ở `.md`. Chưa có TeX engine để compile thực trong môi trường này. | Escape model IDs, đồng bộ cột giữa hai định dạng, compile LaTeX trước đưa vào luận văn. Không coi kiểm có `tabular` là kiểm biên dịch. |
| m03 | **Minor** | `docs/handoff/step3-verify.md:506`; `step4-verify.md:127`; `step6-verify.md:262`; `step10-verify.md` | Trạng thái lịch sử chưa được đối chiếu nhất quán: Step 3 kết thúc FAIL, Step 4 conditional; Step 6 thừa nhận legacy tests đỏ, nhưng Step 10 tuyên bố sẵn sàng toàn repo. Một số metadata cũ đã sửa, nhưng chưa có biên bản đóng toàn bộ lỗi. | Lập closure matrix từng phát hiện, liên kết commit/bằng chứng mới; giữ lịch sử nguyên trạng thay vì đổi verdict cũ. |

**Phân loại đầy đủ 11 test fail của lượt toàn suite:** 9 failure thuộc nhóm regression YOLO đã liệt kê (hai parameterizations của extraction và ba của unknown config được tính riêng); 1 fail oracle AP50 B01; 1 fail smoke provenance `git_commit`.

AP oracle B01: `open_circuit` chênh **−0,0209254** so với report calibration, vượt tolerance 0,02 của test hiện hành. Sự khác biệt convention/matching phải được giải quyết trước khi kết luận metric tương đương; **không** dùng riêng sai lệch này để cáo buộc checkpoint hỏng. Artifact B02 có 9/9 test artifact riêng chạy PASS.

Smoke YOLO đã thực train một epoch và đối chiếu reload, nhưng fail ở `git_commit` trong môi trường này; thêm Git vào PATH chưa giải quyết được lookup subprocess với cwd của helper. Giữ đây là lỗi tái lập/provenance cần điều tra, không quy thành lỗi chất lượng detector. Nó không làm mất hiệu lực các regression độc lập còn lại.

Full-suite skips: 9 augmentation do helper không khớp API; 1 EfficientAD teacher do chưa đặt biến môi trường. Audit đã kiểm riêng augmentation và chạy lại teacher smoke thành công. **Không** cộng những lượt bổ sung thành một tuyên bố “full suite xanh”.

Step 9 có profiling thật: B01/B03/H01 đều 460 calibration records, H00 20 records. Tính lại mean/p50/p95 từ raw runtime records khớp số đã lưu theo làm tròn. Tuy nhiên, H00 và H01 khác cỡ tập, không thể dùng hai mean toàn tập để kết luận mức tăng tốc công bằng. B01 export ONNX/OpenVINO đang fallback PyTorch; parity với chính backend gốc không chứng minh backend mới đã chạy.

Kiểm TorchScript trực tiếp ba mẫu ở min/median/max AD calibration score cho delta map/score=0, decision khớp. Điều này xác nhận ba phép so sánh cụ thể; không hợp thức hóa guard parity đang PASS sai. CLI/UI 10 ca kiểm chứng hoạt động core routing thật, gồm YOLO-positive, AD-positive, GOOD, OR và các model đăng ký; các test synthetic còn kiểm ERROR. Không tuyên bố đã thử thủ công toàn bộ tương tác trình duyệt hoặc stress UI dài hạn.

## 7. Giai đoạn 6 — Bảng nghiệm thu Bước 1–10 và bàn giao

| Bước | Verdict trong phạm vi kiểm toán | Lý do |
|---|---|---|
| 1. Audit dữ liệu | **PASS** | Audit chạy lại, integrity/leakage/counts khớp; các giới hạn N=2 và nguồn template được công bố. |
| 2. Loader, preprocessing, TileManager | **PASS có giới hạn coverage CI** | Tests thực và kiểm augmentation riêng đạt; 9 tests gốc vẫn skip do helper. |
| 3. YOLO baselines | **FAIL** | Checkpoint suy luận tốt nhưng regression, oracle calibration AP và resume/provenance chưa khép lại. |
| 4. EfficientAD | **PASS trong phạm vi CPU/artifact đã xác minh** | Hash, normal-only lists, normalization, reload và teacher smoke đạt; không phát lại toàn bộ GPU training. |
| 5. Calibration | **PASS phần số học/ngưỡng; còn thiếu log lưới H02** | H01 2.050 điểm và H02 independent optimum khớp; không có sai số metric đã đối chiếu. |
| 6. Baseline/ablation/shortlist | **FAIL nghiệm thu đầy đủ** | Thiếu ablation bắt buộc/iso-FRR; lựa chọn và kết luận lợi ích hybrid chưa được chứng minh. |
| 7. Frozen test | **FAIL** | Latency không phải đo pipeline thật, guard freeze sai, raw detection/provenance và metric detection chưa đủ. |
| 8. UI | **PASS trong phạm vi thực thi đã kiểm** | Suite UI và CLI/AppTest 10 ca khớp; bảng benchmark trong UI kế thừa giới hạn Step 7. |
| 9. Runtime/export | **FAIL nghiệm thu** | Profiler chạy được, một số export thật đúng; parity checker có thể PASS khi hỏng và benchmark đối chứng chưa đủ. |
| 10. Đóng gói/bảo vệ | **FAIL** | Không thể công bố toàn đề tài PASS khi C01/C02 và các Major còn mở; docs/exports còn lỗi. |

Kết quả âm của H1/H2 tự nó **không** làm luận văn thất bại: PLAN cho phép kết luận baseline đơn phù hợp hơn. Lý do FAIL là phép đo/guard/bằng chứng và kết luận bàn giao chưa đáp ứng yêu cầu đã công bố.

Thứ tự xử lý trước tái kiểm toán:

1. Sửa C01/C02/M01, thêm negative-control tests trực tiếp và đo lại pipeline/backend thật bằng ngưỡng/checkpoint hiện tại. Tạo bản artifact sửa lỗi có provenance; bảo toàn toàn bộ số liệu gốc.
2. Đóng regression YOLO, xác định nguyên nhân oracle AP và smoke provenance, bảo đảm resume có trạng thái đúng. Chạy toàn suite và phân loại mọi skip rõ ràng.
3. Lưu raw boxes và metadata đủ để người khác tính metric mà không phải suy luận lại; pin evaluator và xuất metric detection còn thiếu.
4. Hoàn thiện thí nghiệm bắt buộc trên dữ liệu được phép và sửa phần kết luận H1–H5/model selection. Những nghiên cứu mới sau khi biết test phải có protocol/version và phạm vi xác nhận phù hợp; không coi test cũ là tập chọn cấu hình mới.
5. Đồng bộ README/cards/reports/thesis, compile `.tex`, thử clean installation, rồi lập biên bản nghiệm thu lại gắn đúng commit mới.

Giới hạn của cuộc kiểm toán: chưa train lại ba baseline đầy đủ hoặc chạy seed43/44; chưa đo lại 7 pipeline × 440 ảnh × 3 timing rounds; không tạo ablation thiếu; không có TeX compiler; không chứng nhận hardware camera, production AOI, hay domain shift ngoài DeepPCB. Đây là những phần cần ghi rõ, không được thay bằng số ước đoán.

## 8. Lệnh tái kiểm và bản đồ bằng chứng

Chạy từ repo-root bằng PowerShell, sử dụng interpreter đã chỉ định trong yêu cầu:

```powershell
$auditPy = '.cache/step3-verifier-b/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:DATASET_ROOT = '../DatasetVer4_Public'
$env:STEP3_ARTIFACT_DIR = Join-Path (Get-Location) 'artifacts/yolo/B01_yolo11n/seed42'
$env:STEP4_ARTIFACT_DIR = Join-Path (Get-Location) 'artifacts/efficientad/B03_efficientad_s/seed42'
& $auditPy scripts/verify_reproducibility.py
& $auditPy reports/independent_audit_20260930/recompute.py
& $auditPy reports/independent_audit_20260930/audit_development.py
& $auditPy reports/independent_audit_20260930/adversarial_probes.py
& $auditPy -m pytest tests/ -q -ra -p no:cacheprovider --basetemp=reports/independent_audit_20260930/rerun_tmp
& $auditPy scripts/verify_step8_parity.py
& $auditPy reports/independent_audit_20260930/runtime_replay.py
```

`adversarial_results.json` chứa các quan sát lỗi có chủ ý; exit 0 của script thu thập bằng chứng không có nghĩa guard production PASS. Các file `probe_frozen_*` và `synthetic_evaluation/` là **fixtures phá hoại/synthetic**, tuyệt đối không dùng làm kết quả khoa học hoặc thay frozen gốc.

`development_provenance.json` chứa inventory/hash/trích trạng thái của **33 file step*** hiện có trong handoff, cùng kết quả calibration, normal-only và runtime. `supplemental_results.json` chứa phiên bản môi trường, checksum manifests, trạng thái last.pt và kiểm exports. `evidence_index.json` liệt kê hash các nguồn chính và bằng chứng; `git_final_status.txt` ghi trạng thái kết thúc.

**Chứng nhận kiểm toán:** số đếm và chất lượng phân loại tại các điểm khóa có bằng chứng tái lập tốt. Hồ sơ hiện tại **chưa được ký PASS cho bảo vệ** cho đến khi các lỗi Critical/Major được khép lại bằng thực thi và bằng chứng mới.
