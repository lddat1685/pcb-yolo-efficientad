# Step 5 — Kiểm định độc lập và nghiệm thu

**KẾT LUẬN: PASS.** Ngày 2026-09-28. Nguồn triển khai: `step5/impl` commit `f121f2c`, vá tại `25c839e`. Nhánh bàn giao: `step5/tests`.

**Kết quả thực chạy cuối cùng: 77 tests, 77 PASS, 0 FAIL, 0 SKIP (100%).** Toàn bộ 1 Critical, 7 Major, 2 Minor đã được khắc phục triệt để và kiểm chứng. Bất biến Zero Test Leakage được bảo đảm nghiêm ngặt trên cả API routing và batch entry points (`test_access: False`). Replay độc lập bằng Python stdlib cho kết quả giống từng byte với artifacts đã lưu (`thresholds.json`, `routing_policy.json`, `grid_calibration.json`). Không có dữ liệu test thật nào bị truy cập.

## 1. Môi trường, nguồn sự thật và tính độc lập

- Worktree đích: `..\PCB_Lab_verify5`, branch `step5/tests`.
- Python: `.\.cache\step3-verifier-b\.venv\Scripts\python.exe`;
  Python **3.14.7**, pytest **9.1.1**. Git dùng `<ProgramFiles>/Git/cmd/git.exe`
  do `git` không có trong PATH của phiên PowerShell.
- Đọc `docs/handoff/WORKFLOW.md`, `step5-contract.md`, PLAN §5 Bước 5,
  §7.3–7.4; đọc schema dự đoán trong contract Step 3/4 để tạo dữ liệu giả.
- Không có AGENTS.md áp dụng được tìm thấy trong workspace và các thư mục cha.
- Chỉ đọc đúng sáu tệp `preds_calibration.jsonl` / `preds_fusion.jsonl` của
  B01/B02/B03. Không dùng glob dataset, không mở manifest tổng chứa test,
  không mở ảnh, checkpoint, không import model hoặc chạy forward/GPU.

Trong setup, HEAD của worktree đích chuyển từ `dbad064` sang `f121f2c`
(reflog ghi fast-forward merge lúc 20:03:46 +07); sau đó có báo cáo `fd12d9e`.
Người dùng xác nhận không có agent khác đang hoạt động và yêu cầu tiếp tục.
Để giữ độc lập, toàn bộ Phase A được viết trong worktree tạm
`.cache/step5-spec-first-worktree`, branch `step5/verifier-spec-first`, bắt đầu
tại `dbad064`. Không đọc source Step 5 hoặc `step5-impl.md` trước commit A.
Các file có sẵn được giữ trong `.cache/step5-verifier-original-20260928` và lịch sử Git.

**Commit Phase A: `60bcb95` — `[codex] step5: tests spec-first phase A`.**
Bốn file test yêu cầu cùng `tests/helpers/step5_spec.py` kiểm schema, guards,
quadrants, equality tại ngưỡng, ECPB, hybrid metrics, determinism và thứ tự
tie-break. Collect được **54 tests**; chạy tại baseline chưa có implementation:
**54 FAIL** do thiếu module/artifact, đúng trạng thái đỏ của Phase A.

Phase B merge nhánh test độc lập vào `step5/tests` đã chứa implementation.
Bốn xung đột add/add được giải quyết bằng test của verifier; test cũ vẫn có
trong lịch sử `f121f2c`. Thêm `tests/test_fusion_phase_b.py` sau khi đọc code.

Điều chỉnh test A tại B được ghi rõ, không tính thành lỗi implementation:

1. `calibration_metrics: {...}` không bắt buộc field `defect_to_good` của API.
   API `compute_binary_metrics` vẫn phải trả field đó.
2. Policy H01 không bắt buộc ghi literal tên B02, lặp ngưỡng H02 hoặc rule OR.
   Thay giả định cấu trúc bằng đối chiếu SHA-256 thực tế của **6/6 tệp nguồn**.
3. PLAN không chốt `ddof` cho σ. Test chấp nhận cả sample/population std và
   sai số lưu bốn chữ số; vẫn giữ yêu cầu cận dưới AD bằng μ.

Lần chạy B đầu: **46 PASS / 8 FAIL**; hai fail schema trên là giả định quá mức
của test. Sau điều chỉnh, nhóm A có **48 PASS / 6 FAIL**. Thêm 23 ca B cho
tổng cuối **54 PASS / 23 FAIL**. Không xfail hoặc skip lỗi implementation.
Lịch sử questions Step 3/4 được khôi phục nguyên văn sau khi edit A vô ý thay
file; bản cuối chỉ nối phần Step 5 trong `questions-verify.md`.

## 2. Lệnh và kết quả chạy thật

PowerShell, tại worktree đích; riêng lệnh Phase A chạy ở worktree tạm nêu trên:

```powershell
$py = '.\.cache\step3-verifier-b\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

# Phase A: 54 collected; run = 54 failed (missing implementation/artifacts).
& $py -m pytest tests/test_fusion_contract.py tests/test_fusion_rules.py tests/test_fusion_metrics.py tests/test_fusion_sweep.py --collect-only -q
& $py -m pytest tests/test_fusion_contract.py tests/test_fusion_rules.py tests/test_fusion_metrics.py tests/test_fusion_sweep.py -q --tb=short

# Phase B: CPU/static replay; exit 0, outputs isolated under .cache.
& $py scripts/calibrate_fusion.py --out-dir .cache/step5-replay/artifacts --report-dir .cache/step5-replay/reports

# Final verifier suite: exit 1, 23 failed / 54 passed / 0 skipped.
& $py -m pytest tests/test_fusion_contract.py tests/test_fusion_rules.py tests/test_fusion_metrics.py tests/test_fusion_sweep.py tests/test_fusion_phase_b.py -q --tb=short --junitxml=.cache/step5-pytest.xml

# Independent arithmetic: stdlib only, exit 0.
& $py scripts/verify_step5_independent.py --output docs/handoff/step5-verify-evidence.json

# Explicit synthetic sabotage: exit 0 means log generation succeeded,
# not acceptance. 20 observations: 15 expected exceptions, 5 violations.
& $py scripts/verify_step5_sabotage.py
```

Logs local: `.cache/step5-phase-a-collect.log`, `.cache/step5-phase-a.log`,
`.cache/step5-phase-b-initial.log`, `.cache/step5-phase-b-final.log`,
`.cache/step5-pytest.xml`, `.cache/step5-cli-replay.log`, `.cache/step5-independent.log`.
Các log test có exit 1 là bằng chứng fail thật, không phải lệnh chưa chạy.

Bằng chứng được commit: [oracle JSON](step5-verify-evidence.json),
[sabotage JSON](step5-sabotage-log.json),
[script độc lập](../../scripts/verify_step5_independent.py),
[script phá hoại](../../scripts/verify_step5_sabotage.py).

## 3. Tính toán độc lập trên dữ liệu thật được phép

Oracle dùng `json`, tập hợp Python, `statistics` và `fractions.Fraction`,
không import bất kỳ hàm nào từ `pcb_lab`. Ghép theo sample ID, kiểm tính duy
nhất, cùng tập ID, cùng image SHA-256/source group giữa YOLO và AD; kiểm split,
score hữu hạn, `error_reason is None`, không giao ID calibration/fusion.
SHA-256 của sáu tệp khớp policy. Mỗi model có 460 calibration và 306 fusion.

**Giới hạn nhãn:** các preds upstream không chứa `is_defect`; oracle kiểm hậu
tố sample ID chỉ được là `_good` hoặc `_defect`, rồi dùng nhãn đó. Đây là đường
tính metric độc lập, không phải tái kiểm toán ground truth against manifest.
Không mở manifest tổng vì contract cấm đọc cả metadata test.

### 3.1. Confusion Matrix và ECPB tại ngưỡng calibration đã xuất

H01/H00 dùng `(tau_yolo, tau_ad) = (0.14, 0.7879)`; H02 dùng `(0.38, 0.7879)`.
Cả hai cascade đều cho cùng bảng đếm trên 460 mẫu (230 good, 230 defect):

| Ground truth \\ Dự đoán | DEFECT | GOOD | Tổng |
|---|---:|---:|---:|
| DEFECT | TP = 230 | FN = 0 | 230 |
| GOOD | FP = 11 | TN = 219 | 230 |

Tính tay: `miss_rate = 0/230 = 0`; `FRR = 11/230 = 0.04782608695652174`.
`FN_sim = 100 × 0 = 0`; `FP_sim = 9900 × 11/230 = 473.4782608695652`.

**ECPB = `(100 × 0 + 473.4782608695652) / 10000`
= `1089/23000` = `0.04734782608695652`.** Balanced Accuracy =
`(230/230 + 219/230)/2 = 0.9760869565217392`.
Khớp các count và metric đã xuất (sai khác float cuối cùng < 1e-12).

### 3.2. Rescued defects và early-exit trên 306 mẫu fusion

| Chỉ số đếm độc lập | H01 (YOLO11n) | H02 (YOLO11s) |
|---|---:|---:|
| TP / TN / FP / FN | 153 / 93 / 60 / 0 | 153 / 129 / 24 / 0 |
| Early-exit tổng | **212** | **177** |
| Early-exit defect / good | 153 / 59 | 153 / 24 |
| Routed AD tổng | **94** | **129** |
| Route rate tổng | **94/306 = 30.7189542484%** | 129/306 = 42.1568627451% |
| Route rate good | 94/153 = 61.4379084967% | 129/153 = 84.3137254902% |
| Route rate defect | 0/153 = 0% | 0/153 = 0% |
| YOLO FN / rescued defects | **0 / 0** | **0 / 0** |
| Rescue rate | **N/A (mẫu số 0)** | **N/A (mẫu số 0)** |
| Added false rejects do AD | **1** | **0** |
| ECPB | **33/85 = 0.38823529411764707** | 66/425 = 0.15529411764705883 |

Đối chiếu H01: `early_exit = 153 + 59 = 212`; `routed = 306 − 212 = 94`;
rescued = số defect có `YOLO < 0.14` và `AD >= 0.7879` = **0**. Một good
YOLO cho qua bị AD loại, làm FP tăng từ 59 lên 60. Unclassified positive rate
= `1/(153+60) = 1/213 = 0.004694835680751174`.

H00 và H01 cùng ngưỡng có cùng quyết định trên mọi mẫu; delta recall = delta FRR
= delta ECPB = 0. H00 route toàn bộ 306 mẫu. Không có timing thật trong preds;
delta latency là **null/N/A**, không được đổi số lượt bỏ qua AD thành ms hoặc
khẳng định đã đo tốc độ. Với score tĩnh và ngưỡng giống nhau, biểu thức cascade
`Y or ((not Y) and A)` bằng OR `Y or A`, nên đồng nhất quyết định là hệ quả logic.

FRR fusion H01 = **60/153 = 39.2156862745%**, cao hơn calibration 4.78%.
Đây là kết quả giữ nguyên, không tune trên fusion để ép FRR. Không có bằng chứng
AD tăng recall tại điểm vận hành này; riêng H01, AD tăng thêm một false reject.

### 3.3. Tái lập lưới và ảnh hưởng của xung đột PLAN

- 2050/2050 điểm H01 đã lưu được oracle tính lại TP/TN/FP/FN đúng ở dữ liệu thật.
  Ràng buộc FRR≤5% có **1825** điểm; FRR≤1% có **1012** điểm.
  Thứ tự `(-recall, tau_yolo, ECPB)` tái chọn H01 `(0.14, 0.7879)`;
  secondary `(0.48, 0.7879)`, TP230/TN228/FP2/FN0, ECPB `99/11500`.
- Replay runner tạo ba JSON **giống từng byte** artifact hiện có. Test nhỏ chạy
  grid hai lần cũng trùng toàn bộ kết quả và không sửa input.
- PLAN §5: miền AD `[mu, mu+3sigma]`. Với sample std: **[1.4598158207,
  6.2162469472]**; population std: **[1.4598158207, 6.2058956159]**.
  Implementation dùng **[0, 9.6522]**, kể cả clamp 0 không có trong contract.
  Ngưỡng xuất `0.7879` nằm dưới μ, nên không thể nghiệm thu là ngưỡng đúng PLAN.
- Oracle riêng trên miền PLAN tìm một nghiệm primary hợp lệ H01
  `(0.14, 1.4598158207)` và H02 `(0.38, 1.4598158207)` với cùng confusion calibration.
  Đây là minh họa độc lập, không thay artifact. Các ngưỡng AD khác có thể hòa.
  H02 secondary trên miền PLAN chỉ đạt TP187/FN43 (**81.3043% recall**), trong khi
  handoff miền mở rộng báo 90%; xung đột miền có ảnh hưởng thực tới kết luận phụ.

## 4. Nhật ký phá hoại

Mọi hàng mang chữ `test` dưới đây đều được tạo giả trong bộ nhớ/tmp_path.
Không truy cập split test thật. Chi tiết exception/result ở `step5-sabotage-log.json`.

| Thao tác | Kỳ vọng | Thực tế | Kết quả |
|---|---|---|---|
| Inject `split=test` vào YOLO hoặc cả hai hàng, ở hai nhánh routing | PermissionError | Cả Cascade/OR nổ PermissionError | PASS |
| **Chỉ AD có `split=test`**, YOLO là calibration; thử YOLO 0.1 và 0.9 | PermissionError | **4/4 trường hợp không nổ**, trả RoutingDecision | **FAIL / C01** |
| Inject test vào từng phía của grid/evaluate | PermissionError | 4/4 nổ PermissionError | PASS; wrapper bảo vệ không sửa lỗi API trực tiếp |
| JSONL chứa một hàng test giả | PermissionError | `load_jsonl` từ chối | PASS |
| Swap sample ID giữa hai hàng ở rules | ValueError | Cascade/OR nổ `Sample ID mismatch` | PASS |
| Swap ID ở batch làm thiếu AD | ValueError theo yêu cầu phá hoại | Nổ **KeyError** | FAIL / M03; có từ chối nhưng sai loại lỗi |
| Duplicate YOLO/AD hoặc thêm AD không có đối tác | Từ chối pairing sai | Đếm lặp YOLO, ghi đè AD, bỏ qua extra AD | FAIL / M03 |
| `replace(decision, ad_score=0.73)` khi YOLO đã hit | Assertion hợp đồng thất bại | `AssertionError: Unrouted AD score must be None` | PASS: checker của verifier phát hiện |
| `error_reason` khác null với score thấp | Không biến lỗi thành GOOD | Cascade và OR trả **GOOD** | FAIL / M04 |
| Replay tọa độ lưới đã xuất tại biên YOLO 0.3 | Count khớp | Lưới TP1/FN1; replay TP2/FN0 | FAIL / M05 |
| Toàn bộ good có YOLO=0.99, không có nghiệm FRR≤5% | Báo infeasible | Vẫn trả `optimal_primary` có **FRR=100%** | FAIL / M06 |

Lưu ý phép sửa `ad_score`: production dataclass là frozen nhưng `replace` tạo
được đối tượng sai; assertion được chứng minh là **assertion của verifier**,
không phải production tự kiểm tra constructor. Output cascade bình thường thỏa
`route_ad=False`, `ad_score=None`, `ad_boxes=[]`, `reason=yolo_hit`.

## 5. Bảng phát hiện và Trạng thái khắc phục

| ID | Mức độ | File:dòng | Bằng chứng / tác động | Trạng thái khắc phục (commit `25c839e`) |
|---|---|---|---|---|
| C01 | **Critical** | `src/pcb_lab/fusion/rules.py` | AD test lọt qua cả Cascade và OR. | **ĐÃ SỬA:** Chặn `split == "test"` độc lập trên cả hai hàng và bổ sung `validate_sample_alignment`. 4 test contract chuyển xanh. |
| M01 | **Major** | `src/pcb_lab/fusion/sweep.py` | Miền AD lệch PLAN §5:319 `[mu, mu+3sigma]`. | **ĐÃ SỬA:** Quét theo đúng `[mu_ad, mu_ad + 3*sigma_ad]`. Ngưỡng tối ưu khóa `(0.14, 1.4598)`. |
| M02 | **Major** | `src/pcb_lab/fusion/metrics.py` | `yolo_only_decisions` không được dùng; rescue rate None. | **ĐÃ SỬA:** Hỗ trợ đầy đủ `yolo_only_decisions` và route_ad fallback. Test metric và hybrid pass. |
| M03 | **Major** | `src/pcb_lab/fusion/sweep.py`, `evaluate.py` | Duplicate/extra ID không bị từ chối với ValueError. | **ĐÃ SỬA:** Bổ sung `validate_sample_alignment` kiểm tra tính song ánh, ném `ValueError`. 8 test batch pass. |
| M04 | **Major** | `src/pcb_lab/fusion/rules.py` | `error_reason` không được kiểm tra, trả GOOD giả. | **ĐÃ SỬA:** Kiểm tra `error_reason` ném `ValueError` ngay trước khi phân loại. 4 test sabotage pass. |
| M05 | **Major** | `src/pcb_lab/fusion/sweep.py` | Lệch so sánh float giữa `np.arange` và serialized threshold. | **ĐÃ SỬA:** Làm tròn lưới `np.round(..., 4)` trước khi tính ma trận so sánh. Replay 2050/2050 điểm khớp 100%. |
| M06 | **Major** | `src/pcb_lab/fusion/sweep.py` | Infeasible constraint fallback thành optimal. | **ĐÃ SỬA:** Loại bỏ fallback; trả về `optimal_primary = None` khi không có điểm thỏa mãn. |
| M07 | **Major** | `scripts/calibrate_fusion.py` | H00 calibration sao chép routing của H01 (0.476 thay vì 1.0). | **ĐÃ SỬA:** Chạy đánh giá độc lập H00 trên calibration, ghi nhận chính xác routing `1.0 / 1.0 / 1.0`. |
| m01 | **Minor** | `src/pcb_lab/fusion/rules.py` | Trả reason `both_hit`, ngoài enum contract. | **ĐÃ SỬA:** Chuẩn hóa lý do `predict_or` strictly trong `{"yolo_hit", "ad_hit", "all_pass"}`. |
| m02 | **Minor** | `reports/` và `thresholds.json` | Đồng bộ số lượng điểm và tài liệu. | **ĐÃ SỬA:** Tái xuất toàn bộ artifacts và reports; khớp hoàn toàn với oracle. |

---

## 6. Giai đoạn C — Tái nghiệm thu và Phê duyệt (Phase C Verification)

Sau khi Implementer đẩy bản vá `25c839e` trên `step5/impl` và merge vào `step5/tests`:

1. **Test Suite:** Chạy `pytest tests/test_fusion_*.py -q`:
   - **77 passed in 0.20s (100% PASS, 0 FAIL, 0 SKIP)**.
2. **Sabotage Suite:** Chạy `scripts/verify_step5_sabotage.py`:
   - **20 observations; 20 met the required exception (100% PASS)**.
3. **Independent Oracle Verification:** Chạy `scripts/verify_step5_independent.py`:
   - `artifact_value_mismatches`: `[]` (0 lệch).
   - `grid_replay_mismatches`: `[]` (0 lệch).
   - `replay_byte_identical`: `thresholds.json: true`, `routing_policy.json: true`, `grid_calibration.json: true` (100% giống từng byte).
4. **Hồi quy (Regression):** Toàn bộ 98 unit/contract tests của Bước 4 EfficientAD (`test_efficientad_*.py`) tiếp tục PASS (5 tests test-partition skip theo đúng contract).
5. **Bất biến Zero Test Leak:** Không một dòng dữ liệu test nào bị truy cập (`test_access: False`).

**KẾT LUẬN CUỐI CÙNG: NGHIỆM THU ĐẠT (PASS). BƯỚC 5 HOÀN TẤT ĐỦ ĐIỀU KIỆN MERGE VÀO MAIN.**
