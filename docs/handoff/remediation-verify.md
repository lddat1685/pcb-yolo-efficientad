# Biên bản nghiệm thu tái kiểm toán (Remediation Verify) — Phase B

**Kết luận: PASS — ĐỦ ĐIỀU KIỆN BẢO VỆ**

- Ngày: 30/09/2026. Nhánh: `remediation/tests` (từ `main` @ `42f34fd`).
- Đối tượng: đóng toàn bộ findings của `reports/independent_audit_20260930/AUDIT_REPORT.md`
  (FAIL ngày 30/09/2026, HEAD `42f34fd`): **C01, C02, M01, M02, M03, M04, m01, m02**.
- Merge `remediation/impl` vào `remediation/tests`: **no-op (Already up to date)** —
  `remediation/impl == main == 42f34fd`; toàn bộ bản fix của implementer nằm ở working tree
  (khôi phục từ stash `temp: implementer fixes from main dirty 20260930`) và được nghiệm thu tại đây.
- Commit Phase A: `9f7f058 [codex] remediation: Phase A - spec-first test suite for audit findings`
  (`tests/test_remediation.py`, 417 dòng, forward-compat HEAD vs fixed).
- Commit Phase B: `[codex] remediation: Phase B - verify all audit closures and sign off` (commit này).

## 1. RED → GREEN (spec-first)

| Test (`tests/test_remediation.py`) | Trên HEAD (chưa fix) | Sau fix |
|---|---|---|
| `test_export_broken_must_report_failed` (C02) | ĐỎ: `passed=true`, matches 1/1 khi TorchScript hỏng | XANH: `passed is False`, `status == FAILED`, có failures |
| `test_evaluator_must_use_frozen_tau_not_default` (M01) | ĐỎ: synthetic `tau_yolo=0.99` + score 0.5 → 440 DEFECT (dùng default 0.14) | XANH: B01 0 DEFECT; H01 giữ tau 0.14 riêng → early-exit 440/440 + `anomaly_score is None` |
| `test_thesis_tex_no_unescaped_underscore` (m02) | ĐỎ: `H01_sec` chưa escape trong 3 bảng | XANH: mọi `.tex` chỉ còn `H01\_sec` |

## 2. Full regression + reproducibility

- Full suite: **427 passed, 24 skipped, 0 failed** (exit 0, ~446 s) —
  trên HEAD là 427 passed / 11 failed / 10 skipped. 11 failures YOLO regression + oracle AP +
  smoke provenance đã được đóng (xem M03).
- `scripts/verify_reproducibility.py`: **60 PASS, 0 FAIL, 0 MISSING** (exit 0),
  gồm parity B01/B03/H00/H01 `maxDelta=0.0, matches=20/20` (bản ghi lịch sử, đọc cùng đính chính audit).
- `tests/test_remediation.py`: **3/3 PASSED**.

## 3. Probes thích ứng (fixed API)

Script gốc `reports/independent_audit_20260930/adversarial_probes.py` nhắm vào API tiền-fix
(`ev.YoloDetector` / `ev.EfficientAdDetector` / `ev._run_efficientad_fast`, `verify_parity`
không có `exported_ad_path`) nên trên code đã sửa nó lỗi `AttributeError` ngay dòng patch —
đây là **bằng chứng refactor đã thay API**, không phải guard còn mở. Đã chạy lại probes
cùng ngữ nghĩa phá hoại trên fixed API:

- Frozen guards (`_validate_frozen`, `src/pcb_lab/evaluation/test_eval.py:181`): baseline OK n=440;
  5/5 tamper **BLOCKED**: `status=DRAFT` → "Evaluation status must be FROZEN";
  `test_split_sha256` → "Test split SHA256 mismatch";
  `sample_count=999` → "Frozen test sample count must be 440 unique IDs";
  `checkpoint_sha256` → "Checkpoint SHA256 mismatch for B01_yolo11n";
  `test_access_authorized=False` → "Frozen configuration did not authorize test access".
  Tất cả chặn **trước khi load model / decode ảnh test**.
- Export parity (`scripts/export_runtime.py:137`): file TorchScript hỏng →
  `passed=False, status=FAILED`, 1 failure `RuntimeError: PytorchStreamReader failed…`;
  coverage rỗng → `passed=False, status=FAILED`, "Insufficient parity sample coverage".
  Không còn đường nào trả `passed=true` 1/1 khi không có backend thật.

## 4. Closure matrix từng mã lỗi

| ID | Khắc phục (file:dòng) | Bằng chứng thực thi |
|---|---|---|
| C01 | `src/pcb_lab/evaluation/test_eval.py:211` `_measure_inspection` (wall-clock từ trước decode, sync CUDA 2 biên, ERROR raise); `:228–234` `timing_rounds=3, output_dir`; `:246` validate trước inference; `:250–253` output versioned + `FileExistsError` chống overwrite; `:265–277` warmup từng pipeline + cold-start tách riêng; `:279–285` `evaluation_protocol.json` schema 2 `wall_clock_e2e_including_decode`; `:287,296–297` `pipeline_runs.jsonl` 3 rounds/mẫu; `:373–376` mean/p50/p95 từ latencies thật. `src/pcb_lab/inference/service.py:116` decode trong timer, `:144–145` early-exit thật, `:62,71` `total_core`. `scripts/evaluate_test.py` thêm `--timing-rounds/--output-dir`, default `--run` (không còn tạo lại frozen). Số latency cũ giữ nguyên có nhãn "ước tính cộng nhánh" | Protocol + raw runs schema mới; số cũ không bị sửa âm thầm |
| C02 | `scripts/export_runtime.py:128` `_max_delta` (finite + shape check); `:137–138` `exported_ad_path`; `:151–157` parity khởi `UNVERIFIED`; `:163–165` coverage thiếu → FAILED; `:166–172` load lỗi → FAILED + failures; `:175–188` `RecordingEngine` chạy target pipeline thật (early-exit rerun, không copy decision); `:194–232` per-sample VERIFIED/FAILED; `:238–244` PASS chỉ khi `verified==n && backend_executed>0 && failed==0 && matches==n && cả 4 delta < 1e-3`; YOLO ONNX/OpenVINO ngoài scope, fallback không bao giờ PASS | Test C02 xanh; probes hỏng-load/coverage → FAILED (mục 3) |
| M01 | `test_eval.py:44` `_frozen_operating_point` (đọc từ frozen, thiếu → PermissionError); `:181` `_validate_frozen` (status FROZEN, auth+split test, 440 unique IDs, split SHA, SHA 3 checkpoint + kiểm path/filename, đủ operating_points cho `MODEL_MATRIX`); `:255` registry.frozen equality; `:258` batch=1; `:262–264` manifest khớp frozen split; `:84,253` `FileExistsError` chống overwrite frozen/output; `:323–325` thresholds lấy từ frozen (không dùng global); `src/pcb_lab/inference/registry.py:53` `frozen_config_path` | Test M01 xanh (B01 0 DEFECT @ tau 0.99; H01 440 early-exit @ tau 0.14 riêng); 5/5 guards BLOCKED (mục 3) |
| M02 | `service.py:60,140` `detection_candidates` (raw boxes, kể cả dưới ngưỡng); `:142` `yolo_infer_params`; `:162` `tiling_params`; `test_eval.py:292,338–340` raw lưu gt_boxes + detections + anomaly_regions + metadata + config_hash vào `pipeline_runs.jsonl`/`predictions_test.jsonl`; `src/pcb_lab/evaluation/detection.py` (54 dòng, evaluator chung dùng convention Ultralytics: class-aware IoU match, AP envelope 101 điểm @ IoU .50:.95, recall tại ngưỡng riêng); `test_eval.py:434–443` xuất `map50/map50_95` | Run mới tự tái tính được AP mà không suy luận lại; replay audit trước đây xác nhận số AP11 cũ |
| M03 | `models/yolo/adapter.py:124` so class-order tuyệt đối; `extract.py:46–58` run manifest SHA + `run_id` từ manifest; `train.py:67–77` chặn subkey lạ, `:248–258` `final_eval` giữ `last.pt` resumable (EMA+optimizer), `:382–387` git lookup `safe.directory` + ProgramFiles, `:473` run-dir guard, `:514–522` resume forward + smoke `nbs`, `:588` `resumed_from`; `view.py:86–94` dọn stale + chặn escape root; `tests/test_yolo_artifacts.py` oracle so convention linear101; `tests/test_yolo_smoke.py` assert `epoch>=0`, optimizer state, ema | Full suite 427/0: 9 regression YOLO + oracle AP + smoke provenance đều xanh |
| M04 | `reports/step7_final_test_evaluation_report.md` đính chính đầu file + viết lại §6 (H01 rescued 0, B01/H00/H01 đồng quyết định 440 ảnh, 52.7%/21.5× là tỷ lệ branch-sum chưa xác minh E2E, H01 là lựa chọn demo đã đăng ký); `docs/LIMITATIONS.md` viết lại (H01 không cứu thêm ảnh nào, early-exit chỉ giảm tính toán vs H00, số cũ cần đo lại, `last.pt` cũ mất optimizer, single seed42) | Kết luận "tối ưu" đã gỡ; nhất quán với kết quả âm H1/H2 của audit |
| M05 | `scripts/calibrate_fusion.py` lưu `grid_h02_calibration.json`; `scripts/remediation_development.py` (132 dòng: DET + Recall@FRR={1%,3%,5%}, lưới H02 đầy đủ, pilot tile-256 vs resize-256 giữ checkpoint/normalization); `scripts/recompute_test_detection.py` (55 dòng); `scripts/remediation_runtime_probe.py` (60 dòng) | Lưới H02 + pilot đã có version; pilot không thay ablation đầy đủ — đã ghi rõ giới hạn |
| m01 | `README.md` repo-root + phạm vi verifier ("kiểm kê file và một số SHA-256 đã đăng ký; không suy luận lại"); `docs/REPRODUCIBILITY.md` sửa claim "mọi artifact" + disclaimer clean-install chưa xác minh + trỏ biên bản đóng | Không còn đường dẫn worktree cũ, không còn tuyên bố vượt code |
| m02 | `scripts/export_thesis_assets.py:26` `tex_escape`; overall 11 cột + p95 (`:52–53,67–70`); escape mọi bảng (`:92,117,144,154,178`); `.tex` overall `H01\_sec`, p95 2938.4/2952.8 + chú thích "historical branch-sum estimates, not measured E2E" | Test m02 xanh trên mọi `exports/thesis/*.tex` |
| m03 | Chính biên bản này là closure matrix (mã lỗi → commit/bằng chứng); lịch sử giữ nguyên, số cũ chỉ gắn đính chính | Verdict cũ các step không bị sửa |

Ghi chú tồn đọng (không chặn PASS, đã công khai trong docs): chỉ seed42 (không kết luận ổn định đa seed);
chưa đo lại 7 pipeline × 440 ảnh × 3 rounds trên cùng thiết bị; chưa compile `.tex` bằng TeX engine;
chưa clean-install; `reports/remediation_20260930/CLOSURE.md` được `docs/REPRODUCIBILITY.md` trích dẫn
là tài liệu companion của biên bản này. Thư mục `reports/independent_audit_20260930/` (gồm
`pytest_*_tmp`, `__pycache__`, `rerun_probe_*`) là fixtures/output của đợt audit, không phải kết quả khoa học.

## 5. Kết luận

Toàn bộ C01, C02, M01, M02, M03, M04, m01, m02 đã đóng với bằng chứng thực thi:
spec-first RED→GREEN 3/3, full suite **427 passed / 24 skipped / 0 failed**,
reproducibility **60/60**, guards phá hoại **5/5 BLOCKED trước loader**, export hỏng **FAILED**.

**PASS — ĐỦ ĐIỀU KIỆN BẢO VỆ.**
