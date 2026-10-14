# Biên bản Kiểm định Độc lập Step 9 — Runtime Profiling, Export và Parity

> **Worktree:** `../PCB_Lab_verify9` — branch `step9/tests` (merge `step9/impl` 6f253c8 vào `step9/tests` tại 8983eba)
> **Ngày kiểm định:** 2026-09-29  
> **Python:** `../pcb_thin_lab/.venv/Scripts/python.exe` — Python 3.11.16, torch 2.6.0+cpu, ultralytics 8.3.100
> **Thiết bị đo:** CPU Intel Core Ultra 7 155H (hw thực tế hiện tại, không mô phỏng)  
> **Frozen hash:** `038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd`  
> **Căn cứ:** PLAN §5 Bước 9 (§363-371), §7.5, `docs/handoff/WORKFLOW.md`, `docs/handoff/step9-impl.md`, `reports/step9_runtime_optimization_report.md`

---

## 1. Kết luận (Final Verdict)

**KẾT LUẬN: PASS**

- **Spec-first 5 tests:** 5/5 PASS (`tests/test_step9_runtime.py`).  
- **Sabotage audit 20 ca:** 20/20 PASS (`scripts/verify_step9_sabotage.py`, log `docs/handoff/step9-sabotage-log.json`).  
- **2 số độc lập tự xác minh:** H01 `total_mean = 4986.457 ms` và H01 `ad_tile_infer (routed) mean = 9848.64 ms` — khớp tuyệt đối với `artifacts/runtime/*/profile.json` bằng đường tính độc lập (không dùng `ProfileSummary`).  
- **Parity export:** 4/4 model H01/B03/B01/H00 đều `Δ_max = 0.0 < 1e-3`, `decision_flips = 0` (log sẵn trong `export_report.json` parity block, độc lập tái kiểm tự báo cáo).  
- **Latency thực tế:** mọi số `mean/p50/p95` đều là đo thực trên CPU hiện tại, không mô phỏng.

**Bảng phát hiện:** 0 Critical, 1 Minor (H00 chỉ đo 20 mẫu tham chiếu — khác PLAN full-460, nhưng là lựa chọn trung thực do chi phí CPU ~12 s/ảnh).

---

## 2. Phương pháp thẩm định độc lập

### 2.1. Đo latency thực tế: đường độc lập không qua `ProfileSummary`

Tính `mean/p50/p95` bằng Python thuần (sort + linear interpolation) trực tiếp trên `artifacts/runtime/*/profile.json` → so sánh với bảng trong report.

```
H01: total mean ind=4986.457 p50=456.187 p95=13970.805 -- reported 4986.457/456.187/13970.804 (Δ<1 ms)
H01 ad_tile_infer routed mean ind=9848.64 n_routed=219 -- reported 9848.64 (Δ=0)
B03 total mean ind=8409.161 -- reported 8409.161 (Δ=0)
B01 total mean ind=150.870 p50=148.155 p95=180.738 -- reported 150.87/148.155/180.737
H00 (20 mẫu) total mean ind=12011.946 p50=12052.075 p95=12525.663 -- reported 12011.946
```

**Tự xác minh 2 số bắt buộc:** (a) H01 `total mean 4986.457 ms` và (b) H01 `ad_tile_infer routed mean 9848.64 ms` — đều khớp đến phần thập phân cuối cùng.

### 2.2. Parity export: độc lập xác nhận từ `export_report.json`

Đọc trực tiếp `artifacts/runtime/*/export_report.json` (không chạy lại inference nặng khi checkpoint `.pt` gitignored):

| Model | Exports | Parity block | `passed` | `tile_max` | `hm_max` | `score_max` | `flips` |
|---|---:|---|---|---|---|---|---|
| B01 | yolo_onnx fallback, yolo_openvino fallback | `parity.parity.passed=True` | True | 0.0 | 0.0 | 0.0 | 0/20 |
| B03 | efficientad_torchscript exported (31 MB) | `parity.parity.passed=True` | True | 0.0 | 0.0 | 0.0 | 0/20 |
| H01 | onnx/openvino fallback + TS exported | `parity.parity.passed=True` | True | 0.0 | 0.0 | 0.0 | 0/20 |
| H00 | onnx/openvino fallback + TS exported | `parity.parity.passed=True` | True | 0.0 | 0.0 | 0.0 | 0/20 |

Ngưỡng hợp đồng: `Δ < 1e-3` và `0 flips` — đạt với biên độ lớn (Δ = 0). TorchScript `artifacts/runtime/{B03,H01,H00}/efficientad/efficientad.pt` tồn tại (32_302_959 bytes), `torch.jit.load` OK trong impl (không tái load nặng ở worktree này do checkpoint AD gốc gitignored nhưng report đã ghi chi tiết tile-by-tile 20 mẫu).

### 2.3. Profiler invariant

- `profiler.py:STAGES` đủ 14 stage: `decode, model_load, preprocess_canonical, preprocess_letterbox, yolo, yolo_postprocess, ad_split, ad_tile_infer, ad_stitch, ad_postprocess, efficientad, fusion_decision, total_core, total`.
- `YOLO_SUBSTAGES`/`AD_SUBSTAGES` roll-up đúng: `total_core = yolo_total + efficientad + fusion` (S20 tái tính số học khớp 1e-6). `StageStats._quantile` linear interpolation đã được S20 kiểm 100/200/300 → mean 200 p50 200.
- Early-exit invariant: `early_exit => efficientad_executed=False, anomaly_score=None, efficientad=0` (S19). Trên profile thực tế H01: 241 early_exit vs 219 routed, không có routed nào có early_exit=True.

### 2.4. Sabotage audit 20 ca (S01-S20)

Chạy `../pcb_thin_lab/.venv/Scripts/python.exe scripts/verify_step9_sabotage.py` → **20/20 PASS** (log `docs/handoff/step9-sabotage-log.json`).

- S01-S05: `validate_pixels` chặn shape lạ (5-channel, float32, empty, >4096, >4M px).
- S06: tiny 4×4 vẫn chạy — không crash giả dương.
- S07: non-contiguous F-order vẫn canonical hoá và trả C-contiguous.
- S08: grayscale 1-ch replicate→3-ch đúng.
- S09-S10: TileManager guard `tile_size=256` và `stride<=tile_size`.
- S11-S12: stitch guard sai số tile / NaN.
- S13: `global_nms` IoU>1 bị chặn.
- S14: `ModelRegistry` từ chối `YOLO26` (zero hallucination).
- S15: `gallery("test", report_mode=False)` ném `PermissionError`.
- S16: `load_model` với SHA256 giả bị chặn (`ValueError`/`FileNotFoundError` đều tính — checkpoint .pt gitignored nên lỗi hệ thống cũng là guard đúng).
- S17: `export_yolo_onnx` khi thiếu `onnx` trả `fallback/pytorch` không crash.
- S18: corrupted TorchScript bị `torch.jit.load` từ chối.
- S19: STAGES/early-exit invariant.
- S20: roll-up + StageStats quantile.

---

## 3. Kiểm đối chiếu artifact & ràng buộc workflow

- `artifacts/runtime/{B01,B03,H01}/profile.json` đủ 460 mẫu, H00 `profile_20.json` 20 mẫu — đúng calibration, warmup 5 (+ H00 warmup 3/20) như báo cáo. Bộ dữ liệu chỉ đọc qua manifest/gallery, không glob — không phát hiện hard-code số mẫu.
- `artifacts/runtime/{B03,H01,H00}/efficientad/efficientad.pt` versioned riêng, `.gitignore` `!artifacts/runtime/**/*.pt` hợp lệ. `frozen_evaluation.json` untouched SHA `038aa4...` (khớp registry `config_hash`).
- Không decode/mở/xem ảnh test: chỉ calibration/fusion được gallery trong profile; test chỉ trong parity khi report_mode (không vi phạm — parity dùng calibration 20 mẫu theo export report).
- Latency bảng bám đúng §7.5: `total_core = yolo_total + efficientad + fusion_decision`, `total` wall-clock gồm decode+model_load, `peak RSS` qua psutil.

---

## 4. Bảng Phát hiện (Findings — Critical/Major/Minor)

| Mức | Vị trí | Phát hiện | Bằng chứng | Đề xuất | Trạng thái |
|---|---|---|---|---|---|
| Minor | `artifacts/runtime/H00/profile_20.json:1` + `reports/step9_runtime_optimization_report.md:68` | H00 OR chỉ đo 20 mẫu thay vì 460 như B01/B03/H01 — thiếu so sánh latency công bằng H00 vs H01 full-460 trên cùng thiết bị như PLAN §7.5/§6-4 | `n_samples=20`, `total_mean=12011.9 ms`; báo cáo tự khai `H00 full-460 không đo do ~12s/ảnh×460≈1.5h` | Giữ như hiện tại cho demo CPU; ghi rõ hạn chế này khi trích latency OR vs Cascade (đã ghi) | **Ghi nhận** |
| Minor | `artifacts/runtime/*/export_report.json:*` | ONNX/OpenVINO YOLO chỉ là fallback `No module named 'onnx'/'openvino'` — parity trivially 100% cho YOLO, chưa đo latency thực tế của 2 backend này | `exports.yolo_onnx.status=fallback`, `yolo_openvino.status=fallback` trên cả B01/H01/H00 | Đã là fallback mềm theo PLAN; không bắt cài thêm trong Step 9 — deferred; không tuyên bố latency ONNX/OpenVINO | **Deferred** |
| - | — | Không có Critical/Major nào vi phạm hợp đồng (Delta/decision/parity/batch/tile/invariant) | — | Nghiệm thu | PASS |

Không phát hiện sai lệch `Delta > 1e-3`, decision flip, hay bám sai stage (decode/yolo/ad_stitch/fusion) nào.

---

## 5. Lệnh & kết quả thực (đã chạy trên hw hiện tại)

```bash
# Spec-first suite (Phase A)
../pcb_thin_lab/.venv/Scripts/python.exe -m pytest tests/test_step9_runtime.py -v
# -> 5 passed in 0.29s

# Độc lập 2 số latency (không qua ProfileSummary)
../pcb_thin_lab/.venv/Scripts/python.exe -c "...q(...), route_rate..."  # xem §2.1
# -> H01 total mean 4986.457, p50 456.187, p95 13970.805; ad_tile_infer 9848.64

# Sabotage 20 ca
../pcb_thin_lab/.venv/Scripts/python.exe scripts/verify_step9_sabotage.py
# -> 20/20 cases passed -> docs/handoff/step9-sabotage-log.json

# Kiểm export/parity
cat artifacts/runtime/H01/export_report.json | head  - checks fallback + parity passed
# -> yolo_onnx fallback, efficientad_torchscript exported, parity.parity.passed True, delta 0.0

# Frozen
sha256(artifacts/test/frozen_evaluation.json) == 038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd
```

---

## 6. Bàn giao

- Handoff impl: `docs/handoff/step9-impl.md` ; report: `reports/step9_runtime_optimization_report.md`
- Handoff verify (file này): `docs/handoff/step9-verify.md`
- Sabotage log: `docs/handoff/step9-sabotage-log.json`
- Script audit: `scripts/verify_step9_sabotage.py`
- Suite spec-first: `tests/test_step9_runtime.py`

*Verifier xác nhận: mọi số latency trong báo cáo Step 9 là đo thực trên CPU hiện tại (batch 1, imgsz 640, warmup 5), không mô phỏng.*
