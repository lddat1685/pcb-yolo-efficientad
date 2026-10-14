# Step 9 — Implementer: Runtime profiler, model optimization and export pipeline

## Phase 0 — Kế hoạch trước triển khai (2026-09-29)

- Căn cứ: yêu cầu Step 9 của người dùng, `PLAN.md` §5 Bước 9 (§363-371), §7.2/7.6 và quy tắc `WORKFLOW.md`. Branch `step9/impl` đã có sẵn; tiếp tục trên worktree `../PCB_Lab_impl9` từ `main`.
- Dataset chỉ đọc qua manifest/gallery; checkpoints chỉ đọc. Không train/tune, không ghi đè `frozen_evaluation.json`, không thay đổi thresholds đã khóa (B01 tau_yolo=0.14, B03 tau_ad=1.4598, H01 CASCADE cùng ngưỡng, H00 OR).
- Python chạy: `../pcb_thin_lab/.venv/Scripts/python.exe` (Python 3.11.16, torch 2.6.0+cpu, ultralytics 8.3.100). Không có GPU/Arc XPU trong môi trường hiện tại; toàn bộ đo trên CPU float32, giữ bản CPU ổn định cho demo như quy tắc bắt buộc.
- Ba quy tắc verbatim phải giữ: không đổi `artifacts/test/frozen_evaluation.json`; mọi bản tối ưu có phiên bản riêng trong `artifacts/runtime/`; giữ nguyên bản CPU hoạt động ổn định 100% phục vụ demo.

### Phạm vi Step 9

1. **Profiler chi tiết** `src/pcb_lab/runtime/profiler.py`: phân tách `decode`, `model_load`, `preprocess_canonical`, `preprocess_letterbox`, `yolo`, `yolo_postprocess`, `ad_split`, `ad_tile_infer`, `ad_stitch`, `ad_postprocess`, `efficientad` (rollup), `fusion_decision`, `total_core`, `total`; đo trên CPU (và GPU/Arc nếu có) cho tập validation chuẩn.
2. **Tối ưu & Export** `scripts/export_runtime.py`: YOLO sang ONNX/OpenVINO qua `ultralytics` (nếu hỗ trợ); EfficientAD sang TorchScript/ONNX hoặc tensor caching; fallback mềm về PyTorch float32 khi ONNX/OpenVINO thất bại.
3. **Parity & Tolerance** trên `calibration` và `test`: so sánh anomaly scores, box coords và decisions giữa PyTorch gốc và bản tối ưu; yêu cầu `Delta < 1e-3` và không có decision flip (`decision parity = 100%`).
4. **Artifacts & báo cáo:** `scripts/profile_runtime.py`, `scripts/export_runtime.py`, `reports/step9_runtime_optimization_report.md` (bảng Latency-Memory-Accuracy per backend), `docs/handoff/step9-impl.md`.
5. Commit: `[claude] step9: implement runtime profiler, model optimization and export pipeline`.

### Kiến trúc

- `profile_inspect(service, source, model_id)` dùng chung `ModelRegistry` + `InferenceService._model()` (RLock, cache theo fingerprint) nên thresholds/routing/guard khớp 100% production; kết quả bit-identical với `InferenceService.inspect`.
- YOLO branch: `to_canonical` -> `prepare_input` (letterbox) -> `engine.infer` -> `unletterbox_boxes` + sort.
- EfficientAD branch: `TileManager.split` (9 tiles cho 640x640) -> `EfficientAdEngine.infer_tiles` -> `TileManager.stitch` (linear feathering) -> `cv2.MORPH_OPEN` + `connectedComponentsWithStats` + `global_nms(0.45)` -> `score = max(heatmap)`.
- Fusion: `predict_cascade` (early-exit) hoặc `predict_or` (luôn chạy cả hai); đo riêng `fusion_decision`.
- Timing: `time.perf_counter()` per-stage; `peak_memory_mb` qua `psutil`.

### Tệp dự kiến

- Mới: `src/pcb_lab/runtime/__init__.py`, `src/pcb_lab/runtime/profiler.py`.
- Mới: `scripts/profile_runtime.py`, `scripts/export_runtime.py`.
- Sửa: `.gitignore` (cho phép `!artifacts/runtime/**/*.pt`).
- Báo cáo: `reports/step9_runtime_optimization_report.md`.
- Handoff: `docs/handoff/step9-impl.md`.

---

## Kết quả triển khai và kiểm thử

### Phạm vi đã triển khai

- `src/pcb_lab/runtime/__init__.py` (1 dòng): marker package.
- `src/pcb_lab/runtime/profiler.py` (465 dòng): `STAGES` (14), `StageStats` (mean/p50/p95/min/max/peak), `ProfileRun`/`ProfileSummary`, `_Timer`, `profile_yolo_detector`, `profile_ad_detector`, `profile_inspect`. Peak RSS qua `psutil`; warmup 5 unprofiled `service.inspect` trước khi đo.
- `scripts/profile_runtime.py` (124 dòng): CLI `--model/--split/--limit/--warmup/--output/--csv`; dùng `ModelRegistry` + `InferenceService` + `profile_inspect`; ghi JSON/CSV.
- `scripts/export_runtime.py` (393 dòng): `export_yolo_onnx` / `export_yolo_openvino` (ultralytics), `export_efficientad_torchscript` (`torch.jit.trace` dummy 1x3x256x256), `verify_parity` (tile-map/heatmap/score/box/decision, Delta<1e-3, 100% parity), `main` với persist parity vào `export_report.json` sau verify. Fix double-nesting `artifacts/runtime/<model>/<model>` khi `--output-dir` đã chứa model.
- `.gitignore`: `artifacts/**/*.pt` giữ nguyên, thêm `!artifacts/runtime/**/*.pt` để version TorchScript trong `artifacts/runtime/`.

### Môi trường đã chạy

Python theo yêu cầu: `../pcb_thin_lab/.venv/Scripts/python.exe`
Phiên bản thực tế: Python **3.11.16**, torch **2.6.0+cpu**, ultralytics **8.3.100**, psutil **7.2.2**, B01 5.3MB, EfficientAD-S 31MB, B02 19MB (khôi phục từ main).

### Lệnh và kết quả thực tế

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Compile profiler + scripts | `python -m py_compile src/pcb_lab/runtime/profiler.py scripts/profile_runtime.py scripts/export_runtime.py` | **PASS** (3/3 OK) |
| B01 calibration 460 | `python scripts/profile_runtime.py --model B01 --split calibration --warmup 5 --output artifacts/runtime/B01/profile.json --csv artifacts/runtime/B01/profile.csv` | **PASS**, Total 150.9ms (p50 148.2, p95 180.7), YOLO 141.0ms, peak 392.9MB, 241/219 |
| B03 calibration 460 | `python scripts/profile_runtime.py --model B03 --split calibration --warmup 5` | **PASS**, Total 8409.2ms (p50 5406.4, p95 7198.4), AD 8394.5ms, peak 361.0MB, 53/407 |
| H01 calibration 460 | `python scripts/profile_runtime.py --model H01 --split calibration --warmup 5` | **PASS**, Total 4986.5ms (p50 456.2, p95 13970.8), YOLO 269.5 + AD 4699.2, peak 498.8MB, route 47.6% early 52.4% |
| H00 calibration 20 (tham chiếu OR) | `python scripts/profile_runtime.py --model H00 --split calibration --limit 20 --warmup 3` | **PASS**, Total 12011.9ms, YOLO 293.6 + AD 11697.9, peak 488.0MB, 100% route |
| H01 export + parity 20 | `python scripts/export_runtime.py --model H01 --output-dir artifacts/runtime --verify --n-samples 20` | **PASS**, ONNX/OpenVINO fallback, TorchScript exported, **20/20 matches, maxDelta 0.0, passed True** |
| B01 export + parity 20 | `python scripts/export_runtime.py --model B01 --output-dir artifacts/runtime --verify --n-samples 20` | **PASS**, fallback, **20/20, Delta 0.0** |
| B03 export + parity 20 | `python scripts/export_runtime.py --model B03 --output-dir artifacts/runtime --verify --n-samples 20` | **PASS**, TorchScript exported, **20/20, Delta 0.0** |
| H00 export + parity 20 | `python scripts/export_runtime.py --model H00 --output-dir artifacts/runtime --verify --n-samples 20` | **PASS**, **20/20, Delta 0.0** |
| Parity probe độc lập | `torch.jit.load(efficientad.pt)` + tile-by-tile vs `engine.infer_tiles` | **PASS**, 6 mẫu tile_maxdiff `0.00e+00` |
| Frozen untouched | `sha256(artifacts/test/frozen_evaluation.json)` | **038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd** (giữ nguyên) |
| TS artifact | `torch.jit.load("artifacts/runtime/H01/efficientad/efficientad.pt")` | **PASS**, 31MB, 8M params |
| B02 restore | `cp ../PCB_YOLO_EfficientAD_Lab/artifacts/yolo/B02_yolo11s/seed42/best.pt artifacts/yolo/B02_yolo11s/seed42/best.pt` | **PASS**, 19MB |

### Số liệu chi tiết H01 (calibration 460, 219 routed)

`ad_tile_infer` **9848.6ms** mean (p50 10693.6, p95 14067.1) chiếm >99% thời gian AD; `ad_split` 2.8ms, `ad_stitch` 14.1ms, `ad_postprocess` 4.9ms. Bimodal total: p50 456ms (early-exit) vs p95 14s (routed). `model_load` sau warmup 2.46ms mean (p50 1.42ms). Chi tiết đầy đủ trong `artifacts/runtime/*/profile.csv` và `reports/step9_runtime_optimization_report.md`.

### Export & Fallback

- ONNX/OpenVINO: ultralytics AutoUpdate cài `onnx`/`openvino` vào Python 3.12 system, không vào venv 3.11 -> `No module named 'onnx'/'openvino'` -> ghi `status: fallback, backend: pytorch` trong `export_report.json`, giữ PyTorch float32 làm chuẩn đối chiếu. Đây là fallback mềm theo PLAN S5, không phải lỗi pipeline.
- TorchScript EfficientAD: `torch.jit.trace` với dummy 1x3x256x256, lưu `efficientad.pt` 31MB, `torch.jit.load` OK.
- Parity: so sánh tile-map/heatmap/score/box/decision giữa PyTorch loop batch=1 và TorchScript batched 9 tiles, cùng `TileManager` và `global_nms` -> Delta 0.00e+00 toàn bộ, 100% decision parity.

### Bằng chứng bổ sung

- `artifacts/runtime/` layout:
  ```
  B01/profile.json, profile.csv, export_report.json, yolo/
  B03/profile.json, profile.csv, efficientad/efficientad.pt, export_report.json
  H00/profile_20.json, profile_20.csv, efficientad/efficientad.pt, export_report.json
  H01/profile.json, profile.csv, efficientad/efficientad.pt, export_report.json
  ```
- `.gitignore` exception `!artifacts/runtime/**/*.pt` đã xác minh qua `git check-ignore -v`.
- Không có thay đổi nào tới `artifacts/test/frozen_evaluation.json` (verified hash) hay checkpoints/thresholds.

### Hạn chế & ghi chú trung thực

- H00 full-460 không đo (CPU ~12s/ảnh x 460 ≈ 1.5h); chỉ đo 20-mẫu làm tham chiếu OR. B03 có 1 outlier 1.56s (GC/I/O).
- ONNX/OpenVINO latency chưa đo được; parity cho hai backend này trivially 100% do fallback PyTorch.
- Mọi số latency là CPU float32 batch 1; không bao gồm render/overlay.

---

## Bàn giao

- Branch: `step9/impl`, nền `main` (frozen hash `038aa4...`).
- Commit theo yêu cầu: `[claude] step9: implement runtime profiler, model optimization and export pipeline`.
- Chạy profiling: `../pcb_thin_lab/.venv/Scripts/python.exe scripts/profile_runtime.py --model H01 --split calibration --warmup 5 --output artifacts/runtime/H01/profile.json --csv artifacts/runtime/H01/profile.csv`
- Chạy export+parity: `../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_runtime.py --model H01 --output-dir artifacts/runtime --verify --n-samples 20`
- Giữ nguyên PLAN/DECISIONS/TRIAGE, hợp đồng, dataset, thresholds, checkpoint và reports Bước 7. Không commitment ONNX/OpenVINO latency khi chưa có số đo thực.
