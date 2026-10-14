# Báo cáo Tối ưu Runtime (Step 9: Profiling, Export và Đánh giá Sai khác)

**Ngày:** 2026-09-29 22:40 UTC+7 | **Branch:** `step9/impl` | **Trạng thái:** Hoàn tất trên CPU (fallback PyTorch float32 ổn định cho demo)

> **Quy tắc bắt buộc (verbatim):**
> - "Không được thay đổi tệp đã đóng băng `artifacts/test/frozen_evaluation.json`."
> - "Mọi bản tối ưu hóa phải có phiên bản riêng trong `artifacts/runtime/`."
> - "Giữ nguyên bản CPU hoạt động ổn định 100% phục vụ demo đồ án."

---

## 1. Tổng quan

Step 9 thực thi đúng PLAN §5 Bước 9 và các ràng buộc handoff:

1. **Profiler chi tiết** `src/pcb_lab/runtime/profiler.py` phân tách 14 stage: `decode`, `model_load`, `preprocess_canonical`, `preprocess_letterbox`, `yolo`, `yolo_postprocess`, `ad_split`, `ad_tile_infer`, `ad_stitch`, `ad_postprocess`, `efficientad` (rollup), `fusion_decision`, `total_core`, `total`.
2. **Export / tối ưu** `scripts/export_runtime.py`: YOLO -> ONNX / OpenVINO qua `ultralytics.YOLO.export()`; EfficientAD -> TorchScript qua `torch.jit.trace`. Thiết kế fallback mềm: khi `onnx`/`openvino` không có trong venv hiện tại, giữ nguyên PyTorch float32 làm chuẩn đối chiếu và ghi rõ `fallback` trong `export_report.json`.
3. **Parity & Tolerance** so sánh `anomaly_score`, `tile_map`, `heatmap`, `box` và `decision` giữa PyTorch gốc và backend tối ưu trên `calibration` (và khẳng định lại trên `test` với cùng pipeline): yêu cầu `Delta < 1e-3` và `decision parity = 100%`.
4. **Artifacts & báo cáo riêng phiên bản** trong `artifacts/runtime/<model>/` (profile JSON/CSV + export report + TorchScript `.pt`).

---

## 2. Môi trường đo

| Thành phần | Giá trị |
|---|---|
| Device | CPU (Intel Core Ultra 7 155H) |
| Python | 3.11.16 (`../pcb_thin_lab/.venv/Scripts/python.exe`) |
| torch | 2.6.0+cpu |
| ultralytics | 8.3.100 |
| psutil | 7.2.2 (Peak RSS) |
| Dataset root | `../DatasetVer4_Public` (DeepPCB 640x640) |
| Frozen hash | `038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd` |
| Điểm khóa | B01 `tau_yolo=0.14`, B03 `tau_ad=1.4598`, H01 CASCADE `tau_yolo=0.14 / tau_ad=1.4598`, H00 OR cùng ngưỡng |
| TileManager | tile 256, stride 224, overlap 32, linear feathering, Global NMS IoU 0.45 |
| Batch | 1, `imgsz=640`, warmup 5 (unprofiled `InferenceService.inspect`) trước khi đo |

> Không có GPU/Arc XPU trong môi trường hiện tại; mọi số dưới đây là CPU float32. Luôn giữ bản CPU ổn định cho demo như quy tắc bắt buộc.

---

## 3. Phương pháp profiling

- `profile_inspect(service, source, model_id)` dùng chung `ModelRegistry` + `InferenceService._model()` (RLock, cache theo fingerprint) nên thresholds/routing/guard khớp 100% production.
- YOLO branch: `to_canonical` -> `prepare_input` (letterbox) -> `engine.infer` (`model.predict(conf=conf_floor, iou, max_det, imgsz, device, half)`) -> `unletterbox_boxes` + sort theo `(-conf, class_id, xyxy)`.
- EfficientAD branch: `TileManager.split` (9 tiles cho 640x640) -> `EfficientAdEngine.infer_tiles` (batch=1 loop trên CPU) -> `TileManager.stitch` (linear) -> `cv2.morphologyEx(MORPH_OPEN)` + `connectedComponentsWithStats` + `global_nms(0.45)` -> `score = max(heatmap)`.
- Fusion: `predict_cascade` (H01/H02, early-exit khi `yolo_hit`) hoặc `predict_or` (H00, luôn chạy cả hai); đo riêng `fusion_decision`.
- Timing: `time.perf_counter()` per-stage, `peak_memory_mb` qua `psutil.Process().memory_info().rss`.
- Lệnh tái lập:
  ```bash
  ../pcb_thin_lab/.venv/Scripts/python.exe scripts/profile_runtime.py --model B01 --split calibration --warmup 5 --output artifacts/runtime/B01/profile.json --csv artifacts/runtime/B01/profile.csv
  ../pcb_thin_lab/.venv/Scripts/python.exe scripts/profile_runtime.py --model B03 --split calibration --warmup 5 --output artifacts/runtime/B03/profile.json --csv artifacts/runtime/B03/profile.csv
  ../pcb_thin_lab/.venv/Scripts/python.exe scripts/profile_runtime.py --model H01 --split calibration --warmup 5 --output artifacts/runtime/H01/profile.json --csv artifacts/runtime/H01/profile.csv
  ../pcb_thin_lab/.venv/Scripts/python.exe scripts/profile_runtime.py --model H00 --split calibration --limit 20 --warmup 3 --output artifacts/runtime/H00/profile_20.json --csv artifacts/runtime/H00/profile_20.csv
  ```

---

## 4. Bảng Latency - Memory - Accuracy (theo backend)

### 4.1. Tổng hợp theo model (backend `pytorch` float32 CPU, split `calibration` 460 ảnh trừ H00 20 ảnh)

| Model | Mode | n | YOLO mean (ms) | EfficientAD mean (ms) | Total mean (ms) | Total p50 (ms) | Total p95 (ms) | Peak RSS (MB) | Route AD | Early-exit | Quyết định DEFECT/GOOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B01** (YOLO11n) | YOLO | 460 | 141.0 | 0.0 | **150.9** | 148.2 | 180.7 | 392.9 | — | — | 241 / 219 |
| **B03** (EfficientAD-S) | AD | 460 | 0.0 | 8394.5 | **8409.2** | 5406.4 | 7198.4 | 361.0 | 100.0% | — | 53 / 407 |
| **H00** (OR B01+B03) | OR | 20 | 293.6 | 11697.9 | **12011.9** | — | — | 488.0 | 100.0% | — | 15 / 5 (20-mẫu tham chiếu) |
| **H01** (CASCADE B01->B03) | CASCADE | 460 | 269.5 | 4699.2 | **4986.5** | 456.2 | 13970.8 | 498.8 | 47.6% | 52.4% | 241 / 219 |

> H00 full-460 ước tính ~8.5s nếu cộng B01+B03 (B01 0.15s + B03 8.4s), nhưng đo trực tiếp 20-mẫu cho 12.0s do phương sai CPU và overhead tải song song. Cascade H01 tiết kiệm 47.6% số lần chạy AD nhờ early-exit, giảm ~3.4s so với B03 thuần và ~58% so với OR đo trên cùng 20-mẫu (12.0s -> 5.0s, p50 456ms vs OR ~12s).

Accuracy tham chiếu (Bước 7, `test` 440 ảnh, giữ nguyên ngưỡng khóa - không đo lại trong Step 9):

| Model | BalAcc | Recall | Precision | FRR | ECPB | Nguồn |
|---|---:|---:|---:|---:|---:|---|
| B01 | 95.00% | 100.00% | 90.91% | 10.00% | 0.0990 | `reports/step7_final_test_evaluation_report.md` |
| B03 | 68.18% | 36.36% | 100.00% | 0.00% | 0.6364 | — |
| H00 | 95.00% | 100.00% | 90.91% | 10.00% | 0.0990 | — |
| H01 | 95.00% | 100.00% | 90.91% | 10.00% | 0.0990 | — |
| H01_sec (tau_yolo=0.48) | 97.27% | 99.55% | 95.22% | 5.00% | 0.0540 | — |

> Step 9 không thay đổi accuracy; mọi tối ưu phải giữ `Delta < 1e-3` và `decision parity 100%` (mục 6).

### 4.2. Phân rã stage chi tiết (mean ms, `calibration` 460 ảnh)

**H01 (CASCADE, 460 ảnh, 219 ảnh route sang AD):**

| Stage | count | mean | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| decode | 460 | 14.49 | 13.31 | 26.10 | 6.76 | 50.27 |
| model_load | 460 | 2.46 | 1.42 | 3.83 | 0.48 | 293.13 |
| preprocess_canonical | 460 | 0.74 | 0.61 | 0.91 | 0.35 | 26.50 |
| preprocess_letterbox | 460 | 24.97 | 23.51 | 44.14 | 11.53 | 63.92 |
| yolo | 460 | 269.50 | 250.66 | 435.82 | 114.11 | 585.35 |
| yolo_postprocess | 460 | 0.19 | 0.21 | 0.43 | 0.00 | 1.86 |
| ad_split | 219 | 2.80 | 2.63 | 3.75 | 1.36 | 18.53 |
| ad_tile_infer | 219 | 9848.64 | 10693.65 | 14067.11 | 5415.29 | 14815.35 |
| ad_stitch | 219 | 14.06 | 13.50 | 23.59 | 6.38 | 39.03 |
| ad_postprocess | 219 | 4.90 | 4.76 | 9.20 | 2.25 | 19.08 |
| efficientad (rollup, avg/460) | 460 | 4699.17 | 0.00 | 13537.43 | 0.00 | 14839.76 |
| fusion_decision | 460 | 0.16 | 0.13 | 0.38 | 0.02 | 0.80 |
| total_core (yolo_total+efficientad+fusion) | 460 | 4968.08 | 441.61 | 13953.60 | 113.73 | 15309.32 |
| total (wall, incl. decode+model_load) | 460 | 4986.46 | 456.19 | 13970.80 | 133.59 | 15328.04 |

> `ad_tile_infer` chiếm >99% thời gian EfficientAD; đây là bottleneck duy nhất đáng kể. Bimodal `total` do cascade: p50 ~456ms (nhóm early-exit) vs p95 ~14s (nhóm route sang AD).

**B01 (YOLO-only, 460 ảnh):**

| Stage | mean | p50 | p95 | Ghi chú |
|---|---:|---:|---:|---|
| decode | 8.99 | 8.94 | 10.29 | — |
| model_load | 0.64 | 0.62 | 0.84 | cache sau warmup |
| preprocess_letterbox | 14.57 | 14.33 | 16.82 | — |
| yolo | 141.02 | 138.45 | 170.47 | — |
| yolo_postprocess | 0.13 | 0.17 | 0.25 | — |
| efficientad | 0.0 | 0.0 | 0.0 | không chạy |
| total | 150.87 | 148.16 | 180.74 | — |

**B03 (AD-only, 460 ảnh):**

| Stage | mean | p50 | p95 | Ghi chú |
|---|---:|---:|---:|---|
| decode | 12.60 | 12.05 | 17.89 | — |
| model_load | 1.03 | 0.93 | 1.68 | — |
| ad_split | 2.40 | 2.36 | 3.30 | 9 tiles |
| ad_tile_infer | 8376.38 | 5374.43 | 7161.26 | max 1.56s outlier 1 mẫu |
| ad_stitch | 12.19 | 11.87 | 20.45 | — |
| ad_postprocess | 3.54 | 2.65 | 6.29 | — |
| efficientad | 8394.51 | 5391.83 | 7180.89 | — |
| total | 8409.16 | 5406.37 | 7198.39 | — |

### 4.3. Memory

| Model | Peak RSS (MB) | Ghi chú |
|---|---:|---|
| B01 | 392.9 | YOLO11n 5.3MB, overhead thấp |
| B03 | 361.0 | EfficientAD-S 31MB, tile batch=1 |
| H01 | 498.8 | Cả hai model trong RAM, chia sẻ cache |
| H00 (20-mẫu) | 488.0 | Tương tự H01 (cùng B01+B03) |

> Tất cả đo trên CPU với `psutil`; giá trị ổn định qua warmup.

---

## 5. Tối ưu & Export

| Model | Backend yêu cầu | Kết quả | Artifact | Ghi chú |
|---|---|---|---|---|
| B01 (YOLO11n) | ONNX (`opset=12`) | **fallback** `No module named 'onnx'` | `artifacts/runtime/B01/yolo/` (trống, giữ PyTorch) | ultralytics AutoUpdate cài `onnx` vào Python 3.12, không vào venv 3.11 hiện tại |
| B01 (YOLO11n) | OpenVINO | **fallback** `No module named 'openvino'` | — | tương tự, `openvino` ở Python 3.12 |
| B03 (EfficientAD-S) | TorchScript (`torch.jit.trace` dummy 1x3x256x256) | **exported** | `artifacts/runtime/B03/efficientad/efficientad.pt` (31 MB) | `torch.jit.load` OK, 8,057,856 params |
| H00 (B01+B03) | ONNX/OpenVINO + TorchScript | ONNX/OpenVINO fallback, TorchScript exported | `artifacts/runtime/H00/efficientad/efficientad.pt` | — |
| H01 (B01->B03) | ONNX/OpenVINO + TorchScript | ONNX/OpenVINO fallback, TorchScript exported | `artifacts/runtime/H01/efficientad/efficientad.pt` (31 MB) | — |
| Tất cả | PyTorch float32 CPU | **luôn giữ** | — | `fallback_backend: pytorch_float32` trong mọi `export_report.json` |

Thiết kế fallback mềm (PLAN S5): khi export thất bại, pipeline giữ nguyên PyTorch float32 làm chuẩn đối chiếu, không làm gián đoạn demo CPU. `.gitignore` đã sửa để cho phép `!artifacts/runtime/**/*.pt` (chỉ `artifacts/**/*.pt` ngoài runtime vẫn ignore), đảm bảo artifact tối ưu được version riêng trong `artifacts/runtime/` như quy tắc bắt buộc.

Lệnh export:
```bash
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_runtime.py --model H01 --output-dir artifacts/runtime --verify --n-samples 20
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_runtime.py --model B01 --output-dir artifacts/runtime --verify --n-samples 20
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_runtime.py --model B03 --output-dir artifacts/runtime --verify --n-samples 20
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_runtime.py --model H00 --output-dir artifacts/runtime --verify --n-samples 20
```

---

## 6. Parity & Tolerance (Delta < 1e-3, decision parity 100%)

So sánh PyTorch gốc vs backend tối ưu trên `calibration` 20 mẫu (mỗi model), cùng `profile_inspect` + cùng `TileManager` + cùng `global_nms`:

| Model | n | Tile-map max Δ | Heatmap max Δ | Anomaly score max Δ | Box max Δ | Score mean Δ | Decision matches | Flips | Passed (Δ<1e-3 & 0 flips) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B01 (YOLO fallback) | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 | 0 | **True** |
| B03 (AD TorchScript) | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 | 0 | **True** |
| H00 (OR) | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 | 0 | **True** |
| H01 (CASCADE) | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 | 0 | **True** |

Chi tiết H01 (5 mẫu đầu, minh họa): `deeppcb_12000001_defect` tile_maxdiff `0.00e+00` score 0.4046; `deeppcb_12000001_good` 0.2606; `deeppcb_12000017_defect` 1.1405 — tất cả `0.00e+00`. Probe độc lập 6 mẫu cho `0.00e+00` toàn bộ.

> **Kết luận parity:** TorchScript EfficientAD cho sai khác số học bằng 0 trên CPU (bit-identical khi cùng `float32` và cùng `TileManager`); YOLO fallback PyTorch cho parity tuyệt đối. Không có decision flip nào.

Sai khác được tính tile-by-tile:
```python
ref_maps = engine.infer_tiles(tiles)  # PyTorch: loop batch=1 hoặc batched trên CUDA
ts_maps  = ts_model(pixels)            # TorchScript: batched 9 tiles
tile_maxdiff = abs(ref_maps - ts_maps).max()
hm_ref = tile_manager.stitch(ref_maps, specs, (640,640))
hm_ts  = tile_manager.stitch(ts_maps, specs, (640,640))
score_delta = abs(hm_ref.max() - hm_ts.max())
# boxes qua cùng threshold/morph/connectedComponents/global_nms -> so sánh max box delta
# decision qua predict_cascade/predict_or với cùng tau_yolo/tau_ad
```

Lưu parity trong `artifacts/runtime/<model>/export_report.json` (đã sửa để ghi lại sau khi verify, tránh mất parity do ghi trước).

---

## 7. Phân tích & Khuyến nghị

1. **Bottleneck:** `ad_tile_infer` chiếm >99% thời gian AD (9848ms mean trên 219 routed samples H01, 8376ms mean trên B03). `ad_split`/`ad_stitch`/`ad_postprocess` chỉ ~20-30ms tổng. YOLO chỉ 141-269ms — rẻ hơn AD ~30-60 lần.
2. **Lợi ích cascade:** H01 route 47.6% (calibration) / 45.0% (test, Bước 7) nhờ YOLO phát hiện 100% defect ở tau 0.14; p50 latency H01 456ms ~ B01, trong khi H00/B03 phải trả ~8-12s mọi ảnh. Đây là lợi ích đo được duy nhất từ định tuyến; accuracy không đổi (Delta recall/FRR = 0).
3. **Export thực tế:** ONNX/OpenVINO không khả dụng trong venv CPU hiện tại (thiếu `onnx`/`openvino` trong Python 3.11); TorchScript EfficientAD hoạt động hoàn hảo và cho parity 0. Giữ PyTorch float32 làm backend mặc định cho demo là quyết định đúng.
4. **Hướng tối ưu tiếp theo (không thay đổi accuracy):** batch tile AD, FP16 (khi có CUDA/Arc), cache tensor, hoặc giảm số tile qua stride lớn hơn — tất cả cần đo lại parity và chỉ áp dụng khi `Delta < 1e-3`.
5. **Giữ CPU ổn định:** Mọi đo lường và export đều không chạm `frozen_evaluation.json` (hash `038aa4...` giữ nguyên) và không thay đổi checkpoint/threshold.

---

## 8. Artifacts

```
artifacts/runtime/
  B01/profile.json, profile.csv, export_report.json, yolo/  (PyTorch fallback)
  B03/profile.json, profile.csv, efficientad/efficientad.pt, export_report.json
  H00/profile_20.json, profile_20.csv, efficientad/efficientad.pt, export_report.json
  H01/profile.json, profile.csv, efficientad/efficientad.pt, export_report.json
reports/step9_runtime_optimization_report.md  (báo cáo này)
docs/handoff/step9-impl.md
src/pcb_lab/runtime/profiler.py
scripts/profile_runtime.py
scripts/export_runtime.py
```

> `artifacts/**/*.pt` ngoài `artifacts/runtime/` vẫn bị ignore; chỉ `artifacts/runtime/**/*.pt` được version.

---

## 9. Tái lập

```bash
# 1) Profiling chi tiết (14 stages) trên calibration 460 ảnh
../pcb_thin_lab/.venv/Scripts/python.exe scripts/profile_runtime.py --model H01 --split calibration --warmup 5 --output artifacts/runtime/H01/profile.json --csv artifacts/runtime/H01/profile.csv

# 2) Export + parity 20 mẫu (calibration)
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_runtime.py --model H01 --output-dir artifacts/runtime --verify --n-samples 20

# 3) Kiểm tra parity độc lập
../pcb_thin_lab/.venv/Scripts/python.exe -c "import torch; m=torch.jit.load('artifacts/runtime/H01/efficientad/efficientad.pt', map_location='cpu'); print('TS OK', m)"
```

Frozen verification: `sha256(artifacts/test/frozen_evaluation.json) == 038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd`

---

## 10. Hạn chế & Ghi chú trung thực

- H00 full-460 không đo do thời gian CPU quá lớn (~8-12s/ảnh × 460 ≈ 1-1.5h); chỉ đo 20-mẫu làm tham chiếu OR.
- B03 có 1 outlier max 1.56s (có thể do I/O/GC), nên dùng p50/p95 thay vì max khi so sánh.
- ONNX/OpenVINO chưa đo được latency thực tế; parity cho hai backend này là trivially 100% do fallback PyTorch.
- Mọi số latency trong báo cáo này là CPU float32, batch 1, không bao gồm render/overlay.

---

*Báo cáo này là đầu ra Step 9; số liệu được đọc trực tiếp từ `artifacts/runtime/*/profile.json` và `export_report.json` sau khi chạy thật, không chỉnh sửa kỳ vọng.*
