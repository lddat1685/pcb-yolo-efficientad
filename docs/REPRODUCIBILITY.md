# Hướng dẫn tái lập benchmark (Reproducibility)

Các lệnh dưới đây mô tả cách tái lập từ repo này + dataset
`DatasetVer4_Public` + checkpoint đã khóa. Tài liệu này liệt kê từng bước, lệnh chạy,
file kiểm tra và checksum xác thực.

> Nguyên tắc: calibration chọn checkpoint/ngưỡng, fusion chọn cấu hình, test chỉ đánh
> giá cuối. Không sửa ngưỡng theo test rồi ghi đè kết quả.

## 0. Chuẩn bị

```powershell
cd <repo-root>
.\setup_env.bat
$env:DATASET_ROOT = "..\DatasetVer4_Public"
```

Python 3.11/3.12, 64-bit. Linux/macOS: `bash setup_env.sh`.
`requirements.txt` khóa bộ demo CPU; các requirements theo Step dành cho lịch sử
training riêng. Dataset mặc định: `data/` trong repo, rồi `../DatasetVer4_Public`.
Chạy `run_verify.bat` hoặc `bash run_verify.sh` để lưu log integrity, toàn bộ tests
và parity CLI/UI trong `reports/portable_verification/`.

Bước nặng GPU (train YOLO/EfficientAD, sweep) chạy trên cloud Colab/Kaggle CUDA 12.x
(Gate 0, PLAN §12); máy local chỉ CLI/UI/EDA/audit/benchmark/profiling.

## 1. Kiểm toán dữ liệu (Bước 1)

```powershell
.venv\Scripts\python.exe scripts/audit_dataset.py
.venv\Scripts\python.exe scripts/check_loaders.py
```

- Kỳ vọng: `reports/data_audit.json` → `"overall": "PASS"`, release 18/18 tệp khớp,
  tổng 2998 ảnh (train 1792 / calibration 460 / fusion 306 / test 440), rò rỉ 0 trên
  mọi cặp partition.
- Overlay minh họa: `reports/overlay_index.json` (60 ảnh train/calibration).

## 2. Train baseline (Bước 3–4, GPU cloud)

```powershell
# YOLO11n (B01) — seed 42, ~0.35 GPU-h
python scripts/train_yolo.py --config configs/models/yolo11n.yaml --seed 42
# YOLO11s (B02) — seed 42, ~0.47 GPU-h
python scripts/train_yolo.py --config configs/models/yolo11s.yaml --seed 42
# EfficientAD-S (B03) — 70k steps, ~1.30 GPU-h
python scripts/train_efficientad.py --config configs/models/efficientad_s.yaml --seed 42
```

- Tổng ngân sách đo được: 2.12 GPU-h (hybrids tái dùng, không train thêm).
- Kiểm tra checkpoint sau train:

| Artifact | SHA-256 kỳ vọng |
|----------|-----------------|
| `artifacts/yolo/B01_yolo11n/seed42/best.pt` | `c0cfe917a685a9d20014d4ece0c0d012f2a8a4d53c100190d7961a3d5cd21b05` |
| `artifacts/yolo/B02_yolo11s/seed42/best.pt` | `aa079248bd566b3adf8863e3b43dde93d3392307a3ef17107e4c126bf410936c` |
| `artifacts/efficientad/B03_efficientad_s/seed42/model.pt` | `9453bc2c740e81dfb8949df929ed001d60ebf8e5b0f6a18c95eddd9b3a2814a0` |
| Tập test canonical (440 ảnh) | `57eadf1235759b4a131cfb45a8a5b98ecee73022e6da05cabaf580468b42af24` |

## 3. Hiệu chỉnh ngưỡng (Bước 5, CPU < 60 s)

```powershell
python scripts/calibrate_fusion.py
```

- Grid 2-D 2050 điểm `(τ_yolo, τ_ad)` trên calibration; tiêu chí tối đa recall tại
  FRR ≤ 5% (phân tích phụ FRR ≤ 1% cho H01_sec).
- Kết quả: `artifacts/fusion/thresholds.json` (`τ_yolo=0.14`, `τ_ad=1.4598` cho H01),
  `artifacts/fusion/routing_policy.json`, `reports/step5_calibration_fusion_report.md`.
- Calibration H01: recall 100%, FRR 4.78%, route 47.6%.

## 4. Benchmark development + ablation (Bước 6)

```powershell
python scripts/benchmark_baseline.py
```

- Kết quả: `artifacts/benchmark/development_summary.csv`,
  `artifacts/benchmark/ablation_summary.csv` (A1 detector B01 vs B02, A2 H00 vs H01,
  A3 primary vs secondary), `artifacts/benchmark/shortlist.json`,
  `reports/step6_baseline_benchmark_report.md`.
- Fusion H01 vs H00: Δrecall = 0, ΔFRR = 0, early-exit tiết kiệm 69.3% lượt chạy AD.

## 5. Đánh giá test cuối (Bước 7, đã khóa)

```powershell
python scripts/evaluate_test.py
```

- Chạy cùng 440 ảnh test cho B01/B02/B03/H00/H01/H02/H01_sec; xuất
  `artifacts/test/benchmark_summary.csv`, `benchmark_per_group.csv`,
  `benchmark_per_class.csv`, `predictions_test.jsonl`, `frozen_evaluation.json`,
  `reports/step7_final_test_evaluation_report.md`.
- Xác thực file frozen (verbatim, không sửa):

```powershell
python -c "import hashlib; print(hashlib.sha256(open('artifacts/test/frozen_evaluation.json','rb').read()).hexdigest())"
# Kỳ vọng: 038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd
```

- Số liệu công bố (test): H01 BalAcc 95.00% / recall 100% / FRR 10.00% / ECPB 0.0990 /
  route 45.0%; H01_sec BalAcc 97.27% / FRR 5.00% / ECPB 0.0540.

## 6. UI + CLI (Bước 8)

```powershell
.venv\Scripts\python.exe scripts/run_ui.py   # http://127.0.0.1:8501
.venv\Scripts\python.exe scripts/inspect_image.py --model H01 --sample-id deeppcb_12000038_good --split calibration
pytest tests/test_step8_ui.py -q -p no:cacheprovider --basetemp=.cache/step8-pytest
python scripts/verify_step8_parity.py
```

Parity CLI↔UI: score/box/decision/routing/hash heatmap khớp (trừ run-id/timestamp/latency).

## 7. Runtime profiling + export (Bước 9)

```powershell
python scripts/profile_runtime.py --model H01 --split calibration --warmup 5 --output artifacts/runtime/H01/profile.json --csv artifacts/runtime/H01/profile.csv
python scripts/export_runtime.py --model H01 --output-dir artifacts/runtime --verify --n-samples 20
```

- Kỳ vọng: H01 calibration mean ~4986 ms (p50 ~456 ms early-exit / p95 ~14 s routed),
  peak RSS ~499 MB; parity TorchScript 20/20, max Δ = 0.0 (ngưỡng Δ < 1e-3).
- YOLO ONNX/OpenVINO fallback PyTorch float32 trong venv CPU (thiếu module) — giữ bản
  CPU ổn định cho demo. Chi tiết: `reports/step9_runtime_optimization_report.md`.

## 8. Đóng gói Step 10 (bước hiện tại)

```powershell
python scripts/verify_reproducibility.py        # kiểm kê file + một số SHA-256 đăng ký
python scripts/export_thesis_assets.py          # xuất bảng .tex + .md vào exports/thesis/
pytest tests/ -q -p no:cacheprovider            # toàn bộ test suite Bước 1–9
```

## 9. Bảng checksum xác thực nhanh

| # | File | SHA-256 / giá trị kỳ vọng |
|---|------|---------------------------|
| 1 | `artifacts/test/frozen_evaluation.json` (file) | `038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd` |
| 2 | test split canonical (trường trong frozen) | `57eadf12…468b42af24` (440 ảnh) |
| 3 | B01 `best.pt` | `c0cfe917…5cd21b05` |
| 4 | B02 `best.pt` | `aa079248…bf410936c` |
| 5 | B03 `model.pt` | `9453bc2c…9b3a2814a0` |
| 6 | B01 `config_hash` | `1361dcc44534ff23c52e97949271dab1aa755fe8a90e85874c962db2ac9482f1` |
| 7 | B02 `config_hash` | `6070bffbb4b71c71929fce974acfa0e81148dc12d42882a40f226e45df3afa7b` |
| 8 | B03 `config_hash` | `cea3b1ce160871586f36c2a4da0d086ff11d2e2ac4546b37a6637b1461c85d29` |
| 9 | preprocessing `prep_v1` | `69e65565a2402077b41380b2d742ec989d939bf2cf6f3c7dbd80182fe177f770` |
| 10 | dataset release | 18/18 tệp khớp (`data_audit.json`) |

`scripts/verify_reproducibility.py` kiểm kê các file bắt buộc, kiểm tra một số hash
checkpoint/frozen và đọc kết quả parity đã lưu. Script không kiểm mọi hash trong
bảng trên, không suy luận lại và không thay thế kiểm toán khoa học. File vắng do
`.gitignore` được báo `MISSING`. Kết quả parity lịch sử phải đọc cùng đính chính
kiểm toán; xem `reports/remediation_20260930/CLOSURE.md` cho bằng chứng sửa lỗi.

Môi trường cài mới từ các lệnh trên chưa được xác minh; phiên bản ML không được
pin đầy đủ. Không coi kiểm kê file thành công là chứng nhận clean installation.
