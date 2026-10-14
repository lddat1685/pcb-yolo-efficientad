# PCB Hybrid YOLO + EfficientAD — Kiểm thử core AI cho ảnh PCB trần

## Hướng dẫn Chạy trên Máy Khác (Quickstart for New Machine)

Chuẩn bị **Python 3.11 hoặc 3.12, 64-bit**, mạng Internet để cài thư viện,
và bản bàn giao có đủ checkpoint trong `artifacts/` (clone Git đơn thuần không có
ba checkpoint gốc). Không sao chép `.venv`, `.cache` hoặc `scratch` sang máy mới.

1. **Windows:** nhấp đúp `setup_env.bat` để tạo `.venv`, cài bộ CPU đã khóa phiên bản
   và kiểm tra tính toàn vẹn 60 mục.
2. Nhấp đúp **`run_ui.bat`** để mở Web UI tại <http://localhost:8501>.
   Giữ cửa sổ launcher mở; nhấn `Ctrl+C` để dừng.

**Linux/macOS:** chạy `bash setup_env.sh`, sau đó `bash run_ui.sh`.
Linux cần gói hỗ trợ `venv` của bản phân phối. Bộ pin này dành cho Windows/Linux x86_64
và macOS Apple Silicon (Python ARM64); chưa kiểm chứng thực thi trên Linux/macOS.
Nếu trình duyệt không tự mở (SSH/server), truy cập địa chỉ trên từ máy chạy demo.

**Camera Dahua A3504MG100 (Ethernet/GigE):** trên máy đang nối camera, cài MV Viewer
và IMV SDK x64, chạy `run_camera_check.bat`, rồi chọn **Camera công nghiệp** trong UI.
Có chụp từng ảnh, software trigger, ROI và nhập ảnh đã lưu từ MV Viewer.
Xem [hướng dẫn cài đặt và kết nối camera](docs/CAMERA_DAHUA.md).
Kết nối trực tiếp hiện dành cho Windows x64; chưa xác nhận trên camera vật lý.

**Dataset:** đặt nội dung `DatasetVer4_Public` trong `data/`, hoặc đặt thư mục
`DatasetVer4_Public` bên cạnh repo. Vị trí khác: đặt biến `DATASET_ROOT` trước khi chạy.
Không có dataset vẫn dùng được Upload ảnh; Gallery và kiểm thử parity cần dataset đầy đủ.

```powershell
# Tùy chọn, chạy từ repo root trong PowerShell:
$env:DATASET_ROOT = (Resolve-Path ../DatasetVer4_Public).Path
.\run_ui.bat
```

Chạy `run_verify.bat` (Linux/macOS: `bash run_verify.sh`) để kiểm tra integrity,
toàn bộ `tests/` và parity CLI/UI. Log, JUnit XML và mã thoát từng bước nằm trong
`reports/portable_verification/<timestamp>/`; lỗi kiểm tra trả mã thoát khác 0.
Một số smoke test cần teacher/pretrained riêng và có thể skip khi chưa cung cấp.

Nếu `.venv` cũ dùng Python khác phiên bản, đổi tên hoặc chuyển nó ra khỏi repo trước
khi chạy setup. Có thể chỉ định trình thông dịch qua `PCB_PYTHON`; launcher không xóa
môi trường cũ. Đặt `PCB_NO_PAUSE=1` khi chạy batch tự động trên Windows.


Đồ án tốt nghiệp: hệ thống kiểm thử core AI phát hiện khuyết tật trên ảnh PCB trần
(bare PCB, chưa gắn linh kiện) bằng kiến trúc hybrid hai tầng
**YOLO11 → EfficientAD**, kèm giao diện kiểm thử Streamlit, CLI, benchmark thống nhất
và gói tái lập toàn bộ thí nghiệm.

> **Phạm vi đã khóa:** DeepPCB 2.998 ảnh 640×640, 6 lớp khuyết tật, protocol
> `deeppcb_cascade_holdout_v1` (train 1.792 / calibration 460 / fusion 306 /
> test 440). Mọi số liệu công bố truy ngược được về `artifacts/test/frozen_evaluation.json`.

---

## 1. Giới thiệu đề tài

### 1.1. Bài toán

Kiểm tra quang học tự động (AOI) cho bo mạch trần cần chặn ảnh lỗi (defect recall cao)
trong khi giữ tỷ lệ loại nhầm ảnh tốt (FRR) thấp và chi phí kỳ vọng mỗi bo (ECPB) thấp.
Đề tài xây dựng **core inference dùng chung** cho cả CLI, UI và benchmark, trả về quyết
định nhị phân `GOOD / DEFECT` (kèm `ERROR` khi lỗi hạ tầng), tuyến xử lý, score, box,
heatmap và thời gian từng tầng.

### 1.2. Kiến trúc Hybrid YOLO11 + EfficientAD

```
Upload / gallery ảnh ─▶ Kiểm tra & tiền xử lý ─▶ YOLO (6 lớp lỗi)
                                                    │
                          ┌─────────────────────────┴──────────────────────────┐
                          │ CASCADE H01 (chế độ sản phẩm chính)                 │
                          │  Có box ≥ τ_yolo → DEFECT, early-exit (không chạy AD)│
                          │  Không box → chạy EfficientAD tile-256 → so τ_ad     │
                          └─────────────────────────┬──────────────────────────┘
                          │ OR H00 (đối chứng kết hợp không điều kiện)           │
                          │  Chạy cả YOLO và AD mọi ảnh, quyết định = OR         │
                          └────────────────────────────────────────────────────┘
```

| ID | Phương pháp | Supervision | Vai trò |
|----|-------------|-------------|---------|
| B01 | YOLO11n | Good + defect bbox | Baseline nhanh, tầng một mặc định |
| B02 | YOLO11s | Good + defect bbox | Đối chứng dung lượng detector |
| B03 | EfficientAD-S (tile 256) | Good-only (895 good train) | Baseline anomaly, tầng hai mặc định |
| H00 | YOLO11n + EfficientAD-S, luật OR | Kế thừa B01 + B03 | Đối chứng không điều kiện (route 100%) |
| H01 | YOLO11n → EfficientAD-S, cascade | Kế thừa B01 + B03 | Hybrid chính (route ~45% test) |
| H02 | YOLO11s → EfficientAD-S, cascade | Kế thừa B02 + B03 | Đổi detector trong hybrid |
| H01_sec | H01 tại điểm vận hành phụ | Kế thừa B01 + B03 | τ_yolo = 0.48, FRR ≤ 1% |

**Ngưỡng đã khóa (frozen):** B01 `τ_yolo = 0.14` · B02 `τ_yolo = 0.38` ·
B03 `τ_ad = 1.4598` · H00/H01 `0.14 / 1.4598` · H02 `0.38 / 1.4598` ·
H01_sec `0.48 / 1.4598`. Chi tiết xem `docs/MODEL_CARD.md` và
`artifacts/test/frozen_evaluation.json`.

**Kết quả test cuối (440 ảnh, 220 good / 220 defect):**

| Model | BalAcc | Recall | Precision | FRR | ECPB | Latency mean |
|-------|-------:|-------:|----------:|----:|-----:|-------------:|
| B01 | 95.00% | 100.00% | 90.91% | 10.00% | 0.0990 | 120 ms |
| B03 | 68.18% | 36.36% | 100.00% | 0.00% | 0.6364 | 2733 ms |
| H00 (OR) | 95.00% | 100.00% | 90.91% | 10.00% | 0.0990 | 2853 ms |
| H01 (Cascade) | 95.00% | 100.00% | 90.91% | 10.00% | 0.0990 | 1349 ms |
| H01_sec | 97.27% | 99.55% | 95.22% | 5.00% | 0.0540 | 1423 ms |

H01 đạt recall tương đương H00 (Δ = 0) nhưng giảm ~1505 ms latency trung bình nhờ
early-exit 55% ảnh test (route 45%). EfficientAD-only không đủ thay detector 6 lớp.
Xem `reports/step7_final_test_evaluation_report.md`.

### 1.3. Cấu trúc repo

```
├── PLAN.md / plan.v2.md / plan.v3.md   # Kế hoạch (bất biến sau khi khóa)
├── README.md                            # Tài liệu này
├── docs/                                # MODEL_CARD, DATASET_CARD, LIMITATIONS, REPRODUCIBILITY, handoff, TRIAGE, DECISIONS
├── configs/                             # dataset.yaml, preprocessing.yaml, models/
├── src/pcb_lab/                         # data, models, inference (+TileManager), fusion, evaluation, runtime, ui
├── scripts/                             # audit/train/calibrate/benchmark/inspect/run_ui/profile/export/verify
├── tests/                               # Unit + contract + parity (Bước 1–9)
├── artifacts/                           # Checkpoint metadata, thresholds, benchmark CSV, runtime profiles
├── reports/                             # data_profile, step5/6/7/9 reports
└── exports/thesis/                      # Bảng LaTeX/Markdown xuất cho luận văn (do script sinh ra)
```

Dataset gốc **không** nằm trong repo. Repo chỉ đọc dataset qua `DATASET_ROOT`.

---

## 2. Cài đặt môi trường

### 2.1. Yêu cầu

- Python 3.11 hoặc 3.12, 64-bit (bộ demo CPU).
- `DATASET_ROOT` trỏ tới `DatasetVer4_Public` (ví dụ `..\DatasetVer4_Public`).
- GPU cloud (Colab/Kaggle, CUDA 12.x) **chỉ** cần cho train lại Bước 3–4. Mọi bước
  CLI / UI / EDA / audit / benchmark / profiling chạy được trên CPU local.

### 2.2. Cài đặt chuẩn (CPU, demo)

```powershell
.\setup_env.bat
```

`requirements.txt` và `pyproject.toml` khóa cùng bộ dependency demo CPU.
Launcher cài PyTorch 2.6.0 / torchvision 0.21.0 từ CPU index trên Windows/Linux;
macOS dùng wheel PyPI theo [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/#v260).
Các tệp `requirements/step*.txt` lưu môi trường thí nghiệm lịch sử, không cài chồng
lên môi trường demo. Phiên bản trực tiếp được khóa; dependency gián tiếp do pip giải quyết.

### 2.3. Cấu hình dataset

```powershell
$env:DATASET_ROOT = "..\DatasetVer4_Public"
# hoặc sửa configs/dataset.yaml cho phù hợp máy
```

Kiểm tra nhanh:

```powershell
.venv\Scripts\python.exe scripts/audit_dataset.py --help
.venv\Scripts\python.exe scripts/check_loaders.py
```

### 2.4. Checkpoint (trọng số)

Checkpoint `.pt` **không** version trong git (xem `.gitignore`), trừ TorchScript export
trong `artifacts/runtime/**/*.pt`. Để chạy inference cần có:

- `artifacts/yolo/B01_yolo11n/seed42/best.pt` (SHA-256 `c0cfe917…`)
- `artifacts/yolo/B02_yolo11s/seed42/best.pt` (SHA-256 `aa079248…`)
- `artifacts/efficientad/B03_efficientad_s/seed42/model.pt` (SHA-256 `9453bc2c…`)

Kiểm tra toàn vẹn:

```powershell
.venv\Scripts\python.exe scripts/verify_reproducibility.py
```

Nếu thiếu file, lấy từ bản bàn giao checkpoint (kèm metadata phù hợp quyền sử dụng)
hoặc train lại theo `docs/REPRODUCIBILITY.md` (cần GPU cloud).

---

## 3. Chạy UI Streamlit

```powershell
.venv\Scripts\python.exe scripts/run_ui.py
# Mở http://127.0.0.1:8501  (đổi cổng: scripts/run_ui.py --server.port=8518)
```

Màn hình:

- **Single Inspection:** upload PNG/JPEG, chọn gallery calibration/fusion, chụp webcam
  hoặc nhập URL ảnh. Bấm **Chạy kiểm tra**; xem tuyến xử lý
  ("YOLO: phát hiện 2 box → kết thúc" hoặc "YOLO: không box → EfficientAD"),
  box + loại lỗi, heatmap stitch 640×640 (nếu AD chạy), score/ngưỡng, thời gian từng
  tầng, tải JSON/PNG. Gallery test chỉ mở ở chế độ report sau khi khóa đánh giá.
- **Benchmark H00 vs H01:** bảng/đồ thị chỉ đọc từ CSV đã khóa; Pareto recall–FRR–ECPB–latency.
- **Batch:** 1–20 ảnh, thống kê GOOD/DEFECT/ERROR, route rate, CSV/JSON từng ảnh.

Lưu ý trung thực hiển thị: YOLO confidence là điểm tin cậy box, anomaly score là max
của heatmap gốc (không phải xác suất); ảnh upload không nhãn ghi rõ
"Chưa có nhãn chuẩn để đánh giá đúng/sai". Khi cascade early-exit, UI ghi
**EfficientAD: Không chạy — không có heatmap**.

Chi tiết UI xem `docs/step8-ui.md`.

---

## 4. Lệnh chạy CLI inspect

CLI và UI gọi cùng `InferenceService`, cho cùng kết quả trên cùng artifact:

```powershell
# Gallery calibration (mặc định, không cần report-mode)
.venv\Scripts\python.exe scripts/inspect_image.py --model H01 --sample-id deeppcb_12000038_good --split calibration

# File ảnh local
.venv\Scripts\python.exe scripts/inspect_image.py --model H00 --image "samples/pcb.png"

# URL ảnh
.venv\Scripts\python.exe scripts/inspect_image.py --model B03 --url "https://example.org/pcb.png"

# Gallery test (chỉ sau khi khóa, cần cờ report)
.venv\Scripts\python.exe scripts/inspect_image.py --model H01 --sample-id <id> --split test --report-mode
```

CLI trả JSON qua stdout, exit 0 khi GOOD/DEFECT, exit 2 khi ERROR. Log thư viện đi
stderr. Model mặc định `H01`; các giá trị hợp lệ: `B01 B02 B03 H00 H01 H02 H01_sec`.

---

## 5. Quy trình tái lập thí nghiệm

Tóm tắt; hướng dẫn từng bước kèm checksum ở `docs/REPRODUCIBILITY.md`.

| Bước | Lệnh | Ghi chú |
|------|------|---------|
| Audit + EDA | `python scripts/audit_dataset.py` | 18/18 file release, 2998 ảnh |
| Loader check | `python scripts/check_loaders.py` | train 1792 / calib 460 / fusion 306 / test 440 |
| Train YOLO (GPU cloud) | `python scripts/train_yolo.py --config configs/models/yolo11n.yaml` | Colab CUDA 12.x, seed 42 |
| Train EfficientAD (GPU cloud) | `python scripts/train_efficientad.py` | 70k steps, tile 256/stride 224 |
| Calibrate (2-D grid) | `python scripts/calibrate_fusion.py` | → `artifacts/fusion/thresholds.json` |
| Benchmark dev | `python scripts/benchmark_baseline.py` | → `artifacts/benchmark/*.csv` |
| Đánh giá test (khóa) | `python scripts/evaluate_test.py` | → `artifacts/test/*.csv` + frozen |
| Profiling runtime | `python scripts/profile_runtime.py --model H01 --split calibration` | → `artifacts/runtime/` |
| Export + parity | `python scripts/export_runtime.py --model H01 --verify --n-samples 20` | Δ < 1e-3, parity 100% |
| Verify toàn vẹn | `python scripts/verify_reproducibility.py` | Kiểm kê file và một số SHA-256 đã đăng ký; không suy luận lại |
| Xuất bảng luận văn | `python scripts/export_thesis_assets.py` | → `exports/thesis/` (.tex + .md) |
| Test suite | `pytest tests/ -q` | Bước 1–9 |

Thứ tự ưu tiên khi quỹ thời gian ngắn (PLAN §12):
**audit → YOLO11n → EfficientAD-S (tile-256) → OR (H00) → cascade (H01) →
evaluator → UI → ba seed/test**.

---

## 6. Tài liệu bàn giao

| Tài liệu | Nội dung |
|----------|----------|
| `docs/MODEL_CARD.md` | Thẻ mô hình B01/B02/B03/H00/H01/H01_sec: kiến trúc, tham số, SHA-256, ngưỡng khóa |
| `docs/DATASET_CARD.md` | Thẻ DeepPCB: nguồn gốc, 6 lớp lỗi, holdout canonical, 2 nhóm test |
| `docs/LIMITATIONS.md` | Giới hạn N=2, bare-PCB, domain shift, AD bị khóa khi YOLO dương |
| `docs/REPRODUCIBILITY.md` | Từng bước tái lập + checksum xác thực |
| `reports/step5_calibration_fusion_report.md` | Hiệu chỉnh ngưỡng 2-D grid |
| `reports/step6_baseline_benchmark_report.md` | Benchmark development + ablation |
| `reports/step7_final_test_evaluation_report.md` | Đánh giá test cuối (số liệu công bố) |
| `reports/step9_runtime_optimization_report.md` | Latency–Memory–Accuracy theo backend |
| `docs/handoff/step10-impl.md` | Tổng kết đóng gói Step 10 |
| `exports/thesis/` | Bảng LaTeX/Markdown cho luận văn (sinh tự động) |

---

## 7. Giới hạn quan trọng (đọc trước khi trích dẫn)

1. **Test chỉ có N=2 nhóm nguồn** (`group44000`, `group92000`). Bootstrap chỉ đo
   sampling variance nội bộ 2 bo, không đại diện domain shift sang bo mới.
2. **Phạm vi bare PCB không linh kiện.** Bo có linh kiện hoặc layout quá dị biệt
   bắt buộc fine-tune lại EfficientAD.
3. **Khi YOLO dương (dù đúng hay sai), EfficientAD bị khóa hoàn toàn** trên ảnh đó
   trong cascade — không cứu false positive, không tìm lỗi lạ thứ hai.
4. Không có mask chuẩn nên **không** báo pixel AUROC/Dice/IoU segmentation.
5. Chênh lệch < ~0.45 điểm phần trăm trên test (1 ảnh / 220) không đủ kết luận vượt trội.

Chi tiết ở `docs/LIMITATIONS.md` (PLAN §1.4, §3.3, §7.6).

---

## 8. Giấy phép dữ liệu & trọng số

- Dataset DeepPCB của tác giả gốc; bản local pin commit ghi trong dataset card.
  Dùng đúng mục đích nghiên cứu/đồ án, tuân thủ license nguồn.
- Pretrained phổ thông (YOLO11, PDN-S teacher) ghi rõ nguồn/phiên bản/hash; mỗi
  baseline vẫn train/fit trên partition cho phép, không nạp checkpoint thiếu chứng cứ
  split/schema.
- Checkpoint và metadata bàn giao kèm theo tuân thủ quyền sử dụng dữ liệu/trọng số;
  không phát tán ngoài phạm vi cho phép.
