# Chạy thật Bước 3 (Baseline YOLO) trên GPU

Bước 3 huấn luyện YOLO11n (B01) và YOLO11s (B02) trên GPU. Môi trường local (worktree
`PCB_Lab_impl3`) chỉ là **CPU dev** để phát triển/smoke; huấn luyện nặng 100 epoch chạy
trên **GPU đám mây** (theo PLAN §5 Bước 0 / Gate 0). Tài liệu này ghi đúng lệnh cho hai
trường hợp: (a) CUDA local, (b) Colab/Kaggle.

> Quy tắc: dataset chỉ đọc. Ultralytics **chỉ** thấy view do `build_yolo_view` dựng
> (hard link + label + `data.yaml`); không bao giờ ghi vào `DATASET_ROOT`. Artifact và
> prediction được đưa về máy local để nghiệm thu.

## 0. Chuẩn bị dataset (cả hai môi trường)

Dataset PCB nằm ở `..\DatasetVer4_Public` (Windows local) hoặc được tải lên
thư mục làm việc trên cloud. Đặt biến `DATASET_ROOT` trỏ vào thư mục gốc chứa
`benchmarks/deeppcb/...`.

- Windows local: `$env:DATASET_ROOT = '..\DatasetVer4_Public'`
- Linux/Colab: `export DATASET_ROOT=/path/to/DatasetVer4_Public`
- Hoặc truyền `--dataset-root` khi chạy script.

## 1a. CUDA local (Windows/Linux, có GPU)

```powershell
# Tạo venv Python 3.14.7 và cài đúng requirements (CUDA wheels)
uv --cache-dir .cache/uv venv --python 3.14.7 .venv
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe `
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple `
    -r requirements/step3.txt
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe --no-build-isolation --no-deps -e .
```
> Với torch CUDA: đổi `--index-url` thành bản cu128 (hoặc cu126 tuỳ driver) thay vì
> `.../cpu`. Các gói còn lại giữ nguyên `requirements/step3.txt` (khóa phiên bản).

Tải pretrained YOLO11 (YOLO11n/YOLO11s) từ Ultralytics về `artifacts/pretrained/`:

```powershell
New-Item -ItemType Directory -Force artifacts/pretrained | Out-Null
# YOLO11n
Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt `
    -OutFile artifacts/pretrained/yolo11n.pt
Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt `
    -OutFile artifacts/pretrained/yolo11s.pt
```
> Ghi nguồn + phiên bản vào `run_manifest.pretrained.source`. Hash được tự động tính và
> ghi `pretrained.sha256`. Nếu không tải thủ công, runner từ chối chạy (phải dùng
> `--allow-download` và ghi rõ nguồn).

Huấn luyện (device 0 = GPU đầu tiên):

```powershell
$env:DATASET_ROOT = '..\DatasetVer4_Public'
python scripts/train_yolo.py --config configs/models/yolo11n.yaml --seed 42 --device 0
python scripts/train_yolo.py --config configs/models/yolo11s.yaml --seed 42 --device 0
python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B01_yolo11n/seed42
python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B02_yolo11s/seed42
```

## 1b. Colab / Kaggle (đám mây)

Tải repo lên hoặc `git clone` branch `step3/impl`, rồi:

```python
import os, subprocess, sys
# 1) Dataset: mount Drive hoặc upload thư mục DatasetVer4_Public, đặt DATASET_ROOT
os.environ["DATASET_ROOT"] = "/content/DatasetVer4_Public"
# 2) Cài requirements (CUDA đã có sẵn trên Colab; Kaggle có P100/V100)
subprocess.run(["pip", "install", "-r", "requirements/step3.txt"], check=True)
# 3) Tải pretrained
!mkdir -p artifacts/pretrained
!wget -O artifacts/pretrained/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
!wget -O artifacts/pretrained/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

Huấn luyện (dùng device mặc định = GPU 0 trên Colab/Kaggle):

```python
!python scripts/train_yolo.py --config configs/models/yolo11n.yaml --seed 42 --device 0
!python scripts/train_yolo.py --config configs/models/yolo11s.yaml --seed 42 --device 0
!python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B01_yolo11n/seed42
!python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B02_yolo11s/seed42
```

Trên GPU, runner tự bật AMP (đổi `amp=False` trong `train.py` overrides thành `True`
cho cloud, hoặc truyền qua biến môi trường `PCB_AMP=1`) và ghi `peak_vram_mb`.

## 2. Mang artifacts/preds về máy local để nghiệm thu

Chỉ mang về những file sau (không mang `runs/` hay dataset):

- `artifacts/yolo/B01_yolo11n/seed42/{best.pt,last.pt,artifact.json,model_card.md,
  calibration_per_class.json,preds_calibration.jsonl,preds_fusion.jsonl}`
- Tương tự cho `B02_yolo11s/seed42`.

Trên Colab: `from google.colab import files; files.download('artifacts/yolo/B01_yolo11n/seed42/best.pt')`
Hoặc nén rồi tải: `!zip -r yolo_artifacts.zip artifacts/`.

Đưa về máy local vào đúng đường dẫn `artifacts/yolo/...` trong worktree `PCB_Lab_impl3`
để VERIFIER nghiệm thu (adapter load lại, kiểm class order + hash, extract validation).

## 3. Tham số đã khoá (đọc từ configs/models/*.yaml)

- `imgsz=640`, `epochs=100`, `patience=20`, `optimizer=AdamW`, `lr0=0.001`, `batch=16`.
- Augmentation `yolo_aug_v1`: hình học thuần (xoay 90°×k + lật ngang), **tắt mọi**
  photometric/mosaic (ghi đè hsv/scale/mosaic=0 trong runner).
- `infer`: `conf_floor=0.001`, `iou(NMS)=0.7`, `max_det=300`, `agnostic_nms=False`, `half=False`.
- Selection: validation trên `calibration`; không dùng test chọn epoch/ngưỡng.

## 4. Lưu ý

- `workers` mặc định 8 trên GPU; giảm nếu thiếu RAM. Nếu B02 (yolo11s) hết VRAM, giảm
  `batch` và ghi lý do vào handoff (hợp đồng cho phép khác batch chỉ khi hết VRAM).
- Không commit checkpoint/preds/ảnh. `.gitignore` đã loại `artifacts/**/*.pt`,
  `artifacts/**/preds_*.jsonl`, `runs/`, `data_refs/yolo_view*`.
- Telemetry Ultralytics bị tắt bằng `ULTRALYTICS_NO_ANALYTICS=1` trong runner.

## 5. Bước 4 — Baseline EfficientAD-S (B03)

Chỉ chạy baseline 70 epoch sau khi VERIFIER Giai đoạn B PASS. Dùng branch có implementation
Step 4, Python 3.12 và requirements riêng `requirements/step4.txt`. Không cài chồng các
phiên bản torch của Step 3 vào môi trường này. CPU smoke đã được chạy thật trên Windows;
CUDA/Colab chưa được thực thi trong phiên implementation này.

Runner bật strict deterministic. Autoencoder dùng bilinear separable qua index_select trên
CUDA vì native CUDA bilinear backward của torch 2.6 không hỗ trợ chế độ này; phép nội suy
và gradient đã được đối chiếu số học trên CPU. Cần xác minh thêm trên GPU trong Giai đoạn B/Gate 0.

Recipe đã chốt với người dùng: AdamW, lr=0.0001, weight_decay=0.00001, batch 1,
70 epoch, seed 42. Standalone PDN-S teacher pretrained 384 kênh, student 768 kênh,
autoencoder; không ImageNette penalty, không augmentation, không auto-threshold.
Không khẳng định tái lập benchmark của bài báo. RGB chuẩn hóa ImageNet đúng một lần
bên trong model. Không thay tile bằng resize ảnh PCB.

### Môi trường Colab (CUDA)

Sau khi đưa code và dataset canonical lên Colab, chạy trong root repo:

```python
import os, subprocess, sys
os.environ["DATASET_ROOT"] = "/content/DatasetVer4_Public"
subprocess.run([sys.executable, "-m", "pip", "install",
    "torch==2.6.0", "torchvision==0.21.0",
    "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/step4.txt"], check=True)
import torch
assert torch.cuda.is_available(), "Cần bật GPU runtime trước khi train"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
```

CLI tự thêm `src` vào import path; không cần `pip install -e .` (pyproject có pins
của Step 1, khác môi trường ML Step 4).

```python
# Chỉ lần đầu cần --allow-download nếu chưa có teacher checkpoint.
subprocess.run([sys.executable, "scripts/train_efficientad.py",
    "--config", "configs/models/efficientad_s.yaml", "--seed", "42",
    "--device", "0", "--allow-download"], check=True)
subprocess.run([sys.executable, "scripts/extract_efficientad_predictions.py",
    "--artifact", "artifacts/efficientad/B03_efficientad_s/seed42",
    "--device", "0"], check=True)
```

Teacher tải từ [anomalib pretrained release](https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights).
Runner kiểm SHA256 archive
`c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a`
trước khi lấy `pretrained_teacher_small.pth`; checkpoint local lưu ở
`artifacts/pretrained/efficientad_teacher_s.pt`, hash/provenance vào run_manifest.
Không có weights và không bật `--allow-download` thì runner dừng rõ ràng.

Baseline đọc 895 good train (=8.055 tile), teacher stats chỉ từ train good;
quantile 0.90/0.995 chỉ từ 230 calibration good. Calibration 230 defect chỉ dùng báo cáo
phân phối sau khi khóa thống kê. Các số đếm lấy từ manifest, không hard-code trong runner.
70 epoch trên release này tương ứng 563.850 optimizer step; kiểm số thực tế trong manifest.
Không truy cập pixel test. Chưa chọn tau_ad; tau_pixel=0.5 chỉ phục vụ candidate boxes.

### Smoke CPU và resume

```powershell
# Trong venv Python 3.12 riêng; CPU wheels cho local.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/step4.txt
python scripts/train_efficientad.py --config configs/models/efficientad_s.yaml --device cpu --smoke --allow-download
python scripts/extract_efficientad_predictions.py --artifact runs/smoke/B03_efficientad_s/seed42/artifact --allow-smoke --limit 4
```

Smoke: 1 epoch, 4 good train/36 optimizer step, map norm trên 4 calibration good,
báo cáo score trên tối đa 4 good + 4 defect calibration. Output model/metadata/report
chỉ ở `runs/smoke/B03_efficientad_s/seed42/`; pretrained download là cache tài nguyên,
không phải artifact B03 chính thức. `--limit` chỉ cho extraction smoke. Không dùng smoke
để đánh giá chất lượng hoặc chọn ngưỡng.

Run đã có sẽ bị từ chối ghi đè. Thêm `--resume` để tiếp tục checkpoint `checkpoints/last.pt`
đã kết thúc một epoch; recipe/config hash, seed, smoke flag và dataset release phải khớp.
Resume khôi phục optimizer và RNG; nếu đã đủ epoch thì chỉ tái tạo calibration/export.
Không có checkpoint epoch hoàn chỉnh thì dùng một `--out-root` mới. Runner không tự giảm
budget, không tự chọn checkpoint theo defect calibration.

### Artifact cần mang về và kiểm tra

- `runs/efficientad/B03_efficientad_s/seed42/`: run_manifest.json, env.json,
  effective_config.json, normal_only_audit.json, train.log; giữ checkpoints/last.pt nếu cần resume.
- `artifacts/efficientad/B03_efficientad_s/seed42/`: model.pt, artifact.json,
  model_card.md, calibration_stats.json, preds_calibration.jsonl, preds_fusion.jsonl.
- `reports/efficientad_B03_efficientad_s_seed42_validation.md`.

```python
import json
from pathlib import Path
artifact = Path("artifacts/efficientad/B03_efficientad_s/seed42")
meta = json.loads((artifact / "artifact.json").read_text())
assert meta["smoke"] is False
for partition, expected in [("calibration", 460), ("fusion", 306)]:
    rows = [json.loads(line) for line in (artifact / f"preds_{partition}.jsonl").read_text().splitlines()]
    assert len(rows) == expected
    assert len({r["sample_id"] for r in rows}) == expected
    assert all(r["error_reason"] is None for r in rows)
    assert all(r["config_hash"] == meta["config_hash"] and
               r["checkpoint_sha256"] == meta["checkpoint_sha256"] for r in rows)
```

Extractor luôn giữ một dòng khi ảnh lỗi: anomaly_score=null, boxes=[], error_reason rõ ràng;
CLI trả mã lỗi 1 nếu có bất kỳ dòng lỗi. Không biến lỗi thành score=0/GOOD.
Không commit dataset, checkpoint hoặc prediction JSONL.
