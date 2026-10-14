# Hợp đồng Step 4 = Bước 4 của PLAN (Baseline EfficientAD: B03 EfficientAD-S tile 256)

Quy tắc chung: docs/handoff/WORKFLOW.md. Mâu thuẫn với docs/plan.md thì plan thắng.

## Phạm vi
- Có: Runner huấn luyện EfficientAD-S (B03) có cấu hình khóa; mô hình chỉ học từ 895 ảnh good train; tiền xử lý và cắt tile 256x256 qua TileManager (stride 224, overlap 32, linear blending stitching); tính thống kê teacher từ train good; tính quantile chuẩn hóa map từ calibration good; adapter suy luận xuất stitched heatmap (640x640), image anomaly score (max heatmap), và bounding boxes (thuật toán Map -> threshold tau_pixel -> Morph Open(k=3) -> Connected Components -> Area > A_min -> Bounding Rect -> Global NMS IoU=0.45); trích dự đoán thô (calibration, fusion); artifact + model card + báo cáo calibration; smoke CPU.
- Không: YOLO, cascade routing / logic OR (Bước 5), tối ưu ngưỡng tau_ad (Bước 5), evaluator/benchmark đầy đủ (Bước 6), test partition evaluation (Bước 7), UI, seed 43/44, export runtime ONNX/TensorRT.
- Thư viện: PyTorch (torch, torchvision), anomalib hoặc standalone EfficientAD-S implementation. Khóa phiên bản trong requirements/step4.txt.
- Tái dùng Step 1-2: `pcb_lab.data.samples` (AdTrainSet, EvalSet), `pcb_lab.inference.preprocessing` (`prepare_input(image, "ad")`), `pcb_lab.inference.tiling` (`TileManager`).

## Cấu hình (giá trị null = CHƯA duyệt: trainer phải từ chối chạy)
configs/models/efficientad_s.yaml:

```yaml
model_id: B03_efficientad_s
recipe_id: efficientad_recipe_v1
architecture: efficientad_s           # EfficientAD-S (small)
model_size: small
pretrained_teacher: artifacts/pretrained/efficientad_teacher_s.pt  # hoặc torch hub / torchvision pretrained backbone
train:
  tile_size: 256
  stride: 224
  batch_size: 1                       # Native EfficientAD batch=1
  epochs: 70                          # Ngân sách tham khảo theo PLAN §5 Bước 4
  lr: null
  weight_decay: null
  optimizer: AdamW
  seed: 42
  deterministic: true
  device: null
normalization:
  teacher_stats_split: train          # Chỉ tính từ 895 train good (8055 tiles)
  map_norm_split: calibration         # Quantile chuẩn hóa map CHỈ tính từ calibration good
  quantile_min: 0.90
  quantile_max: 0.995
infer:
  tile_size: 256
  stride: 224
  stitch_method: linear               # Linear Blending (feathering) overlap 32px
  score_metric: max                   # Max của stitched anomaly map
  tau_pixel: 0.5                      # Ngưỡng binarize heatmap ban đầu (để sinh candidate boxes)
  morph_k: 3                          # Kernel size Morphological Opening (3x3)
  min_area: 20                        # A_min pixel area threshold
  nms_iou: 0.45                       # Global NMS IoU threshold
  box_class_name: anomaly_unclassified
```

- config_hash = sha256 của cấu hình đã hợp nhất và chuẩn hoá (không gồm seed). Seed qua CLI, mặc định 42.
- Khóa lạ trong config -> ConfigError.

## Giao diện (src/pcb_lab/models/efficientad/)

### dataset.py
EfficientAdDataset(dataset_root, split="train", limit=None)
- split="train": CHỈ tải ảnh good (AdTrainSet từ Step 2, đúng 895 ảnh good). Nếu gặp ảnh defect -> DataLeakError / ValueError.
- split="calibration" hoặc "fusion": dùng EvalSet (460 và 306 ảnh).
- split="test" -> PermissionError. Tuyệt đối không chạm vào test.
- Tiling: Mỗi ảnh được cắt thành 9 tiles kích thước 256x256 bằng TileManager. Trả về tensor [9, 3, 256, 256] hoặc iterator từng tile.

### train.py
train_efficientad(config_path, seed=42, out_root=".", device=None, smoke=False, resume=False, allow_download=False) -> RunResult
- Từ chối chạy (ConfigError) khi lr, weight_decay còn null; từ chối ghi đè thư mục run có sẵn (trừ resume=True).
- Dữ liệu train: 100% normal-only (895 good images = 8,055 tiles). Ghi log kiểm toán normal-only.
- Thống kê teacher: tính từ train good.
- Thống kê chuẩn hóa map: tính trên 230 good images của calibration. Tuyệt đối không dùng defect để chuẩn hóa.
- Tắt mọi cơ chế auto-thresholding của framework.
- smoke=True: epochs=1, limit={"good": 4}, out_root="runs/smoke", KHÔNG tạo artifact chính thức trong artifacts/.
- Cuối train: xuất model checkpoint chứa đầy đủ weights (teacher, student, autoencoder), normalization stats (mean, std, quantiles), và tiling parameters.
- Sao chép vào artifacts/efficientad/B03_efficientad_s/seed<seed>/: model.pt, artifact.json, model_card.md, calibration_stats.json.

### adapter.py
- Engine (giao thức): `infer_tiles(tiles_uint8_nhwc: np.ndarray) -> np.ndarray`
  Nhận mảng N tile RGB uint8 `[N, 256, 256, 3]` (canonical RGB uint8 HWC, N=9 đối với ảnh 640x640 qua TileManager).
  Trả về mảng N tile anomaly map float32 `[N, 256, 256]`.
  Test độc lập có thể tiêm engine giả (FakeEngine / MockEngine) chỉ cần hiện thực phương thức này để kiểm tra toàn bộ luồng số học, stitching, và heatmap->bbox mà không cần checkpoint thật.
- EfficientAdEngine: Triển khai Engine bọc mô hình PyTorch đã load weights (Teacher, Student, Autoencoder).
- Detection: `@dataclass(frozen=True) class Detection`: `class_id: int | None` (None cho AD), `class_name: str` ("anomaly_unclassified"), `confidence: float`, `xyxy_original: tuple[float, float, float, float]`.
- EfficientAdResult: `@dataclass(frozen=True) class EfficientAdResult`: `anomaly_score: float`, `anomaly_map: np.ndarray` (float32 [H,W]), `boxes: list[Detection]`.
- EfficientAdDetector(engine: Engine, meta: dict) và EfficientAdDetector.from_artifact(artifact_dir, allow_smoke=False):
  - predict(image_rgb_640x640) -> EfficientAdResult:
    1. Nhận ảnh RGB uint8 canonical [H, W, 3] (mặc định 640x640).
    2. Cắt tile bằng `TileManager(tile_size=256, stride=224, pad_mode="reflect", nms_iou=0.45)`.
    3. Gọi `engine.infer_tiles(tiles)` -> `tile_maps` [N, 256, 256] float32.
    4. Tái tạo map toàn ảnh bằng `TileManager.stitch(tile_maps, specs, (H, W))` (linear blending).
    5. `anomaly_score = float(np.max(stitched_heatmap))` (baseline image score).
    6. Trích xuất bounding boxes từ heatmap theo thuật toán R8 đã khóa:
       `Map -> binarize tại tau_pixel -> Morph Open(k=3) -> Connected Components -> Area > min_area -> Bounding Rect -> Global NMS (IoU=0.45)`.
       Confidence của mỗi box là giá trị pixel anomaly score lớn nhất bên trong connected component đó.
  - describe() -> dict (model_id, checkpoint_sha256, norm_stats, infer_params, input_color="rgb").
  - from_artifact: sha256 checkpoint khác artifact.json -> ArtifactMismatchError; artifact smoke khi allow_smoke=False -> SmokeArtifactError.

- normalization_stats (cấu trúc bắt buộc trong artifact.json và meta["normalization_stats"]):
  ```json
  {
    "source_split": "calibration",
    "n_good_images": 230,
    "quantile_min": 0.0,
    "quantile_max": 1.0,
    "mean": 0.0,
    "std": 1.0
  }
  ```
  (Trong đó `quantile_min`, `quantile_max`, `mean`, `std` là các số thực float được tính trên tập calibration good; detector giả có thể gán giá trị mặc định ví dụ 0.0, 1.0).

### extract.py
extract_efficientad_predictions(artifact_dir, dataset_root, partitions=("calibration","fusion"), out_dir=None) -> dict
- Partition "test" (hoặc ngoài calibration/fusion) -> PermissionError. Đọc mẫu qua ManifestDataset / EvalSet.
- Ghi preds_<partition>.jsonl (mỗi ảnh đúng một dòng, sắp xếp theo sample_id, tất định):
  `run_id`, `sample_id`, `image_sha256`, `split`, `source_group`, `model_id`, `checkpoint_sha256`, `config_hash`, `preprocessing_version`, `tile_params`, `anomaly_score`, `boxes`: list of `{xyxy, confidence, class_name: "anomaly_unclassified"}`, `error_reason` (null nếu ổn), `timing_ms` (null).
- Không bỏ sót dòng, không đổi lỗi thành GOOD / score 0 im lặng.
- CLI: scripts/extract_efficientad_predictions.py --artifact <dir> [--dataset-root <p>]

## Artifact và schema

```text
runs/efficientad/<model_id>/seed<seed>/        run_manifest.json, env.json, train.log, checkpoints
artifacts/efficientad/<model_id>/seed<seed>/   model.pt, artifact.json, model_card.md, calibration_stats.json,
                                               preds_calibration.jsonl, preds_fusion.jsonl
reports/efficientad_<model_id>_seed<seed>_validation.md
```

run_manifest.json (khoá bắt buộc): schema_version, run_id, model_id, architecture, seed, smoke, config_hash, recipe_id, git_commit, git_dirty, anomalib_version|null, torch_version, cuda_version|null, device, python, dataset_release_sha256, partitions_used{train: 895, val: "calibration", n_good: 230, n_defect: 230}, normal_only_verified: true, args_used{}, epochs_run, train_time_s, peak_vram_mb, peak_ram_mb, checkpoint_sha256.

artifact.json: schema_version, model_id, seed, smoke, checkpoint_sha256, config_hash, preprocessing_version, tile_params, normalization_stats, inference_params, run_manifest{path, sha256}, created_by_git_commit.

calibration_stats.json: split, n_good, n_defect, good_scores{min, max, mean, std, p50, p90, p95, p99}, defect_scores{min, max, mean, std, p50, p90, p95, p99}, source.

model_card.md: Tiếng Việt: mục đích mô hình, kiến trúc EfficientAD-S, dữ liệu huấn luyện (chỉ 895 good train, 8055 tiles, 0 defect), cơ chế tiling 256x256 linear blending, thống kê phân phối điểm trên calibration, giới hạn (không phân loại 6 lớp lỗi, N=2 nhóm nguồn), không rò rỉ test.

## Bất biến bắt buộc
1. Normal-only: Huấn luyện CHỈ dùng 895 ảnh good của split train (tương ứng 8.055 tiles 256x256). Bất kỳ ảnh defect nào lọt vào tập train đều kích hoạt DataLeakError ngay lập tức.
2. Khóa test: Mọi thao tác truy cập split "test" đều bị chặn và quăng PermissionError.
3. Tiling 256x256 native: Dùng đúng TileManager từ Step 2 (tile 256x256, stride 224, linear blending). Tuyệt đối không dùng tile 320x320 hoặc resize 256x256 làm luồng chính.
4. Trích xuất box từ heatmap: Triển khai đúng thuật toán đã khóa (R8): Map -> threshold tau_pixel -> Morph Open(k=3) -> Connected Components -> Area > min_area -> Bounding Rect -> Global NMS (IoU=0.45). Class name luôn là "anomaly_unclassified".
5. Tính nhất quán sau reload: Checkpoint save -> reload -> predict cho kết quả anomaly_score và anomaly_map sai số <= 1e-5.
6. Dự đoán xuất ra: Đủ chính xác 460 dòng cho calibration và 306 dòng cho fusion. Mỗi ảnh đúng 1 dòng. Mã hash checkpoint và config_hash khớp 100% với artifact.json.

## Hoàn thành (plan §5 Bước 4)
Xác minh được normal-only training, nguồn thống kê chuẩn hóa không chứa defect, reload checkpoint nhất quán, trích xuất đầy đủ preds_calibration.jsonl (460) và preds_fusion.jsonl (306).

## Lệnh chạy thật (sau khi Giai đoạn B PASS; tài liệu chạy GPU trong docs/RUN_GPU.md)
python scripts/train_efficientad.py --config configs/models/efficientad_s.yaml --seed 42 --device 0
python scripts/extract_efficientad_predictions.py --artifact artifacts/efficientad/B03_efficientad_s/seed42
