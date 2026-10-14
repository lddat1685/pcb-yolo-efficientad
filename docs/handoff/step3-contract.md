\# Hợp đồng Step 3 = Bước 3 của PLAN (Baseline YOLO: B01 YOLO11n, B02 YOLO11s)



Quy tắc chung: docs/handoff/WORKFLOW.md. Mâu thuẫn với docs/plan.md thì plan thắng.



\## Phạm vi

\- Có: dựng view dữ liệu YOLO từ manifest; runner huấn luyện có cấu hình khoá; adapter suy luận; trích dự đoán thô

&#x20; (calibration, fusion); artifact + model card + báo cáo validation; smoke CPU.

\- Không: EfficientAD, cascade/OR, chọn tau\_yolo (Bước 5), evaluator/benchmark đầy đủ, UI, seed 43/44, export runtime.

\- Được thêm: torch, torchvision, ultralytics. Khoá phiên bản trong requirements/ theo môi trường (cpu-local, cuda-cloud).

&#x20; Tra API theo bản ultralytics đã cài (đọc docs/mã của chính bản đó), không dựa vào trí nhớ.

\- Tái dùng Step 1-2 (pcb\_lab.data.samples, pcb\_lab.inference.preprocessing, configs/preprocessing.yaml).



\## Cấu hình (giá trị null = CHƯA duyệt: trainer phải từ chối chạy)

configs/models/yolo\_common.yaml + yolo11n.yaml + yolo11s.yaml, hợp nhất khi nạp:



```yaml

model\_id: B01\_yolo11n              # B02\_yolo11s cho bản s

recipe\_id: yolo\_recipe\_v1

architecture: yolo11n              # yolo11n | yolo11s

pretrained: artifacts/pretrained/yolo11n.pt

train: {imgsz: 640, epochs: 100, patience: 20, optimizer: AdamW, lr0: null, batch: null, workers: null,

&#x20;       deterministic: true, augment: null}   # augment = tham số augmentation Ultralytics ("yolo\_aug\_v1")

selection: {split: calibration, metric: ultralytics\_fitness}

infer: {conf\_floor: 0.001, iou: 0.7, max\_det: 300, imgsz: 640, agnostic\_nms: false, half: false}

classes\_ref: benchmarks/deeppcb/configs/classes.json

```



\- B01 và B02 chỉ được khác nhau ở model\_id, architecture, pretrained (batch khác chỉ khi hết VRAM, phải ghi lý do).

\- config\_hash = sha256 của cấu hình đã hợp nhất và chuẩn hoá (không gồm seed). Seed qua CLI, mặc định 42.

\- Khoá lạ trong config -> ConfigError.



\## Giao diện (src/pcb\_lab/models/yolo/)



\### view.py

build\_yolo\_view(dataset\_root, out\_dir, partitions=("train","calibration"), link="hardlink", limit=None) -> YoloView

\- Chỉ cho train và calibration; "fusion" và "test" -> PermissionError. Không bao giờ ghi vào DATASET\_ROOT.

\- Sinh out\_dir/images/<partition>/ (hard link, hoặc copy nếu link="copy" hay khác ổ đĩa; ghi link\_mode),

&#x20; out\_dir/labels/<partition>/\*.txt từ Sample.boxes của manifest (định dạng YOLO: class cx cy w h, chuẩn hoá, 6 chữ số thập phân),

&#x20; ảnh good có nhãn rỗng, và data.yaml.

\- data.yaml: path/train/val (val = calibration), names theo đúng thứ tự classes.json (0..5); KHÔNG có khoá "test".

\- limit={"good": n, "defect": m}: chọn n good + m defect đầu tiên theo sample\_id (dùng cho smoke).

\- YoloView: data\_yaml, counts{partition: {images, good, defect}}, box\_counts{partition: {class\_name: n}}, link\_mode,

&#x20; view\_signature (sha256 trên các dòng sample\_id + image\_sha256 + sha256 nhãn, sắp xếp).

\- Tất định: hai lần dựng cho cùng view\_signature. Dựng lại vào out\_dir đã có -> dùng lại hoặc ghi đè có kiểm soát, không lẫn dữ liệu cũ.



\### train.py

train\_yolo(config\_path, seed=42, out\_root=".", device=None, smoke=False, resume=False, allow\_download=False) -> RunResult

\- Từ chối chạy (ConfigError) khi lr0, batch, workers hoặc augment còn null; từ chối ghi đè thư mục run có sẵn (trừ resume=True).

\- pretrained phải tồn tại (artifacts/pretrained/\*.pt) và ghi sha256; chỉ được tải khi allow\_download=True và phải ghi nguồn + phiên bản.

\- Ghi đủ tham số Ultralytics thực tế đã dùng (args\_used). Không truy cập mạng hay ghi file ngoài out\_root, DATASET\_ROOT chỉ đọc.

\- smoke=True: khởi tạo ngẫu nhiên từ file kiến trúc (không cần pretrained), epochs=1, view nhỏ (limit), device tuỳ chọn;

&#x20; run\_manifest.smoke=true; KHÔNG được tạo artifact chính thức trong artifacts/ (chỉ runs/smoke/...).

\- Cuối train: validate best.pt trên calibration bằng infer{}; ghi calibration\_per\_class.json; sao chép best.pt/last.pt vào artifacts/;

&#x20; ghi artifact.json và model\_card.md (tiếng Việt: mục đích, dữ liệu dùng, cấu hình, phiên bản, số liệu calibration, giới hạn N=2 nhóm,

&#x20; không dùng test; chỉ nêu họ model đã dùng, không đưa khẳng định về model khác).



\### adapter.py

\- Engine (giao thức): infer(pixels\_rgb\_uint8\_HWC) -> ndarray\[N,6] (x1,y1,x2,y2,conf,cls) theo tọa độ của pixels đầu vào.

&#x20; UltralyticsEngine nhận RGB, tự đổi sang kênh mà Ultralytics mong đợi; test có thể tiêm engine giả.

\- YoloDetector(engine, meta) và YoloDetector.from\_artifact(artifact\_dir, allow\_smoke=False):

&#x20; load(), predict(image\_rgb) -> list\[Detection{class\_id, class\_name, confidence, xyxy\_original}] sắp xếp confidence giảm dần

&#x20; (bằng nhau: theo class\_id rồi tọa độ), describe() -> dict (model\_id, checkpoint sha256, class\_order, infer params, input\_color="rgb").

\- predict dùng prepare\_input(image, "yolo") và unletterbox\_boxes của Step 2 (một nguồn logic duy nhất).

\- image\_score(dets) = confidence lớn nhất, 0.0 khi rỗng (plan §7.3).

\- from\_artifact: sha256 checkpoint khác artifact.json -> ArtifactMismatchError; thứ tự lớp của model khác classes.json -> ClassOrderError;

&#x20; artifact smoke khi allow\_smoke=False -> SmokeArtifactError. Không bao giờ nạp ngưỡng/cấu hình từ artifact khác.



\### extract.py

extract\_predictions(artifact\_dir, dataset\_root, partitions=("calibration","fusion"), out\_dir=None) -> dict

\- Partition "test" (hoặc ngoài calibration/fusion) -> PermissionError. Đọc mẫu qua ManifestDataset (Step 2).

\- Ghi preds\_<partition>.jsonl (mỗi ảnh đúng một dòng, sắp xếp sample\_id, tất định): run\_id, sample\_id, image\_sha256, split, source\_group,

&#x20; model\_id, checkpoint\_sha256, config\_hash, preprocessing\_version, inference\_params, detections\[...], yolo\_image\_score,

&#x20; error\_reason (null nếu ổn), timing\_ms=null (không dùng benchmark).

\- Ảnh lỗi khi suy luận: ghi error\_reason, KHÔNG bỏ dòng, KHÔNG thành GOOD/0 im lặng. Confidence >= conf\_floor.

\- CLI: scripts/extract\_yolo\_predictions.py --artifact <dir> \[--dataset-root <p>]



\## Artifact và schema

```text

runs/yolo/<model\_id>/seed<seed>/        train/ (đầu ra Ultralytics), run\_manifest.json, env.json, train.log

artifacts/yolo/<model\_id>/seed<seed>/   best.pt, last.pt, artifact.json, model\_card.md, calibration\_per\_class.json,

&#x20;                                       preds\_calibration.jsonl, preds\_fusion.jsonl

reports/yolo\_<model\_id>\_seed<seed>\_validation.md

```

run\_manifest.json (khoá bắt buộc, được thêm): schema\_version, run\_id, model\_id, architecture, seed, smoke, config\_hash, recipe\_id,

git\_commit, git\_dirty, ultralytics\_version, torch\_version, cuda\_version|null, device, python, dataset\_release\_sha256, view\_signature,

link\_mode, partitions\_used{train:n, val:"calibration", n}, pretrained{path, sha256, source}, args\_used{}, epochs\_run, best\_epoch,

early\_stopped, train\_time\_s, peak\_vram\_mb{value|null, method}, peak\_ram\_mb, checkpoints{best\_sha256, last\_sha256}, resumed\_from|null.

artifact.json: schema\_version, model\_id, seed, smoke, checkpoint\_best\_sha256, checkpoint\_last\_sha256, class\_order\[6], config\_hash,

preprocessing\_version, preprocessing\_hash, inference\_params, view\_signature, run\_manifest{path, sha256}, created\_by\_git\_commit.

calibration\_per\_class.json: split, n\_images, infer params, overall{map50, map50\_95, precision, recall, pr\_definition, pr\_conf},

per\_class{<class\_name>: {n\_gt\_boxes, precision, recall, ap50, ap50\_95}}, source, ultralytics\_version.

(precision/recall của Ultralytics là tại confidence tối đa hoá F1, KHÔNG phải ngưỡng vận hành; phải ghi pr\_definition và pr\_conf.)



\## Bất biến bắt buộc

1\. View: train 1.792 ảnh (895 good + 897 defect), calibration 460 (230/230); không có ảnh fusion/test (kiểm bằng sha256);

&#x20;  tổng box theo lớp của train khớp plan §1.3 (1.216/973/1.228/981/884/865); nhãn round-trip sai số <= 0.01 px; không ghi vào DATASET\_ROOT.

2\. data.yaml không có khoá test; names đúng thứ tự classes.json.

3\. Trainer từ chối null; smoke không tạo artifact chính thức; không ghi đè run cũ.

4\. Adapter nhận RGB, đúng kênh màu, box trả về đúng tọa độ ảnh gốc, kiểm lớp/hash/smoke như trên.

5\. Extraction: đủ 460 và 306 dòng, mỗi ảnh một dòng, test bị chặn, lỗi không biến mất.

6\. Hash trong run\_manifest/artifact.json khớp file thật.



\## Hoàn thành (plan §5 Bước 3)

Checkpoint nạp lại được, class order đúng, không dùng test để chọn epoch hay ngưỡng, artifact đầy đủ cho B01 và B02.



\## Lệnh chạy thật (bạn chạy sau khi Giai đoạn B PASS; nội dung chi tiết nằm trong docs/RUN\_GPU.md do implementer viết)

python scripts/train\_yolo.py --config configs/models/yolo11n.yaml --seed 42 --device 0

python scripts/train\_yolo.py --config configs/models/yolo11s.yaml --seed 42 --device 0

python scripts/extract\_yolo\_predictions.py --artifact artifacts/yolo/B01\_yolo11n/seed42

python scripts/extract\_yolo\_predictions.py --artifact artifacts/yolo/B02\_yolo11s/seed42

