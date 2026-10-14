\# Hợp đồng Step 2 = Bước 2 của PLAN (Loader, TileManager và tiền xử lý dùng chung)



Chỉ chốt phần giao tiếp để hai agent làm song song. Mâu thuẫn với docs/plan.md thì plan thắng,

TRỪ các mục ghi "\[đính chính]". Chỉ người điều phối được sửa file này. Agent thấy sai/thiếu thì ghi

docs/handoff/questions-<agent>.md, không tự sửa.



\## Phạm vi

\- Có: Sample + loader theo manifest; tiền xử lý chuẩn (canonical RGB, letterbox cho YOLO, resize-256 và tile-256

&#x20; cho EfficientAD); TileManager (tile, stitch, ánh xạ box, global NMS); recipe augmentation (chỉ train);

&#x20; configs/preprocessing.yaml; script kiểm tra một batch mỗi loader.

\- Không: train/inference model, ngưỡng, benchmark, UI, protocol.yaml, Gate 0.

&#x20; Không torch/ultralytics/anomalib. Chỉ numpy, Pillow, pyyaml, stdlib (thêm thư viện khác phải ghi lý do vào handoff).

\- Tái dùng Step 1 (pcb\_lab.data.manifest); không parse lại manifest ở chỗ khác. Chỉ sửa code Step 1 khi bắt buộc và ghi vào handoff.

\- Giữ quy ước Step 1: dataset chỉ đọc; không glob; không hard-code số dataset trong code; không decode/xem ảnh test.



\## Cấu hình: configs/preprocessing.yaml (tên khoá cố định)



```yaml

preprocessing\_version: prep\_v1

canonical: {color: rgb, dtype: uint8, layout: HWC}

yolo: {imgsz: 640, pad\_value: 114}

ad:

&#x20; tile: {size: 256, stride: 224, pad\_mode: reflect, blend: linear, global\_nms\_iou: 0.45}

&#x20; resize: {size: 256, interpolation: bilinear}

partition\_alias: {development\_selection: fusion}

augment: {recipe\_id: aug\_v1}

```



\- pad\_mode ∈ {reflect, replicate, constant}. tile.size chỉ nhận 256 (khác thì ValueError; plan: không dùng 320).

\- preprocessing\_hash(cfg) -> str (sha256 của cấu hình đã chuẩn hoá) được ghi vào mọi meta.



\## Giao diện (src/pcb\_lab/...)



\### data/samples.py

\- Sample (frozen): sample\_id, source\_group, pair\_id, partition, image\_path, width, height, is\_defect,

&#x20; boxes\[Box(class\_id, class\_name, xyxy)], mask\_status, sha256; load\_image() -> np.ndarray canonical RGB uint8 HxWx3.

\- ManifestDataset(dataset\_root, partition, \*, allow\_test=False): len() lấy từ manifest, KHÔNG decode ảnh;

&#x20; \_\_getitem\_\_/\_\_iter\_\_ trả Sample. Partition "test" với allow\_test=False: len() và metadata được phép,

&#x20; còn load\_image() và mọi truy cập ảnh -> PermissionError.

&#x20; "development\_selection" là alias của "fusion" (cùng mẫu, cùng thứ tự).

\- YoloTrainSet(root): toàn bộ train (good + defect).

&#x20; AdTrainSet(root): chỉ ảnh good của train; thuộc tính tiles\_per\_image (lấy từ TileManager.plan, không hard-code),

&#x20; num\_tiles = số ảnh \* tiles\_per\_image; iter\_tiles() -> Iterator\[(Sample, TileSpec, np.ndarray tile)].

&#x20; EvalSet(root, partition, allow\_test=False): mỗi ảnh đúng một lần.

\- Thứ tự tất định (sort theo sample\_id); hai lần duyệt cho cùng thứ tự.



\### inference/preprocessing.py

\- to\_canonical(image\_or\_path) -> uint8 RGB HWC (L, LA, RGBA -> RGB; L nhân bản 3 kênh, không đảo kênh).

\- letterbox(img, target=640, pad\_value=114) -> (img, LetterboxMeta{scale, pad\_left, pad\_top, orig\_w, orig\_h});

&#x20; ảnh 640x640 -> scale 1.0, pad 0.

\- unletterbox\_boxes(boxes\_xyxy, meta) -> boxes trong tọa độ ảnh gốc, clip vào ảnh; round-trip sai số <= 1 px.

\- normalize(img, mean, std) -> (float32, meta) đặt meta.normalized=True; gọi lại trên dữ liệu đã normalized -> ValueError.

\- resize\_for\_ad(img, size=256) -> (img256, meta); upsample\_map(map2d, out\_hw) -> float32 (bilinear, không overshoot).

\- prepare\_input(image\_or\_path, mode, cfg=None) -> PreparedInput, mode ∈ {"yolo","ad\_tile","ad\_resize"}:

&#x20; PreparedInput{pixels (uint8), meta (dict JSON-được: preprocessing\_version, preprocessing\_hash, mode, orig\_hw,

&#x20; scale/pad hoặc tile\_specs, color\_order="rgb", normalized=false), sha256() -> str}.

&#x20; Với ad\_tile: pixels có shape \[N,256,256,3].



\### inference/tiling.py: TileManager(tile\_size=256, stride=224, pad\_mode="reflect", nms\_iou=0.45)

\- plan(h, w) -> list\[TileSpec{tile\_id,row,col,x0,y0,size}], thứ tự row-major.

&#x20; Ảnh 640x640 -> 9 tile, x0,y0 ∈ {0,224,448}, overlap 32 px.

&#x20; \[đính chính] Số tile mỗi chiều = ceil((H-256)/224)+1 nếu H > 256, và 1 nếu H <= 256

&#x20; (plan §4.4 viết (640-256)/224+1 = 3 là sai số học; kết quả 3 vẫn đúng).

&#x20; Đệm chỉ ở phải/dưới (canvas 704x704 cho ảnh 640x640); tọa độ ảnh gốc trùng tọa độ canvas.

\- split(img) -> (tiles\[N,256,256,3], specs). Phần không đệm phải bằng đúng pixel ảnh gốc.

\- stitch(tile\_maps\[N,256,256], specs, out\_hw) -> float32\[H,W]: trộn tuyến tính (feathering) theo khoảng cách tới tâm tile

&#x20; (plan §4.4), chuẩn hoá tổng trọng số = 1, không bao giờ chia cho 0; vùng đệm bị cắt bỏ.

\- boxes\_to\_global(per\_tile\_dets: list\[np.ndarray\[N,5]], specs, out\_hw) -> np.ndarray\[M,5]

&#x20; (x1,y1,x2,y2,score): cộng offset tile, clip vào ảnh, bỏ box rỗng sau clip, giữ nguyên score.

\- global\_nms(dets\[N,5], iou=0.45) -> np.ndarray\[K,5]: sắp xếp score giảm dần; loại box khi IoU > ngưỡng (nghiêm ngặt).



\### data/augment.py

\- Recipe aug\_v1 chỉ áp cho split "train"; áp cho split khác -> ValueError. Tất định theo seed.

&#x20; Biến đổi box nhất quán với ảnh; box vẫn nằm trong ảnh; không sửa mảng đầu vào tại chỗ.



\### scripts/

\- check\_loaders.py --dataset-root <p> --out reports/loader\_check.json: mỗi loader một batch:

&#x20; n\_samples, shape, dtype, min/max, color\_order, normalized, hash. Tất định, không timestamp.

\- prep\_preview.py --image <path> --mode <m> --out-json <path>: ghi meta + sha256 của PreparedInput;

&#x20; phải trùng với gọi thẳng prepare\_input (cùng một hàm dùng chung).



\## Hoàn thành (plan §5 Bước 2)

Cùng một ảnh qua API và qua CLI cho cùng đầu vào model và cùng tọa độ đầu ra.

