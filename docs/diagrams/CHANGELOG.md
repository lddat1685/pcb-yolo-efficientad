# CHANGELOG — sửa sơ đồ theo thẩm định độc lập

Ngày: 2026-09-28. Bản trước: commit `c36cf17`. Phạm vi: chỉ `docs/diagrams/`; không đổi nhánh, không merge. Mục 4′ và 7′ thay thế bản cũ; bổ sung mục 8. Tất cả tám mục đều được nguồn ủng hộ và đã áp dụng; **không có mục bị từ chối**.

Đã đọc nguồn trước khi chuẩn bị sửa: `PLAN.md` (đường dẫn thực, xem NOTES N01), `docs/DECISIONS.md`, các giao diện Step 2/3/4; đối chiếu TRIAGE. Phần chuẩn hóa ưu tiên contract Step 4. Không sử dụng `docs/archive/`.

## Chi tiết trước/sau

Các đoạn `diff` ghi chính xác dòng hoặc phần nhãn thay đổi. Số dòng “trước” theo c36cf17; “sau” theo bản hiện tại. Với đoạn chỉ là một bullet trong nhãn dài, giữ phần còn lại của dòng nguyên trạng.

### 1. Model tùy chọn qua CORE

**Lý do:** Model chạy inference qua core; evaluator nhận prediction chuẩn và GT, không nhận model.
**Nguồn đã xác nhận:** PLAN §4.2, §7.1, §8; DECISIONS R9/S3.

**PLAN_OVERVIEW.mmd — trước dòng 203; sau dòng 207.**

```diff
-  B04 -. "đối chứng khi đủ ngân sách" .-> EVALUATOR
+  B04 -. "khi đủ ngân sách" .-> CORE
```

**PLAN_OVERVIEW.mmd — trước dòng 204; sau dòng 208.**

```diff
-  B06 -. "mở rộng AD" .-> EVALUATOR
+  B06 -. "khi đủ ngân sách" .-> CORE
```

**PLAN_OVERVIEW.mmd — trước dòng 205; sau dòng 209.**

```diff
-  B07 -. "mở rộng AD" .-> EVALUATOR
+  B07 -. "khi đủ ngân sách" .-> CORE
```

### 2. ImageNette là đầu vào regularization

**Lý do:** ImageNette là dữ liệu regularization đầu vào khi triển khai cần; không là sản phẩm của train và không tính vào 895 PCB good.
**Nguồn đã xác nhận:** PLAN §4.2, §5 Bước 4; Step 4 contract: Phạm vi cho phép anomalib hoặc standalone.

**PLAN_OVERVIEW.mmd — trước dòng 170; sau dòng 173.**

```diff
-  ADTRAIN -. "chỉ nếu triển khai cần" .-> IMAGENETTE
+  IMAGENETTE -. "dữ liệu regularization, chỉ nếu triển khai cần" .-> ADTRAIN
```

### 3. Ablation resize-256 vào ADINFER

**Lý do:** Ablation resize-256 có luồng suy luận riêng; đường chính B03 vẫn native tile-256.
**Nguồn đã xác nhận:** PLAN §5 Bước 2, §9; DECISIONS R7; Step 2 preprocessing; Step 4 bất biến 3.

**PLAN_OVERVIEW.mmd — trước dòng 226; sau dòng 230.**

```diff
-  RESIZE -->|"tile vs resize"| ABLATE
+  RESIZE -->|"tile vs resize"| ABLATE
+  RESIZE -. "ablation resize-256, §9" .-> ADINFER
```

### 4. 4′ — Map quantiles sau fit

**Lý do:** Map quantiles từ model đã fit và 230 calibration good; stats đi vào artifact, không quay vào tối ưu weights. Teacher mean/std vẫn lấy train good.
**Nguồn đã xác nhận:** PLAN §5 Bước 4; Step 4 train.py, normalization_stats, Artifact và schema.

Contract Step 4, dòng 63–69: teacher mean/std từ train good; map quantiles từ 230 calibration good và được lưu cùng checkpoint cuối train. Config dòng 34–35 khóa mức quantile 0.90/0.995. Dòng 92–103, 124 đặt `normalization_stats` trong `artifact.json` và `meta["normalization_stats"]`; checkpoint `model.pt` cũng chứa stats. Dòng 126 định nghĩa `calibration_stats.json` là phân phối score good/defect, không thay thế normalization_stats. Không đổi teacher statistics sang calibration và không nối stats trở lại tối ưu trọng số.

**PLAN_OVERVIEW.mmd — trước dòng 160; sau dòng 161.**

```diff
-  CA -->|"CHỈ 230 good để chuẩn hóa"| NORMAUDIT
+  CA -->|"230 good"| NORMAUDIT
```

**PLAN_OVERVIEW.mmd — trước dòng 161; sau dòng 162.**

```diff
-  NORMAUDIT -->|"normalization sau fit, không dùng defect"| ADTRAIN
+  ADTRAIN -->|"checkpoint đã fit"| NORMAUDIT
+  NORMAUDIT -->|"normalization_stats"| B03
```

**PLAN_OVERVIEW.mmd — trước dòng 63; sau dòng 64.**

```diff
-Quantile map chỉ good; tau_ad quyết định dùng cả good/defect để calibration
+Map quantiles 0.90/0.995 tính SAU khi fit trên 230 calibration good; tau_ad dùng good/defect để calibration
```

**INVENTORY.md — trước dòng 198; sau dòng 198.**

```diff
-| I168 — Phân biệt chuẩn hóa và ngưỡng — Quantile map chỉ good; tau_ad quyết định dùng cả good/defect để calibration | §5 Bước 4–5; contract Step 4 |
+| I168 — Phân biệt chuẩn hóa và ngưỡng — Map quantiles 0.90/0.995 tính SAU khi fit trên 230 calibration good; tau_ad dùng good/defect để calibration | §5 Bước 4–5; contract Step 4 |
```

**COVERAGE.md — trước dòng 174; sau dòng 174.**

```diff
-| I168 — Phân biệt chuẩn hóa và ngưỡng — Quantile map chỉ good; tau_ad quyết định dùng cả good/defect để calibration | NORMAUDIT | có |
+| I168 — Phân biệt chuẩn hóa và ngưỡng — Map quantiles 0.90/0.995 tính SAU khi fit trên 230 calibration good; tau_ad dùng good/defect để calibration | NORMAUDIT | có |
```

### 5. Artifact B03 riêng

**Lý do:** Tách artifact B03 khỏi runner ADTRAIN; một nguồn checkpoint và normalization cho ADINFER/H00/H01/extraction.
**Nguồn đã xác nhận:** PLAN §5 Bước 4, §7.1; Step 4 train.py/adapter.py và artifact.json.

Di chuyển hai bullet artifact từ ADTRAIN sang B03 và cập nhật I152–I153; không bỏ mục kiểm kê nào. B03 là đầu mối duy nhất đưa checkpoint + normalization vào ADINFER. Nút mới dùng màu model/artifact, không phải biến thể tùy chọn.

**PLAN_OVERVIEW.mmd — trước dòng 34; sau dòng 34.**

```diff
-    ADTRAIN["<b>Bước 4 — B03 EfficientAD-S</b><br/><i>§5 Bước 4; §7.1; contract Step 4</i><br/>• Vào: chỉ 895 train good, 0 defect; DataLeakError nếu lẫn defect<br/>• Teacher pretrained + student + autoencoder; học sai khác đặc trưng và tái tạo<br/>• PyTorch; anomalib hoặc standalone; model_size=small, batch=1, 70 epoch tham khảo, AdamW<br/>• Teacher mean/std chỉ train good; map quantiles 0.90/0.995 chỉ 230 calibration good<br/>• Không chuẩn hóa ImageNet hai lần; tắt auto-threshold / auto-split / cập nhật ngưỡng tự động<br/>• Null lr/weight_decay/device chưa duyệt hoặc khóa lạ: từ chối; seed=42, deterministic=true<br/>• Ra: model.pt đầy đủ teacher/student/autoencoder, normalization_stats, tile/inference params<br/>• artifact.json, model_card.md, calibration_stats.json; phân phối good/defect và false positive<br/>• Fit → save → reload → predict: map/score sai số không quá 1e-5; ghi optimizer steps thực"]
+    ADTRAIN["<b>Bước 4 — B03 EfficientAD-S</b><br/><i>§5 Bước 4; §7.1; contract Step 4</i><br/>• Vào: chỉ 895 train good, 0 defect; DataLeakError nếu lẫn defect<br/>• Teacher pretrained + student + autoencoder; học sai khác đặc trưng và tái tạo<br/>• PyTorch; anomalib hoặc standalone; model_size=small, batch=1, 70 epoch tham khảo, AdamW<br/>• Teacher mean/std chỉ train good; map quantiles 0.90/0.995 chỉ 230 calibration good<br/>• Không chuẩn hóa ImageNet hai lần; tắt auto-threshold / auto-split / cập nhật ngưỡng tự động<br/>• Null lr/weight_decay/device chưa duyệt hoặc khóa lạ: từ chối; seed=42, deterministic=true<br/>• Fit → save → reload → predict: map/score sai số không quá 1e-5; ghi optimizer steps thực"]
+    B03["<b>B03 — EfficientAD-S (tile 256)</b><br/><i>§5 Bước 4; §7.1; contract Step 4</i><br/>• model.pt chứa weights teacher/student/autoencoder, normalization_stats và tile/inference params<br/>• artifact.json: normalization_stats; model_card.md; calibration_stats.json: phân phối score good/defect; báo cáo false positive"]
```

**PLAN_OVERVIEW.mmd — trước dòng 167; sau dòng 169.**

```diff
-  ADTRAIN -->|"checkpoint B03"| EXTRACT
+  ADTRAIN -->|"checkpoint đã fit"| B03
+  B03 -->|"checkpoint B03"| EXTRACT
```

**PLAN_OVERVIEW.mmd — trước dòng 173; sau dòng 176.**

```diff
-  ADTRAIN -->|"checkpoint B03 + normalization"| ADINFER
+  B03 -->|"checkpoint B03 + normalization"| ADINFER
```

**PLAN_OVERVIEW.mmd — trước dòng 175; sau dòng 178.**

```diff
-  ADTRAIN -->|"H00 dùng lại B03"| ORRUN
+  B03 -->|"H00 dùng lại B03"| ORRUN
```

**PLAN_OVERVIEW.mmd — trước dòng 177; sau dòng 180.**

```diff
-  ADTRAIN -->|"H01 cùng B03 với H00"| CASCADE
+  B03 -->|"H01 cùng B03 với H00"| CASCADE
```

**PLAN_OVERVIEW.mmd — trước dòng 286; sau dòng 291.**

```diff
-  class R8,YOLOTRAIN,B01,B02,ADTRAIN,EXTRACT,CORE,YINFER,ADINFER,YEXIT,ORRUN,DIAG,RESULT model;
+  class R8,YOLOTRAIN,B01,B02,ADTRAIN,B03,EXTRACT,CORE,YINFER,ADINFER,YEXIT,ORRUN,DIAG,RESULT model;
```

**INVENTORY.md — trước dòng 182; sau dòng 182.**

```diff
-| I152 — Bước 4 — B03 EfficientAD-S — Ra: model.pt đầy đủ teacher/student/autoencoder, normalization_stats, tile/inference params | §5 Bước 4; §7.1; contract Step 4 |
+| I152 — B03 — EfficientAD-S (tile 256) — model.pt chứa weights teacher/student/autoencoder, normalization_stats và tile/inference params | §5 Bước 4; §7.1; contract Step 4 |
```

**INVENTORY.md — trước dòng 183; sau dòng 183.**

```diff
-| I153 — Bước 4 — B03 EfficientAD-S — artifact.json, model_card.md, calibration_stats.json; phân phối good/defect và false positive | §5 Bước 4; §7.1; contract Step 4 |
+| I153 — B03 — EfficientAD-S (tile 256) — artifact.json: normalization_stats; model_card.md; calibration_stats.json: phân phối score good/defect; báo cáo false positive | §5 Bước 4; §7.1; contract Step 4 |
```

**COVERAGE.md — trước dòng 158; sau dòng 158.**

```diff
-| I152 — Bước 4 — B03 EfficientAD-S — Ra: model.pt đầy đủ teacher/student/autoencoder, normalization_stats, tile/inference params | ADTRAIN | có |
+| I152 — B03 — EfficientAD-S (tile 256) — model.pt chứa weights teacher/student/autoencoder, normalization_stats và tile/inference params | B03 | có |
```

**COVERAGE.md — trước dòng 159; sau dòng 159.**

```diff
-| I153 — Bước 4 — B03 EfficientAD-S — artifact.json, model_card.md, calibration_stats.json; phân phối good/defect và false positive | ADTRAIN | có |
+| I153 — B03 — EfficientAD-S (tile 256) — artifact.json: normalization_stats; model_card.md; calibration_stats.json: phân phối score good/defect; báo cáo false positive | B03 | có |
```

**INVENTORY.md — trước dòng 486; sau dòng 486.**

```diff
-thuộc **98 nút nội dung**
+thuộc **99 nút nội dung**
```

**NOTES.md — trước dòng 127; sau dòng 127.**

```diff
-409 mục được gộp trong 98 nút nội dung theo làn.
+409 mục được gộp trong 99 nút nội dung theo làn.
```

### 6. AD route% của B04

**Lý do:** Bảng nguồn ghi AD route% của B04 là N/A.
**Nguồn đã xác nhận:** PLAN §7.2, hàng B04.

**PLAN_OVERVIEW.mmd — trước dòng 84; sau dòng 85.**

```diff
-E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A
+E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A, B04=N/A
```

**INVENTORY.md — trước dòng 281; sau dòng 281.**

```diff
-| I241 — Bảng tổng hợp và bảng phụ — E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A | §7.2; §7.8 |
+| I241 — Bảng tổng hợp và bảng phụ — E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A, B04=N/A | §7.2; §7.8 |
```

**COVERAGE.md — trước dòng 247; sau dòng 247.**

```diff
-| I241 — Bảng tổng hợp và bảng phụ — E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A | TABLE | có |
+| I241 — Bảng tổng hợp và bảng phụ — E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A, B04=N/A | TABLE | có |
```

### 7. 7′ — Score/timing null cho tầng không chạy

**Lý do:** Điểm và thời gian của tầng không chạy phải null, không phải 0; OR/diagnostic_both vẫn chạy hai tầng.
**Nguồn đã xác nhận:** PLAN §3.1, §3.4, §8; DECISIONS S1.

Tách cạnh efficientad_only khỏi or/diagnostic_both để quy tắc YOLO=null chỉ áp dụng khi YOLO không chạy. Không thay score=0 của một tầng đã chạy nhưng không có candidate bằng null.

**PLAN_OVERVIEW.mmd — trước dòng 188; sau dòng 192.**

```diff
-  YEXIT -->|"DEFECT; AD không chạy"| RESULT
+  YEXIT -->|"DEFECT; AD bị khóa: anomaly_score=null; timing_ms.efficientad=null"| RESULT
```

**PLAN_OVERVIEW.mmd — trước dòng 183; sau dòng 186.**

```diff
-  CORE -->|"efficientad_only / or / diagnostic_both: mọi ảnh"| ADINFER
+  CORE -->|"efficientad_only: yolo_image_score=null; timing_ms.yolo=null"| ADINFER
+  CORE -->|"or / diagnostic_both: mọi ảnh"| ADINFER
```

### 8. Test cuối và báo cáo mọi đường đi

**Lý do:** Ảnh test qua core thực hiện preprocessing; evaluator chỉ nhận nhãn/bbox GT và sample_id, không retune.
**Nguồn đã xác nhận:** PLAN §5 Bước 7, §7.5, §8; Step 2 prepare_input.

Giữ TESTFINAL → CORE, không thêm TESTFINAL → PRE. Chỉ đổi nhãn hai cạnh được yêu cầu; không sửa đường gián tiếp tới SHORTLIST. Báo cáo đầy đủ tại `TESTFINAL_PATHS.md`: 56 đường đơn phân biệt theo cạnh (41 chuỗi nút), tất cả tới SHORTLIST; 0 đường tới YOLOTRAIN, ADTRAIN, NORMAUDIT, CALGRID, THRESH. Đồ thị có chu trình nên số lượt đi có lặp là vô hạn; báo cáo ghi rõ cách đếm, nút dùng chung và các chu trình liên quan.

**PLAN_OVERVIEW.mmd — trước dòng 236; sau dòng 241.**

```diff
-  TESTFINAL -->|"cùng evaluator; chỉ báo cáo cuối, không retune"| EVALUATOR
+  TESTFINAL -->|"chỉ nhãn/bbox GT và sample_id; chỉ báo cáo cuối, không retune"| EVALUATOR
```

**PLAN_OVERVIEW.mmd — trước dòng 235; sau dòng 240.**

```diff
-  TESTFINAL -->|"440 ảnh qua core với cấu hình đã khóa; không retune"| CORE
+  TESTFINAL -->|"440 ảnh qua core với cấu hình đã khóa; không retune; preprocessing chung (prepare_input) chạy trong core, §7.5"| CORE
```

**COVERAGE.md — trước dòng 426; sau dòng 426.**

```diff
-**đủ 409/409 mục**
+## Đối chiếu các nút/cạnh sau review
+
+| Mục | Nút/cạnh đã đối chiếu | Có/không | Kết quả |
+| --- | --- | --- | --- |
+| 1 | B04/B06/B07 → CORE; RESULT → EVALUATOR | có | Ba model tùy chọn đi nét đứt vào core, “khi đủ ngân sách”; prediction chuẩn đi vào evaluator. |
+| 2 | IMAGENETTE → ADTRAIN | có | Dữ liệu regularization đầu vào, nét đứt khi triển khai cần. |
+| 3 | RESIZE → ADINFER; TILE → ADINFER | có | Thêm cạnh ablation resize-256 nét đứt; đường chính B03 native tile-256 giữ nguyên. |
+| 4′ | ADTRAIN/CA → NORMAUDIT → B03; I168 | có | Quantiles 0.90/0.995 sau fit trên 230 calibration good; không còn NORMAUDIT → ADTRAIN. Teacher mean/std vẫn từ train good. |
+| 5 | ADTRAIN → B03 → ADINFER/EXTRACT/ORRUN/CASCADE; I152–I153 | có | B03 là artifact riêng và nguồn duy nhất đưa checkpoint + normalization vào ADINFER; artifact.json chứa normalization_stats, calibration_stats.json báo cáo phân phối score. |
+| 6 | TABLE; I241 | có | AD route% bổ sung B04=N/A. |
+| 7′ | YEXIT → RESULT; CORE → ADINFER | có | Tầng không chạy có score và timing=null; efficientad_only tách cạnh khỏi or/diagnostic_both. |
+| 8a–b | TESTFINAL → EVALUATOR/CORE | có | GT/sample_id chỉ sang evaluator; ảnh qua core với prepare_input; không thêm TESTFINAL → PRE. |
+| 8c | TESTFINAL tới sáu đích yêu cầu | có | 56 đường đơn phân biệt theo cạnh tới SHORTLIST (41 chuỗi nút); 0 đường tới năm đích khác. Chỉ báo cáo, chưa sửa liên thông qua node dùng chung. |
+
+Đường đi, nhãn cạnh và nút dùng chung được liệt kê đầy đủ tại [TESTFINAL_PATHS.md](TESTFINAL_PATHS.md). Độ bao phủ nội dung không thay thế kiểm tra liên thông: các đường đi cấu trúc tới SHORTLIST vẫn đang chờ người dùng quyết định.
+
+**đủ 409/409 mục**
```

## Các file kiểm chứng mới/cập nhật

| File | Trước | Sau | Lý do / nguồn |
| --- | --- | --- | --- |
| TESTFINAL_PATHS.md | Chưa có | Danh sách đủ 56 đường đơn, bảng 6 đích, ID/nhãn cạnh, nút dùng chung và chu trình | Yêu cầu 8c; PLAN §5 Bước 7, §7.5, §8; chỉ báo cáo, chờ quyết định |
| export/validation.json | 98 nút nội dung; 132 cạnh; SHA-256 bản c36cf17 | 99 nút nội dung; 136 cạnh; kiểm tra 409/409 và SHA-256 bản sửa; kết quả kiểm tra yêu cầu và đường đi | Đồng bộ kiểm chứng sau mục 1–8; không xem kiểm tra cấu trúc là parser Mermaid |
| CHANGELOG.md | Chưa có | Nhật ký trước/sau, căn cứ từng mục và các phần giữ nguyên | Yêu cầu ghi mọi thay đổi; § nguồn nằm ở từng mục phía trên |

`INVENTORY.md` chỉ đồng bộ I152–I153, I168, I241 và tổng số nút 98→99; `COVERAGE.md` giữ 409 mục, đổi ánh xạ artifact sang B03 và thêm bảng đối chiếu review. `NOTES.md` chỉ đổi tổng số nút 98→99; mọi ghi chú quyết định cũ giữ nguyên.

## Xác nhận alignment_jitter bằng Select-String

Đã chạy trước khi sửa:

```powershell
Select-String -LiteralPath PLAN.md -Pattern 'alignment_jitter' -Encoding UTF8
```

Kết quả khớp hai dòng:

```text
PLAN.md:14:> 7. **[v3-P2.1] Alignment Jitter:** Bổ sung `alignment_jitter` (±1–±2 px) vào Robustness Suite (§9).
PLAN.md:557:**[v3-P2.1] Alignment Jitter (lệch pha căn chỉnh quang học):** Bổ sung `alignment_jitter(dx ∈ [-2, 2], dy ∈ [-2, 2], dθ ∈ [-0.5°, 0.5°])` vào Robustness Suite — dịch chuyển tịnh tiến ±1 đến ±2 pixel và xoay vi phân. Mục đích: kiểm tra EfficientAD có nổ False Positive dọc theo mép đường mạch khi ảnh bị lệch nhẹ (dung sai template matching AOI 1–3 pixel). Đánh giá riêng trên EfficientAD-only (B03) và cascade (H01), không trộn vào benchmark chính.
```

Trước/sau: biên độ góc trong PLAN §9 và nút STRESS đều là ±0.5°, giữ nguyên toàn bộ dòng STRESS. Không có thay đổi alignment_jitter.

## Các phần giữ nguyên và kiểm chứng

- COST/COSTPENDING, ECPB và B05: giữ nguyên nội dung và mọi cạnh liên quan theo yêu cầu chờ quyết định. Chỉ thêm B04=N/A vào bullet AD route% của TABLE, không đổi bullet ECPB.
- Giữ nguyên nhánh `main`; không merge; không sửa PLAN, DECISIONS, TRIAGE, contracts, code hoặc tests.
- `mermaid-config.json` là file untracked có sẵn trước tác vụ; không sửa, không đưa vào commit.
- Một flowchart, 99 nút nội dung + 2 nút chú giải, 136 cạnh; mọi đầu cạnh có node, không trùng ID; 10 subgraph đóng đủ. Đối chiếu đủ 409/409 mục với nhãn node.
- Kiểm tra cạnh theo tám yêu cầu và đối chiếu độc lập danh sách đường đi bằng duyệt đồ thị. Các đường qua node dùng chung giữ nguyên để người dùng quyết định.
- **Chưa kiểm cú pháp bằng công cụ Mermaid:** lần xuất trước không tìm thấy `npx` (log tại `export/mmdc.log`); tác vụ sửa này không cài công cụ hay tạo SVG/PNG. Kiểm tra cấu trúc/nội dung không thay thế parser hoặc kiểm tra bố cục render.
