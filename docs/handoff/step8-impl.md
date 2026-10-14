# Step 8 — Implementer: Streamlit inspection UI

## Phase 0 — Kế hoạch trước triển khai (2026-09-29)

- Căn cứ: yêu cầu Bước 8 của người dùng, `PLAN.md` §5/§10 và `docs/handoff/WORKFLOW.md`. Chưa có `step8-contract.md`; yêu cầu trực tiếp xác định phạm vi triển khai, không sửa các hợp đồng trước.
- Workspace hiện tại đã ở branch `step8/impl`; dùng branch này theo phương án người dùng cho phép. Giữ nguyên các tệp không thuộc nhiệm vụ đang có trong workspace.
- Dataset chỉ đọc, gallery lấy sample ID từ manifest. Mặc định calibration/fusion. Gallery test chỉ xuất hiện trong chế độ báo cáo sau khi xác minh freeze và các báo cáo Bước 7; quyền xem test sau khóa có trong yêu cầu hiện tại và PLAN §5 Bước 8.
- Không train/tune, không ghi đè checkpoints, thresholds, benchmark hoặc dataset. Mức giảm latency H01/H00 tính từ CSV; không ép kết quả về 52.7%.

### Màn hình

1. Single Inspection: nguồn upload PNG/JPEG, gallery, webcam snapshot hoặc URL ảnh; chọn B01/B02/B03/H00/H01/H02 (và điểm H01_sec đã khóa nếu có). Hiển thị ảnh gốc, box theo sáu màu, heatmap khi AD thực sự chạy, bảng box/route/lý do/latency, provenance và tải JSON/overlay. Giữ kết quả khi thay tùy chọn hiển thị. Ảnh không nhãn không có accuracy.
2. Benchmark: chỉ đọc ba CSV đã khóa, metadata freeze, so sánh trực tiếp H00/H01, đồ thị Recall/FRR/ECPB/latency, phân rã lớp và nhóm nguồn. Hiển thị phạm vi N=2 nhóm, giới hạn số liệu và device/seed/split.
3. Batch: danh sách nhỏ upload hoặc gallery, progress, GOOD/DEFECT/ERROR, route rate, bảng từng ảnh, tải CSV/JSON. Không suy đoán ground truth từ filename.

### Kiến trúc service và caching

- `BaseImageSource.get_frame() -> (sample_id, RGB uint8 HWC)` chuẩn hóa I/O. Các nguồn file/upload/gallery/webcam/URL dùng cùng kiểm tra ảnh; UI chỉ gọi service/source, không tự mở đường dẫn ảnh.
- `InferenceService` dùng các adapter YOLO/EfficientAD hiện có và luật fusion đã kiểm thử. CLI single-image mới dùng đúng service như UI. Kết quả hợp nhất có decision, scores, boxes tọa độ gốc, heatmap, route, timings, provenance, ground truth khi thực sự có và lỗi rõ ràng (không chuyển lỗi thành GOOD).
- `@st.cache_resource` giữ adapter theo artifact/checkpoint fingerprint và device; H00/H01 chia sẻ B01+B03, H02 chia sẻ B02+B03. Khóa đồng bộ bảo vệ adapter dùng chung giữa các phiên. Nạp lười để benchmark không cần checkpoints.
- Cache metadata/report tách khỏi model. Session state giữ kết quả theo nội dung ảnh + model để thay overlay không chạy lại inference hoặc trình bày kết quả cũ dưới model mới.
- Ngưỡng lấy từ freeze Bước 7; MVP chỉ dùng các điểm đã đăng ký, không cung cấp thao tác thay ngưỡng.

### Tệp dự kiến

- Thêm `src/pcb_lab/inference/sources.py`, `service.py` và lõi đọc báo cáo/registry nếu cần.
- Thêm `src/pcb_lab/ui/{__init__,app,resources,visuals}.py`.
- Thêm `scripts/run_ui.py`, `scripts/inspect_image.py`.
- Khai báo phụ thuộc UI tùy chọn trong `pyproject.toml`; tài liệu khởi chạy.
- Test routing thật qua adapter giả lập có kiểm soát, nguồn ảnh/lỗi, caching/provenance, benchmark và Streamlit AppTest. Smoke CLI/UI trên checkpoint thật bằng Python được chỉ định.

### Điểm cần xác minh trong triển khai

- Python được chỉ định có torch/ultralytics/pytest, chưa có streamlit/pandas/altair: cần cài phụ thuộc UI để chạy kiểm thử thật.
- EfficientAD làm việc ở không gian tile; kiểm tra quy tắc chuẩn hóa kích thước và ánh xạ heatmap lên ảnh upload để giữ đúng hình học.
- Benchmark lịch sử có latency riêng; timing live không được coi là phép tái lập số latency đã khóa. So sánh CLI/UI yêu cầu trùng score/box/decision, loại trừ thời gian và run ID.

## Kết quả triển khai và kiểm thử

### Phạm vi đã triển khai

- `inference/sources.py`: BaseImageSource, upload/file/gallery/webcam snapshot/URL; PNG/JPEG → RGB uint8, EXIF orientation cho ảnh tự do, giới hạn dung lượng/kích thước. Gallery từ manifest, kiểm tra hash nội dung; không suy đoán nhãn từ tên file.
- `inference/registry.py`: đọc freeze Bước 7 và đăng ký B01/B02/B03/H00/H01/H02/H01_sec. Ngưỡng chỉ đọc; kiểm tra đường dẫn/hash/model identity. Gallery test yêu cầu report mode, quyền trong freeze, CSV hợp lệ và đúng hash tập sample ID.
- `inference/service.py`: dùng adapter hiện có và luật fusion Bước 5; H00 chạy cả hai, H01/H02 early-exit thực, AD không nạp/chạy khi YOLO đã dương. Kết quả có GOOD/DEFECT/ERROR, score, box tọa độ gốc, heatmap, trace, timing, provenance và metadata nhãn. Lỗi không trở thành GOOD.
- `ui/resources.py`: `st.cache_resource` giữ model/service; H00/H01 chia sẻ B01+B03, H02 chia sẻ B02+B03. Cache key gồm fingerprint artifact/checkpoint và device. `RLock` bảo vệ model dùng chung. `st.cache_data` chỉ giữ metadata/report; ảnh upload và kết quả nằm trong session.
- `ui/app.py`, `ui/visuals.py`: ba màn hình; box sáu màu, heatmap thật, tùy chọn overlay/opacity/GT, zoom box, bảng route/timing/score, JSON/PNG; benchmark với Pareto và phân rã nhóm/lớp; batch tối đa 20 ảnh, progress, dòng lỗi, confusion matrix khi có nhãn, xem lại và CSV/JSON. Thay ảnh/model ẩn kết quả cũ; thay display không suy luận lại.
- `inference/reports.py`: đọc nguyên ba CSV, kiểm tra schema, model, số mẫu nhóm và số hữu hạn. So sánh H00/H01 và Pareto được tính từ số liệu nguồn.
- `scripts/run_ui.py`, `scripts/inspect_image.py`: runner localhost và CLI dùng cùng service. `scripts/verify_step8_parity.py`: kiểm tra checkpoint thật qua tiến trình CLI riêng và Streamlit AppTest. Hướng dẫn sử dụng ở `docs/step8-ui.md`.
- `pyproject.toml`: thêm extra `ui` khóa Streamlit 1.55.0 / Altair 6.0.0 / pandas 2.3.3. Không thay dependencies core hoặc artifact đã khóa.

### Môi trường đã chạy

Python theo yêu cầu: `.\.cache\step3-verifier-b\.venv\Scripts\python.exe`.

Phiên bản thực tế: Python **3.14.7**, torch **2.14.0+cpu**, ultralytics **8.4.161**, Streamlit **1.55.0**, pandas **2.3.3**, Altair **6.0.0**. Các phiên bản torch/ultralytics/numpy/Pillow có sẵn của môi trường này không bị thay bởi nhiệm vụ.

Lần gọi pip đầu báo `No module named pip`; chạy `python -m ensurepip` thành công. Cài UI trong sandbox bị WinError 10013 khi truy cập PyPI; đã chạy lại với quyền mạng được duyệt và cài thành công ba dependency UI. Không dùng kết quả lần cài lỗi làm bằng chứng thành công.

### Lệnh và kết quả thực tế

Các lệnh dưới dùng `$py` là Python nêu trên, `$git = (Get-Command git).Source`.

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Test Bước 8 cuối cùng | `& $py -m pytest tests/test_step8_ui.py -q -p no:cacheprovider --basetemp=.cache/step8-final-tests` | **19 passed**, 39.82s |
| Parity checkpoint thật | `& $py scripts/verify_step8_parity.py` | **10/10 ca PASS**, 15 trường bằng nhau tuyệt đối, gồm hash heatmap |
| Hồi quy toàn bộ (trước khi bổ sung test CLI thứ 19) | `& $py -m pytest tests -q -m 'not smoke and not artifacts' -p no:cacheprovider --basetemp=.cache/step8-regression` | **390 passed, 9 failed, 15 skipped, 10 deselected**, 214.46s; 9 lỗi có sẵn, xem đối chứng dưới |
| Đối chứng mã gốc | `& $git archive --format=zip --output=.cache/step8-baseline.zip HEAD src tests configs`, giải nén vào `.cache/step8-baseline`, chạy 4 file `test_yolo_adversarial.py`, `test_yolo_config.py`, `test_yolo_extract.py`, `test_yolo_view.py` từ bản giải nén | **45 passed, đúng 9 failed**, 75.67s; traceback trỏ vào `.cache/step8-baseline/src`, commit gốc `0c1c3a6` |
| Startup runner | `& $py scripts/run_ui.py --server.port=8518` | Streamlit khởi động thành công; `GET http://127.0.0.1:8518/_stcore/health` → **200 / ok**; đã dừng server sau smoke |
| Compile | `& $py -m compileall -q src/pcb_lab/inference src/pcb_lab/ui scripts/run_ui.py scripts/inspect_image.py scripts/verify_step8_parity.py` | **PASS** |
| Bảo toàn artifact | `& $git diff HEAD -- artifacts` | Không có thay đổi |

Test Bước 8 bao phủ so sánh bằng ngưỡng (`>=`), early-exit không gọi/nạp AD, OR vẫn gọi AD khi YOLO dương, cache chia sẻ và invalidation, checkpoint lệch hash, ảnh hỏng, NaN AD, lỗi model, batch tiếp tục sau ảnh lỗi, URL qua HTTP mock, map đúng kích thước, gallery test bị khóa, CSV/Pareto, CLI exit code và ba màn hình AppTest. AppTest xác nhận thay opacity không gọi model lại và không hiển thị kết quả model cũ.

Đã thử kiểm tra trình duyệt bằng Playwright nhưng runtime Node hiện tại không nạp được module (`default export` / `Cannot find module 'playwright'`); không tuyên bố đã QA bằng screenshot trình duyệt. Ba màn hình và các biểu đồ được chạy thực qua Streamlit AppTest. Webcam phần cứng và URL internet thật chưa chạy; nguồn snapshot và HTTP adapter có kiểm tra nội bộ.

### Bằng chứng CLI/UI trên checkpoint thật

Tất cả ca dùng gallery **calibration**, không điều chỉnh ngưỡng. Dưới đây là số được đọc trực tiếp từ kết quả; bỏ tiền tố `deeppcb_` cho gọn trong bảng.

| Model | Sample | Decision | YOLO score | AD score | Số box YOLO | Route |
|---|---|---|---:|---:|---:|---|
| B01 | 12000001_defect | DEFECT | 0.8993628025054932 | null | 20 | YOLO |
| B02 | 12000001_defect | DEFECT | 0.9105978608131409 | null | 21 | YOLO |
| B03 | 12000001_defect | GOOD | null | 0.404594749212265 | 0 | AD |
| H00 | 12000001_defect | DEFECT | 0.8993628025054932 | 0.404594749212265 | 20 | OR, cả hai tầng |
| H01 | 12000001_defect | DEFECT | 0.8993628025054932 | null | 20 | Early-exit |
| H02 | 12000001_defect | DEFECT | 0.9105978608131409 | null | 21 | Early-exit |
| H01_sec | 12000001_defect | DEFECT | 0.8993628025054932 | null | 16 | Early-exit |
| H01 | 12000001_good | DEFECT | 0.43486037850379944 | null | 5 | Early-exit, false reject được giữ nguyên |
| B03 | 12000052_defect | DEFECT | null | 1.9171160459518433 | 0 | AD-positive |
| H01 | 12000038_good | GOOD | 0.0012245114194229245 | 0.11631742119789124 | 0 | Routed sang AD |

So sánh bằng `==`, không tolerance: sample/model/mode/decision/reason/trace, cờ thực thi/early-exit, YOLO score, AD score, detections, anomaly regions, metadata/hash heatmap và error. Run ID, timestamp và latency không yêu cầu bằng nhau. Nhánh cascade AD-positive được kiểm thử có kiểm soát; không tìm thấy mẫu rescue trong metadata calibration tại ngưỡng chính, nên không tạo bằng chứng rescue giả.

### Số liệu benchmark và giới hạn

- H00 mean = **2853.383165903506 ms**, H01 mean = **1348.7899934015745 ms** → giảm **52.730158%**, hiển thị **52.7%**; delta Recall/FRR/ECPB đều 0 tại điểm chính.
- Không hard-code số mẫu/lớp/nhóm trong code để làm khớp dataset. Ba CSV hiện cho sáu lớp và hai nhóm nguồn; gallery test kiểm tra metadata **440** mẫu khớp freeze.
- Qua đọc `test_eval.py`, latency hybrid Bước 7 là tổng thời gian từng tầng, không phải phép đo E2E trực tiếp. UI ghi rõ điều này. Live timing tách decode, load/cache, YOLO, AD, core và tổng không gồm render.
- EfficientAD CPU trong service dùng adapter xử lý từng tile, Bước 7 dùng batch tile; không tuyên bố bit-for-bit so với score lịch sử hoặc tái lập latency Bước 7. CLI và UI hiện tại dùng đúng cùng adapter và đã khớp tuyệt đối.
- Pareto không tô đậm một kết luận định trước: B01 có chất lượng bằng H01 với latency thấp hơn; H01 thể hiện lợi ích early-exit so với H00. Không suy rộng số liệu N=2 nhóm nguồn, không báo pixel metrics hoặc accuracy trên ảnh upload.

### Các lỗi hồi quy có sẵn tại commit gốc

| Nhóm test | Số ca lỗi | Phát hiện trên mã gốc |
|---|---:|---|
| `test_yolo_adversarial.py` | 3 | Chưa từ chối thư mục run rỗng đã tồn tại; chấp nhận checkpoint có lớp thứ bảy; chưa chuyển resume vào trainer |
| `test_yolo_config.py` | 3 | Chưa từ chối unknown key trong train/selection/infer |
| `test_yolo_extract.py` | 2 | run_id output khác run_manifest của fixture |
| `test_yolo_view.py` | 1 | Rebuild view có limit còn ảnh cũ |

Không sửa mã/hợp đồng Bước 3 trong nhiệm vụ UI. Các lỗi này được tái hiện trên mã gốc, độc lập với Bước 8. **Bước 8: PASS; hồi quy toàn repository: PASS-WITH-CONDITIONS do 9 lỗi baseline nêu trên.**

### Bàn giao

- Branch: `step8/impl`, nền `0c1c3a6`.
- Commit theo yêu cầu: `[claude] step8: implement streamlit inspection ui and benchmark explorer`.
- Chạy demo: `& $py scripts/run_ui.py`; xem hướng dẫn `docs/step8-ui.md`.
- Giữ nguyên PLAN/DECISIONS/TRIAGE, hợp đồng, dataset, thresholds, checkpoint và reports Bước 7. Các tệp PowerPoint/script/sơ đồ không thuộc nhiệm vụ không được đưa vào commit.
