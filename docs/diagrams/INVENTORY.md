# Kiểm kê toàn bộ kế hoạch

Nguồn chính: [PLAN.md](../../PLAN.md), đã đọc từ §1 đến §13. `docs/plan.md` không tồn tại; xem [NOTES.md](NOTES.md), N01. Đã đọc toàn bộ [DECISIONS](../DECISIONS.md), [TRIAGE](../TRIAGE.md) và các hợp đồng hiện có: [Step 2](../handoff/step2-contract.md), [Step 3](../handoff/step3-contract.md), [Step 4](../handoff/step4-contract.md). Không dùng `docs/archive/`.

Mỗi hàng là một phần tử hoặc một tập trường có cùng ý nghĩa; tên nút nhóm đứng trước dấu `—`. ID `Ixxx` dùng đối chiếu COVERAGE. Các tham số có xung đột giữ trạng thái theo nguồn ưu tiên mà người dùng chỉ định, giải thích trong NOTES. Sơ đồ là thiết kế tổng thể, không là chứng nhận các bước đã triển khai.

## Dữ liệu và partition (§1)

| Tên phần tử | § nguồn |
| --- | --- |
| I001 — Nguồn dữ liệu canonical — DATASET_ROOT = ../DatasetVer4_Public; chỉ đọc | §1.1; §4.2–4.3 |
| I002 — Nguồn dữ liệu canonical — DeepPCB: 2.998 ảnh RGB/grayscale 640×640; 1.498 good + 1.500 defect; 11 source groups | §1.1; §4.2–4.3 |
| I003 — Nguồn dữ liệu canonical — README, DATASET_CARD, status.json; manifest samples.jsonl; classes.json; holdout.json; protocol.json | §1.1; §4.2–4.3 |
| I004 — Nguồn dữ liệu canonical — build_summary.json, verification.json, release.json; 18/18 tệp release đã kiểm SHA-256 trong plan | §1.1; §4.2–4.3 |
| I005 — Nguồn dữ liệu canonical — Không glob, không chia ngẫu nhiên theo ảnh; không tự ghép template; hash ảnh/pixel/group/pair không giao partition | §1.1; §4.2–4.3 |
| I006 — Nguồn dữ liệu canonical — Không sửa protocol/release nguồn; output, cache và YAML nằm trong dự án riêng | §1.1; §4.2–4.3 |
| I007 — Partition train — 1.792 ảnh = 895 good + 897 defect; 5 nhóm nguồn | §1.2; §5 Bước 2 |
| I008 — Partition train — YoloTrainSet: toàn bộ train; AdTrainSet: chỉ 895 good | §1.2; §5 Bước 2 |
| I009 — Partition train — Đọc manifest, sắp sample_id tất định; mọi tile của cùng ảnh giữ cùng partition | §1.2; §5 Bước 2 |
| I010 — Partition calibration — 460 ảnh = 230 good + 230 defect; 2 nhóm nguồn | §1.2; §5 Bước 3–5 |
| I011 — Partition calibration — Chọn checkpoint và ngưỡng; 230 good để chuẩn hóa map AD | §1.2; §5 Bước 3–5 |
| I012 — Partition calibration — Toàn bộ 230 defect đi qua AD độc lập để hiệu chỉnh tau_ad; không chỉ mẫu routed | §1.2; §5 Bước 3–5 |
| I013 — Partition fusion — 306 ảnh = 153 good + 153 defect; 2 nhóm nguồn | §1.2; §5 Bước 5–6 |
| I014 — Partition fusion — Alias development_selection; cùng mẫu và cùng thứ tự | §1.2; §5 Bước 5–6 |
| I015 — Partition fusion — So sánh, chọn cấu hình sau calibration; không fit logistic fusion head trong baseline | §1.2; §5 Bước 5–6 |
| I016 — Taxonomy sáu lớp — 0 open_circuit — đứt mạch; bbox train/final: 1.216/298 | §1.3 |
| I017 — Taxonomy sáu lớp — 1 short — chập mạch; bbox train/final: 973/215 | §1.3 |
| I018 — Taxonomy sáu lớp — 2 mouse_bite — khuyết mép; bbox train/final: 1.228/250 | §1.3 |
| I019 — Taxonomy sáu lớp — 3 spur — gai đồng; bbox train/final: 981/228 | §1.3 |
| I020 — Taxonomy sáu lớp — 4 spurious_copper — đồng thừa; bbox train/final: 884/194 | §1.3 |
| I021 — Taxonomy sáu lớp — 5 pin_hole — lỗ nhỏ; bbox train/final: 865/222 | §1.3 |
| I022 — Taxonomy sáu lớp — Đa box, đa lớp trong một ảnh; pin_hole không đồng nhất missing_hole PKU | §1.3 |
| I023 — Giới hạn dữ liệu và kết luận — Chỉ bbox, không mask chuẩn: pixel AUROC / Dice / IoU segmentation / AUPRO không được công bố | §1.4; §7.6; §11.2 |
| I024 — Giới hạn dữ liệu và kết luận — Good template đã làm sạch, defect có bổ sung thủ công; kiểm shortcut và dấu vết xử lý | §1.4; §7.6; §11.2 |
| I025 — Giới hạn dữ liệu và kết luận — source_group chưa xác minh là bo/design vật lý; đơn vị kết quả là ảnh/crop | §1.4; §7.6; §11.2 |
| I026 — Giới hạn dữ liệu và kết luận — N=2 nhóm cho mỗi tập ngoài train; dedicated line, không universal zero-shot AOI | §1.4; §7.6; §11.2 |
| I027 — Giới hạn dữ liệu và kết luận — Bootstrap không chứng minh domain shift; không LOGO-CV 11 groups | §1.4; §7.6; §11.2 |
| I028 — Giới hạn dữ liệu và kết luận — Hybrid chưa chứng minh tốt hơn YOLO đơn | §1.4; §7.6; §11.2 |

## Phạm vi, sản phẩm và giả thuyết (§2)

| Tên phần tử | § nguồn |
| --- | --- |
| I029 — Phạm vi và sản phẩm cần bàn giao — Core AI kiểm thử ảnh PCB trần có sẵn; không yêu cầu camera, robot hoặc phân loại vật lý | §2.1 |
| I030 — Phạm vi và sản phẩm cần bàn giao — Không phát triển lại kiến trúc ba nhánh; YOLO11n+AD-S là cấu hình xuất phát chưa xác minh tốt nhất | §2.1 |
| I031 — Phạm vi và sản phẩm cần bàn giao — Dataset adapter + báo cáo audit tái chạy; baseline checkpoint + cấu hình đầy đủ | §2.1 |
| I032 — Phạm vi và sản phẩm cần bàn giao — OR/cascade hai tầng + ngưỡng khóa; benchmark chung + prediction từng ảnh | §2.1 |
| I033 — Phạm vi và sản phẩm cần bàn giao — UI upload/bbox/heatmap/routing/batch/compare; gói chạy lại, hướng dẫn, báo cáo chọn model | §2.1 |
| I034 — Giả thuyết H1–H5 cần kiểm chứng — H1: AD cứu một phần defect YOLO không có box đạt ngưỡng | §2.2 |
| I035 — Giả thuyết H1–H5 cần kiểm chứng — H2: recall tăng đáng chi phí false reject và tầng hai | §2.2 |
| I036 — Giả thuyết H1–H5 cần kiểm chứng — H3: resolution/tiling ảnh hưởng lỗi nhỏ, phải đo | §2.2 |
| I037 — Giả thuyết H1–H5 cần kiểm chứng — H4: cân bằng trên nhóm nguồn chưa thấy và nhiều seed | §2.2 |
| I038 — Giả thuyết H1–H5 cần kiểm chứng — H5: H01 recall tương đương H00 nhưng latency và/hoặc FRR tốt hơn — mục tiêu chưa chứng minh | §2.2 |
| I039 — Giả thuyết H1–H5 cần kiểm chứng — Sáu lớp đã biết không chứng minh mọi lỗi mới; mở rộng cần protocol riêng | §2.2 |

## Cascade, OR và các chế độ (§3)

| Tên phần tử | § nguồn |
| --- | --- |
| I040 — Một core — InferenceService — Chọn mode: yolo_only / efficientad_only / or / cascade / diagnostic_both | §3.4; §4.2; §8 |
| I041 — Một core — InferenceService — Nạp artifact đã đăng ký; preprocessing và output thống nhất cho CLI, UI, benchmark | §3.4; §4.2; §8 |
| I042 — Một core — InferenceService — YOLO không có box đạt ngưỡng không đồng nghĩa ảnh không lỗi; box thấp lưu được để phân tích | §3.4; §4.2; §8 |
| I043 — YOLO inference — Engine.infer RGB uint8 HWC → N×6 (xyxy, confidence, class) | §3.1; §7.3; contract Step 3 |
| I044 — YOLO inference — YoloDetector.predict trả Detection tọa độ gốc, sort confidence/class_id/xyxy | §3.1; §7.3; contract Step 3 |
| I045 — YOLO inference — accepted_boxes: đúng schema và confidence ≥ tau_yolo | §3.1; §7.3; contract Step 3 |
| I046 — YOLO inference — yolo_image_score = max confidence trước ngưỡng; 0 khi không có candidate | §3.1; §7.3; contract Step 3 |
| I047 — EfficientAD inference — EfficientAdEngine.infer_tiles: RGB uint8 NHWC → float32 [N,256,256] | §3.1–3.3; §7.3; contract Step 4 |
| I048 — EfficientAD inference — Teacher / student / autoencoder + stats đã khóa; ghép toàn ảnh qua TileManager | §3.1–3.3; §7.3; contract Step 4 |
| I049 — EfficientAD inference — Không cần template tham chiếu; không chọn ROI bằng ground truth | §3.1–3.3; §7.3; contract Step 4 |
| I050 — EfficientAD inference — anomaly_score = max(stitched map); DEFECT nếu anomaly_score ≥ tau_ad | §3.1–3.3; §7.3; contract Step 4 |
| I051 — EfficientAD inference — Vùng bất thường: anomaly_unclassified, class_id=None; không phải lớp GT thứ bảy | §3.1–3.3; §7.3; contract Step 4 |
| I052 — H01 / H02 — Cascade — H01 = B01 + B03; H02 = B02 + B03, bắt buộc vòng sàng lọc | §3.1,3.3; §7.1 |
| I053 — H01 / H02 — Cascade — Cùng checkpoint/ngưỡng: giữ mọi ảnh YOLO đã báo lỗi; recall ảnh không chứng minh đủ mọi box | §3.1,3.3; §7.1 |
| I054 — H01 / H02 — Cascade — Chỉ gọi AD khi accepted_boxes rỗng; không sửa được false positive của YOLO | §3.1,3.3; §7.1 |
| I055 — YOLO dương → DEFECT — decision_source=YOLO; efficientad_executed=false | §3.1,3.3; DECISIONS S1 |
| I056 — YOLO dương → DEFECT — Kể cả FP: kết thúc cascade, KHÔNG gọi EfficientAD | §3.1,3.3; DECISIONS S1 |
| I057 — YOLO dương → DEFECT — AD bị khóa nên không cứu good bị báo nhầm, không tìm lỗi thứ hai bị sót | §3.1,3.3; DECISIONS S1 |
| I058 — H00 — OR chạy cả hai — Mỗi ảnh gọi cả YOLO và AD; cùng checkpoint B01+B03 như H01 | §3.2; §7.1 |
| I059 — H00 — OR chạy cả hai — DEFECT nếu YOLO dương HOẶC anomaly_score ≥ tau_ad; ngược lại GOOD | §3.2; §7.1 |
| I060 — H00 — OR chạy cả hai — decision_source=OR; AD route=100%; không early-exit | §3.2; §7.1 |
| I061 — H00 — OR chạy cả hai — Giá trị recall / FRR / latency là kết quả cần đo; H5 không là kết luận đã xác minh | §3.2; §7.1 |
| I062 — diagnostic_both — Chạy cả YOLO và AD mọi ảnh để phân tích bù trừ đa lỗi | §3.4; DECISIONS R5 |
| I063 — diagnostic_both — Không lấy latency diagnostic_both làm latency cascade | §3.4; DECISIONS R5 |
| I064 — Nghiên cứu AD ngoài bbox — Background-only / vùng ngoài box là optional hoặc ablation | §3.4; §9; DECISIONS R5; NOTES N06 |
| I065 — Nghiên cứu AD ngoài bbox — Không thay routing toàn ảnh của cascade cơ bản | §3.4; §9; DECISIONS R5; NOTES N06 |
| I066 — Quyết định từ AD — Score ≥ tau_ad: DEFECT, decision_source=EFFICIENTAD | §3.1; §3.3 |
| I067 — Quyết định từ AD — Score dưới tau_ad: GOOD — chưa phát hiện bất thường | §3.1; §3.3 |
| I068 — Kết quả và lỗi — GOOD / DEFECT cùng decision_source, routing_reason, score/ngưỡng và thời gian | §3.3; §8 |
| I069 — Kết quả và lỗi — Đọc ảnh hỏng, thiếu weights, timeout, inference lỗi: ERROR; không trả GOOD | §3.3; §8 |
| I070 — Kết quả và lỗi — Không dùng max/trung bình confidence YOLO với anomaly score thô làm score cascade | §3.3; §8 |
| I071 — REVIEW nếu được bật — Ảnh ngoài phạm vi có thể cảnh báo hoặc REVIEW; anomaly score không mặc nhiên đáng tin cho OOD | §3.3–3.4; §7.4; §10.1 |
| I072 — REVIEW nếu được bật — Coverage, review rate riêng good/defect, defect accepted GOOD, ma trận ba trạng thái | §3.3–3.4; §7.4; §10.1 |
| I073 — REVIEW nếu được bật — Không loại REVIEW để làm đẹp accuracy; giữ bảng binary baseline riêng | §3.3–3.4; §7.4; §10.1 |

## Cấu trúc, công nghệ, TileManager và interface (§4)

| Tên phần tử | § nguồn |
| --- | --- |
| I074 — Sample và loader chuẩn — Sample frozen: sample_id, source_group, pair_id, partition, image_path, width, height, is_defect | §4.4; §5 Bước 2; contract Step 2 |
| I075 — Sample và loader chuẩn — boxes[Box(class_id,class_name,xyxy)], mask_status, sha256; load_image trả RGB uint8 HWC | §4.4; §5 Bước 2; contract Step 2 |
| I076 — Sample và loader chuẩn — ManifestDataset / YoloTrainSet / AdTrainSet / EvalSet; mỗi ảnh một lần | §4.4; §5 Bước 2; contract Step 2 |
| I077 — Sample và loader chuẩn — allow_test=false: metadata được đọc, truy cập pixel partition cuối bị PermissionError | §4.4; §5 Bước 2; contract Step 2 |
| I078 — Nhánh YOLO 640 — Letterbox target=640, pad_value=114; lưu scale, pad_left/top, orig_w/h | §4.4; §5 Bước 2; contract Step 2–3 |
| I079 — Nhánh YOLO 640 — Ảnh 640×640: scale=1, pad=0; unletterbox_boxes + clip về ảnh gốc | §4.4; §5 Bước 2; contract Step 2–3 |
| I080 — Nhánh YOLO 640 — Round-trip tọa độ sai số không quá 1 px | §4.4; §5 Bước 2; contract Step 2–3 |
| I081 — Nhánh AD native tile-256 — TileManager: tile_size=256, stride=224, overlap=32, pad_mode=reflect; không tile-320 | §4.4; §5 Bước 2,4; contract Step 2–4 |
| I082 — Nhánh AD native tile-256 — Số bước: H lớn hơn 256 thì ceil((H−256)/224)+1; còn lại 1 | §4.4; §5 Bước 2,4; contract Step 2–4 |
| I083 — Nhánh AD native tile-256 — 640×640 thành 9 tile; x0/y0 = 0,224,448; row-major; canvas 704×704, đệm phải/dưới | §4.4; §5 Bước 2,4; contract Step 2–4 |
| I084 — Nhánh AD native tile-256 — TileSpec: tile_id,row,col,x0,y0,size; split giữ nguyên pixel vùng không đệm | §4.4; §5 Bước 2,4; contract Step 2–4 |
| I085 — Nhánh AD native tile-256 — AdTrainSet.num_tiles theo plan hình học: 895 × 9 = 8.055 tile | §4.4; §5 Bước 2,4; contract Step 2–4 |
| I086 — Nhánh AD native tile-256 — Engine RGB uint8 [N,256,256,3]; dataset tensor [9,3,256,256] | §4.4; §5 Bước 2,4; contract Step 2–4 |
| I087 — TileManager hậu xử lý — stitch float32 tile maps: Linear Blending / feathering, tổng trọng số=1, không chia 0 | §4.4; §7.7; contract Step 2–4 |
| I088 — TileManager hậu xử lý — Crop padding: anomaly_map float32 trở về 640×640; image score = max(stitched map) | §4.4; §7.7; contract Step 2–4 |
| I089 — TileManager hậu xử lý — boxes_to_global: cộng offset, clip, bỏ box rỗng, giữ score | §4.4; §7.7; contract Step 2–4 |
| I090 — TileManager hậu xử lý — Global NMS IoU=0.45; loại khi IoU lớn hơn ngưỡng; thứ tự score giảm dần | §4.4; §7.7; contract Step 2–4 |
| I091 — Augmentation chỉ train — Recipe aug_v1 / yolo_aug_v1 có phiên bản; nhẹ và phù hợp PCB, không phá bản chất lỗi | §5 Bước 2; contract Step 2–3 |
| I092 — Augmentation chỉ train — Seed tất định; biến đổi bbox đồng bộ ảnh; không mutate input; split khác bị từ chối | §5 Bước 2; contract Step 2–3 |
| I093 — Augmentation chỉ train — Thay recipe chỉ qua ablation có tên | §5 Bước 2; contract Step 2–3 |
| I094 — Nhánh ablation resize-256 — resize_for_ad bilinear 256; upsample_map bilinear không overshoot | §5 Bước 2; §9; contract Step 2 |
| I095 — Nhánh ablation resize-256 — Bắt buộc đối chứng tile-256 vs resize-256; không thay luồng B03 native | §5 Bước 2; §9; contract Step 2 |
| I096 — Nguồn ảnh và interface chung — BaseImageSource.get_frame trả (sample_id,image); FileImageSource / DirectoryWatcherSource / MockFrameGrabber | §4.4; §8; contract Step 3–4 |
| I097 — Nguồn ảnh và interface chung — BaseDetector / BaseAnomalyDetector: load(artifact), predict(image), describe() | §4.4; §8; contract Step 3–4 |
| I098 — Nguồn ảnh và interface chung — Detection: class_id, class_name, confidence, xyxy_original | §4.4; §8; contract Step 3–4 |
| I099 — Nguồn ảnh và interface chung — EfficientAdResult: anomaly_score, anomaly_map, boxes; AD class_id=None | §4.4; §8; contract Step 3–4 |
| I100 — Nguồn ảnh và interface chung — CLI / UI gọi cùng InferenceService; evaluator chỉ nhận output chuẩn, GT không vào inference | §4.4; §8; contract Step 3–4 |
| I101 — Cấu trúc dự án: cấu hình và core — PLAN.md, plan.v2.md, plan.v3.md, Review.md, review2.md; docs/TRIAGE.md, docs/DECISIONS.md | §4.1 |
| I102 — Cấu trúc dự án: cấu hình và core — README.md, pyproject.toml, requirements khóa theo môi trường | §4.1 |
| I103 — Cấu trúc dự án: cấu hình và core — configs/dataset.yaml, protocol.yaml, models/, experiments/ | §4.1 |
| I104 — Cấu trúc dự án: cấu hình và core — src/pcb_lab/data/, models/, inference/ (TileManager), evaluation/, registry/ | §4.1 |
| I105 — Cấu trúc dự án: ứng dụng và output — app/ Streamlit; scripts/ audit/train/calibrate/benchmark/export; tests/ dữ liệu/routing/metrics | §4.1 |
| I106 — Cấu trúc dự án: ứng dụng và output — data_refs/ snapshot manifest và chữ ký, không nhân bản dataset | §4.1 |
| I107 — Cấu trúc dự án: ứng dụng và output — artifacts/, runs/, reports/ audit/bảng/hình/phân tích lỗi | §4.1 |
| I108 — Cấu trúc dự án: ứng dụng và output — uploads/ ảnh thử riêng không tự vào train; exports/ ZIP, CSV/JSON, ảnh kết quả | §4.1 |
| I109 — Runner GPU khi dùng từ xa — notebooks/ Colab/Kaggle hoặc package riêng dự án | §4.1; §5 Bước 10 |
| I110 — Runner GPU khi dùng từ xa — Resume và kiểm artifact tải về; không mặc định runner ba nhánh cũ | §4.1; §5 Bước 10 |
| I111 — Công nghệ và môi trường — Python, PyTorch, Ultralytics; AD dùng anomalib hoặc standalone | §4.2–4.3; contract Step 4 |
| I112 — Công nghệ và môi trường — Streamlit tiếng Việt; JSON/JSONL config/metadata/prediction; CSV benchmark | §4.2–4.3; contract Step 4 |
| I113 — Công nghệ và môi trường — Venv mới; khóa Python, torch/torchvision, CUDA/driver, anomalib, Ultralytics, UI; không dùng latest | §4.2–4.3; contract Step 4 |
| I114 — Công nghệ và môi trường — Không thay môi trường cũ; kiểm tác dụng phụ loader/audit/packaging nguồn trước tái sử dụng | §4.2–4.3; contract Step 4 |
| I115 — Công nghệ và môi trường — Runner riêng phục vụ baseline/cascade, output root riêng; không chạy lại kiến trúc ba nhánh | §4.2–4.3; contract Step 4 |
| I116 — FastAPI khi cần nhiều client — Chỉ bổ sung khi nhiều client hoặc tích hợp hệ thống ngoài | §4.2 |
| I117 — FastAPI khi cần nhiều client — Không bắt buộc frontend/backend riêng | §4.2 |
| I118 — SQLite khi JSONL bất tiện — Lịch sử UI có thể chuyển sang SQLite khi cần | §4.2 |

## Bước 0–10: đầu vào, xử lý, output chính (§5)

| Tên phần tử | § nguồn |
| --- | --- |
| I119 — Bước 0 — Protocol và môi trường — Vào: release nguồn + ma trận baseline + phần cứng thực có | §5 Bước 0 |
| I120 — Bước 0 — Protocol và môi trường — Protocol deeppcb_cascade_holdout_v1; fusion → development_selection ghi trong protocol mới | §5 Bước 0 |
| I121 — Bước 0 — Protocol và môi trường — Đăng ký trước baseline/metrics/seed42,43,44/ngưỡng mục tiêu/ngân sách; experiment manifest | §5 Bước 0 |
| I122 — Bước 0 — Protocol và môi trường — Kiểm CPU/GPU, RAM/VRAM, dung lượng đĩa, framework/device thực; khóa môi trường local và cloud | §5 Bước 0 |
| I123 — Bước 0 — Protocol và môi trường — Lưu dataset hash, code revision/snapshot, environment lock; không sửa dự án cũ | §5 Bước 0 |
| I124 — Bước 0 — Protocol và môi trường — Ra: configs/protocol.yaml, reports/environment.json, reports/experiment_register.md | §5 Bước 0 |
| I125 — Bước 1 — Audit và EDA — Vào: manifest + release + ảnh/nhãn phát triển; holdout cuối chỉ metadata | §5 Bước 1 |
| I126 — Bước 1 — Audit và EDA — Kiểm decode/hash/kích thước, ID lớp, bbox, nhãn good rỗng, giao group/pair/hash mọi cặp partition | §5 Bước 1 |
| I127 — Bước 1 — Audit và EDA — Thống kê ảnh/box/lớp/số lỗi/kích thước/nhóm/tỷ lệ; đo pin_hole và mouse_bite thực tế | §5 Bước 1 |
| I128 — Bước 1 — Audit và EDA — Overlay train/calibration đủ sáu lớp, lỗi nhỏ và biên; giữ good nền trắng hợp lệ | §5 Bước 1 |
| I129 — Bước 1 — Audit và EDA — Không dùng filename/split làm feature; không suy luận nhãn từ cặp template/test nguồn | §5 Bước 1 |
| I130 — Bước 1 — Audit và EDA — Ra: reports/data_audit.json, reports/data_profile.md, overlay phát triển | §5 Bước 1 |
| I131 — Bước 1 — Audit và EDA — Lỗi nghiêm trọng hoặc thay dữ liệu: tạo phiên bản dữ liệu mới trước khi tiếp tục | §5 Bước 1 |
| I132 — Bước 2 — Tiền xử lý chung — Vào: Sample / ảnh canonical; L, LA, RGBA chuyển RGB nhất quán, không lẫn BGR | §5 Bước 2; contract Step 2 |
| I133 — Bước 2 — Tiền xử lý chung — prepare_input: yolo / ad_tile / ad_resize; PreparedInput.pixels uint8 | §5 Bước 2; contract Step 2 |
| I134 — Bước 2 — Tiền xử lý chung — meta: preprocessing_version/hash, mode, orig_hw, scale/pad hoặc tile_specs, color_order, normalized | §5 Bước 2; contract Step 2 |
| I135 — Bước 2 — Tiền xử lý chung — normalize float32 đúng một lần; gọi lại dữ liệu normalized thì ValueError | §5 Bước 2; contract Step 2 |
| I136 — Bước 2 — Tiền xử lý chung — Ra: configs/preprocessing.yaml (prep_v1), adapter, TileManager, reports/loader_check.json | §5 Bước 2; contract Step 2 |
| I137 — Bước 2 — Tiền xử lý chung — check_loaders.py kiểm n_samples/shape/dtype/min/max/color/normalized/hash; prep_preview.py khớp API/CLI | §5 Bước 2; contract Step 2 |
| I138 — Bước 3 — Baseline YOLO — Vào: train good + defect bbox; calibration chọn checkpoint bằng ultralytics_fitness | §5 Bước 3; contract Step 3 |
| I139 — Bước 3 — Baseline YOLO — Pretrained phổ thông có nguồn/phiên bản/SHA-256; YOLO11n trước, YOLO11s đối chứng dung lượng | §5 Bước 3; contract Step 3 |
| I140 — Bước 3 — Baseline YOLO — imgsz=640, epochs=100, patience=20, AdamW; batch theo VRAM, seed=42 ở vòng đầu | §5 Bước 3; contract Step 3 |
| I141 — Bước 3 — Baseline YOLO — B01/B02 cùng recipe, chỉ khác model_id/architecture/pretrained; khác batch phải ghi lý do | §5 Bước 3; contract Step 3 |
| I142 — Bước 3 — Baseline YOLO — Null lr0/batch/workers/augment hoặc khóa lạ: ConfigError; resume rõ ràng, không ghi đè run | §5 Bước 3; contract Step 3 |
| I143 — Bước 3 — Baseline YOLO — YOLO view: train + calibration, hardlink/copy; labels từ manifest, good rỗng, data.yaml không có test | §5 Bước 3; contract Step 3 |
| I144 — Bước 3 — Baseline YOLO — Ra: best.pt + last.pt, artifact.json, model_card.md, calibration_per_class.json, validation.md; log loss/mAP/recall từng lớp | §5 Bước 3; contract Step 3 |
| I145 — Bước 3 — Baseline YOLO — Validation precision/recall tại confidence tối đa F1 có pr_definition/pr_conf; không nhầm ngưỡng vận hành | §5 Bước 3; contract Step 3 |
| I146 — Bước 4 — B03 EfficientAD-S — Vào: chỉ 895 train good, 0 defect; DataLeakError nếu lẫn defect | §5 Bước 4; §7.1; contract Step 4 |
| I147 — Bước 4 — B03 EfficientAD-S — Teacher pretrained + student + autoencoder; học sai khác đặc trưng và tái tạo | §5 Bước 4; §7.1; contract Step 4 |
| I148 — Bước 4 — B03 EfficientAD-S — PyTorch; anomalib hoặc standalone; model_size=small, batch=1, 70 epoch tham khảo, AdamW | §5 Bước 4; §7.1; contract Step 4 |
| I149 — Bước 4 — B03 EfficientAD-S — Teacher mean/std chỉ train good; map quantiles 0.90/0.995 chỉ 230 calibration good | §5 Bước 4; §7.1; contract Step 4 |
| I150 — Bước 4 — B03 EfficientAD-S — Không chuẩn hóa ImageNet hai lần; tắt auto-threshold / auto-split / cập nhật ngưỡng tự động | §5 Bước 4; §7.1; contract Step 4 |
| I151 — Bước 4 — B03 EfficientAD-S — Null lr/weight_decay/device chưa duyệt hoặc khóa lạ: từ chối; seed=42, deterministic=true | §5 Bước 4; §7.1; contract Step 4 |
| I152 — B03 — EfficientAD-S (tile 256) — model.pt chứa weights teacher/student/autoencoder, normalization_stats và tile/inference params | §5 Bước 4; §7.1; contract Step 4 |
| I153 — B03 — EfficientAD-S (tile 256) — artifact.json: normalization_stats; model_card.md; calibration_stats.json: phân phối score good/defect; báo cáo false positive | §5 Bước 4; §7.1; contract Step 4 |
| I154 — Bước 4 — B03 EfficientAD-S — Fit → save → reload → predict: map/score sai số không quá 1e-5; ghi optimizer steps thực | §5 Bước 4; §7.1; contract Step 4 |
| I155 — Tài nguyên regularization khi dùng anomalib — Kiểm pretrained teacher và ImageNette theo triển khai đã khóa | §5 Bước 4 |
| I156 — Tài nguyên regularization khi dùng anomalib — ImageNette phục vụ regularization, không cộng vào 895 PCB good | §5 Bước 4 |
| I157 — Trích dự đoán thô phát triển — preds_calibration.jsonl: 460 dòng; preds_fusion.jsonl: 306 dòng; mỗi sample một dòng, sort sample_id | §5 Bước 3–4; contract Step 3–4 |
| I158 — Trích dự đoán thô phát triển — YOLO conf_floor=0.001, NMS iou=0.7, max_det=300, agnostic_nms=false, half=false | §5 Bước 3–4; contract Step 3–4 |
| I159 — Trích dự đoán thô phát triển — Candidate floor độc lập tau_yolo; mAP confidence sweep tách recall tại ngưỡng vận hành | §5 Bước 3–4; contract Step 3–4 |
| I160 — Trích dự đoán thô phát triển — Test bị PermissionError; lỗi giữ dòng/error_reason, score=null, không thành GOOD/0 | §5 Bước 3–4; contract Step 3–4 |
| I161 — Trích dự đoán thô phát triển — timing_ms=null trong raw extraction; không dùng cache prediction để đo latency | §5 Bước 3–4; contract Step 3–4 |
| I162 — Bước 5 — Hiệu chỉnh có điều kiện — Vào: score toàn calibration, gồm toàn bộ defect qua AD độc lập | §5 Bước 5; DECISIONS R2 |
| I163 — Bước 5 — Hiệu chỉnh có điều kiện — 2-D Grid Search (tau_yolo,tau_ad): [0.1,0.9] × [mu_ad,mu_ad+3sigma_ad] | §5 Bước 5; DECISIONS R2 |
| I164 — Bước 5 — Hiệu chỉnh có điều kiện — mu_ad/sigma_ad từ phân phối defect calibration; lưu toàn bộ điểm lưới | §5 Bước 5; DECISIONS R2 |
| I165 — Bước 5 — Hiệu chỉnh có điều kiện — Tối đa Defect Recall với FRR ≤5%; mốc 1% là phân tích phụ; hòa thì ưu tiên latency thấp | §5 Bước 5; DECISIONS R2 |
| I166 — Bước 5 — Hiệu chỉnh có điều kiện — Ngưỡng tái tạo được; không fit logistic fusion head | §5 Bước 5; DECISIONS R2 |
| I167 — Phân biệt chuẩn hóa và ngưỡng — normalization_stats: source_split=calibration, n_good_images=230, quantile_min/max, mean/std | §5 Bước 4–5; contract Step 4 |
| I168 — Phân biệt chuẩn hóa và ngưỡng — Map quantiles 0.90/0.995 tính SAU khi fit trên 230 calibration good; tau_ad dùng good/defect để calibration | §5 Bước 4–5; contract Step 4 |
| I169 — Phân biệt chuẩn hóa và ngưỡng — Theo dõi riêng good routed, cỡ mẫu và phân phối; không ước lượng cực trị quá mức | §5 Bước 4–5; contract Step 4 |
| I170 — Artifact ngưỡng và routing — Ra: thresholds.json, routing_policy.json, báo cáo calibration/development selection | §5 Bước 5 |
| I171 — Artifact ngưỡng và routing — Fusion chạy H00 và H01 thật: delta_recall, delta_FRR, delta_latency; route_rate good/defect | §5 Bước 5 |
| I172 — Artifact ngưỡng và routing — Đổi checkpoint/resize/tile/score pooling/hậu xử lý: calibration lại và tăng version | §5 Bước 5 |
| I173 — Artifact ngưỡng và routing — Giữ tau_pixel sinh candidate độc lập với tau_ad quyết định ảnh | §5 Bước 5 |
| I174 — Bước 6 — Development và shortlist — Vào: calibration/fusion + evaluator chung + baseline/OR/cascade đã hiệu chỉnh | §5 Bước 6; §7.1; §12 |
| I175 — Bước 6 — Development và shortlist — Vòng seed42: B01/B02/B03/H00/H01; H02 thuộc ma trận sàng lọc | §5 Bước 6; §7.1; §12 |
| I176 — Bước 6 — Development và shortlist — Ưu tiên ablation dung lượng YOLO, tile vs resize, ngưỡng, rescue và added false reject | §5 Bước 6; §7.1; §12 |
| I177 — Bước 6 — Development và shortlist — Chọn shortlist có lý do; chạy thêm seed43/44, không chọn seed đẹp nhất | §5 Bước 6; §7.1; §12 |
| I178 — Bước 6 — Development và shortlist — Tập báo cáo cuối tối thiểu B01/B03/H00/H01; đối thủ mạnh bổ sung theo ngân sách | §5 Bước 6; §7.1; §12 |
| I179 — Bước 6 — Development và shortlist — Ra: bảng development, bảng ablation, shortlist và ngân sách thực đã dùng | §5 Bước 6; §7.1; §12 |
| I180 — Bước 7 — Khóa trước đánh giá cuối — Vào: shortlist và cấu hình demo đã chọn bằng development | §5 Bước 7 |
| I181 — Bước 7 — Khóa trước đánh giá cuối — frozen_evaluation.json: checkpoint hash, tau, schema, split hash, preprocessing | §5 Bước 7 |
| I182 — Bước 7 — Khóa trước đánh giá cuối — runtime/backend, seed, metrics, quy tắc chọn model; danh sách phương pháp gồm H00/H01 | §5 Bước 7 |
| I183 — Bước 7 — Khóa trước đánh giá cuối — Chỉ sau khóa mới mở luồng test; không dùng test để chọn checkpoint/ngưỡng | §5 Bước 7 |
| I184 — Partition test — chỉ đánh giá cuối — 440 ảnh = 220 good + 220 defect; 220 pair; 2 nhóm group44000/group92000 | §1.2; §5 Bước 7; §7.6 |
| I185 — Partition test — chỉ đánh giá cuối — Cùng danh sách canonical cho mọi model đã khóa; mỗi ảnh đúng một lần | §1.2; §5 Bước 7; §7.6 |
| I186 — Partition test — chỉ đánh giá cuối — Xuất từng seed / source_group / worst-group; đo pipeline routing thật | §1.2; §5 Bước 7; §7.6 |
| I187 — Partition test — chỉ đánh giá cuối — Ra: benchmark_summary.csv, benchmark_per_class.csv, benchmark_per_group.csv, predictions.jsonl | §1.2; §5 Bước 7; §7.6 |
| I188 — Partition test — chỉ đánh giá cuối — Không đạt thì báo không đạt; không sửa tau rồi ghi đè; nghiên cứu tiếp cần protocol/tập xác nhận mới | §1.2; §5 Bước 7; §7.6 |
| I189 — Partition test — chỉ đánh giá cuối — Phân tích lỗi test sau đánh giá chỉ giải thích hạn chế, không tuyên bố cải tiến trên cùng test | §1.2; §5 Bước 7; §7.6 |
| I190 — Bước 8 — Streamlit nội bộ — Vào: core đã kiểm Bước 5 + registry; sườn UI sớm có dữ liệu minh họa được đánh dấu | §5 Bước 8; §10 |
| I191 — Bước 8 — Streamlit nội bộ — UI gọi InferenceService qua BaseImageSource, không đọc file trực tiếp | §5 Bước 8; §10 |
| I192 — Bước 8 — Streamlit nội bộ — Chỉ load model đăng ký; cache giữa ảnh, trạng thái tải/chạy/lỗi | §5 Bước 8; §10 |
| I193 — Bước 8 — Streamlit nội bộ — Benchmark khóa tách thử ngưỡng; mỗi chỉnh tham số tạo run mới, không ghi đè artifact | §5 Bước 8; §10 |
| I194 — Bước 8 — Streamlit nội bộ — Ảnh mẫu mặc định calibration/fusion; gallery cuối chỉ trong chế độ xem báo cáo sau khóa | §5 Bước 8; §10 |
| I195 — Bước 8 — Streamlit nội bộ — Ra: ứng dụng upload/batch/benchmark; overlay và số liệu khớp core/CLI | §5 Bước 8; §10 |
| I196 — Bước 9 — Profile sau baseline đúng — Vào: baseline đúng + artifact đã khóa | §5 Bước 9 |
| I197 — Bước 9 — Profile sau baseline đúng — Profile decode/preprocess/YOLO/AD/stitch/NMS/hậu xử lý/render/load | §5 Bước 9 |
| I198 — Bước 9 — Profile sau baseline đúng — Giữ model trong RAM, tránh copy tensor thừa; cascade chỉ gọi AD khi routed | §5 Bước 9 |
| I199 — Bước 9 — Profile sau baseline đúng — Kiểm backend Intel Arc XPU / CUDA cloud; giữ CPU demo nếu khả thi | §5 Bước 9 |
| I200 — Bước 9 — Profile sau baseline đúng — Ra: bảng accuracy–latency–memory theo backend; lợi ích phải đo, sai khác trong mức đã định | §5 Bước 9 |
| I201 — Export runtime khi hỗ trợ — FP16 / ONNX / OpenVINO: thử theo phần cứng và khả năng export thực | §5 Bước 9; §9 |
| I202 — Export runtime khi hỗ trợ — So PyTorch với export trên cùng ảnh: score/box/decision và E2E | §5 Bước 9; §9 |
| I203 — Export runtime khi hỗ trợ — Artifact version riêng; calibration lại nếu cần rồi khóa lại trước đánh giá | §5 Bước 9; §9 |
| I204 — Bước 10 — Đóng gói bàn giao — Vào: checkpoint + metadata + benchmark + UI đã kiểm | §5 Bước 10; §2.1 |
| I205 — Bước 10 — Đóng gói bàn giao — README tiếng Việt, model card, dataset/protocol card, limitations và hướng dẫn tái lập | §5 Bước 10; §2.1 |
| I206 — Bước 10 — Đóng gói bàn giao — Demo phát triển: YOLO TP, AD rescue, good đúng, FP/FN tiêu biểu nếu có | §5 Bước 10; §2.1 |
| I207 — Bước 10 — Đóng gói bàn giao — Benchmark OR/cascade, confusion matrix, PR/ROC/DET phù hợp, sáu lớp, Pareto latency–recall, ablation | §5 Bước 10; §2.1 |
| I208 — Bước 10 — Đóng gói bàn giao — Ra: gói chạy / ZIP, báo cáo và demo tái lập; checkpoint đúng quyền sử dụng | §5 Bước 10; §2.1 |
| I209 — Bước 10 — Đóng gói bàn giao — Kiểm cài/chạy từ thư mục dự án; người khác theo README đối chiếu được kết quả | §5 Bước 10; §2.1 |
| I210 — CLI dùng core và evaluator chung — Lệnh audit / train / calibrate / benchmark / UI; cấu hình mẫu | §4.2; §5 Bước 10; §11.1 |
| I211 — CLI dùng core và evaluator chung — Inference gọi InferenceService; benchmark gọi cùng evaluator như UI | §4.2; §5 Bước 10; §11.1 |
| I212 — CLI dùng core và evaluator chung — Công bố tác vụ cần cloud GPU và tài nguyên cần tải | §4.2; §5 Bước 10; §11.1 |
| I213 — Smoke CPU theo hợp đồng — Step 3: 1 epoch, view nhỏ, random từ kiến trúc, không cần pretrained | §5 Bước 3–4; contract Step 3–4; NOTES N04 |
| I214 — Smoke CPU theo hợp đồng — Step 4: 1 epoch, limit good=4, CPU; không tạo artifact chính thức | §5 Bước 3–4; contract Step 3–4; NOTES N04 |
| I215 — Smoke CPU theo hợp đồng — runs/smoke, smoke=true; không thay baseline, không chứng nhận Gate 0 cloud | §5 Bước 3–4; contract Step 3–4; NOTES N04 |

## Quy tắc đánh giá công bằng (§6)

| Tên phần tử | § nguồn |
| --- | --- |
| I216 — Đánh giá công bằng: dữ liệu và lựa chọn — Cùng danh sách canonical, không thêm template riêng và không bỏ ảnh model lỗi | §6 |
| I217 — Đánh giá công bằng: dữ liệu và lựa chọn — Công bố supervision: detector bbox+good/defect, anomaly good-only, OR/cascade kế thừa hai nhánh | §6 |
| I218 — Đánh giá công bằng: dữ liệu và lựa chọn — Calibration chọn checkpoint/ngưỡng; fusion chọn cấu hình; holdout cuối không tham gia lựa chọn | §6 |
| I219 — Đánh giá công bằng: dữ liệu và lựa chọn — Công bố số thử, GPU-hours, external/pretrained data và ngân sách tuning từng model | §6 |
| I220 — Đánh giá công bằng: dữ liệu và lựa chọn — Cùng device/backend/precision/batch/timing mỗi bảng; khác điều kiện tách bảng | §6 |
| I221 — Đánh giá công bằng: dữ liệu và lựa chọn — Công bố input size, tile count/stride; resolution ablation tách kiến trúc khỏi lượng pixel | §6 |
| I222 — Đánh giá công bằng: báo cáo — Upload không nhãn: chỉ prediction/score; accuracy phải kèm dataset/split/N/artifact | §6; §8 |
| I223 — Đánh giá công bằng: báo cáo — Error rate + số mẫu thành công; run thiếu dự đoán không được xếp hạng như run đầy đủ | §6; §8 |
| I224 — Đánh giá công bằng: báo cáo — Dataset khác domain/schema/protocol có bảng riêng; phân biệt frozen zero-shot transfer và retrain/recalibration | §6; §8 |
| I225 — Đánh giá công bằng: báo cáo — Dataset adapter mới khai báo taxonomy, split/group, nhãn good, annotation, nguồn | §6; §8 |
| I226 — Đánh giá công bằng: báo cáo — Tập chỉ defect không đủ FRR / calibrate normal; thiếu điều kiện ghi N/A | §6; §8 |
| I227 — Đánh giá công bằng: báo cáo — Không gộp missing_hole với pin_hole để tạo taxonomy giả | §6; §8 |

## Ma trận model và bảng tổng hợp (§7.1–7.2, §7.8)

| Tên phần tử | § nguồn |
| --- | --- |
| I228 — B01 — YOLO11n — Detector sáu lớp, supervision good + defect bbox; baseline nhanh bắt buộc | §7.1; §5 Bước 3 |
| I229 — B01 — YOLO11n — Checkpoint B01 dùng lại cho H00 và H01, không train lại để đổi đối chứng | §7.1; §5 Bước 3 |
| I230 — B02 — YOLO11s — Detector sáu lớp, supervision good + defect bbox; bắt buộc vòng sàng lọc | §7.1; §5 Bước 3 |
| I231 — B02 — YOLO11s — Checkpoint B02 dùng lại cho H02 | §7.1; §5 Bước 3 |
| I232 — B04 — PatchCore — Good-only train; đối chứng memory bank, nên có nếu đủ ngân sách | §7.1; §12; DECISIONS S3 |
| I233 — B04 — PatchCore — PaDiM đã smoke ở nguồn; tham khảo nếu PatchCore không chạy được | §7.1; §12; DECISIONS S3 |
| I234 — B05 — placeholder, VERIFY V1 — Loại khỏi ma trận bắt buộc; chỉ tái đưa detector mở rộng khi được xác minh | §7.1; DECISIONS R3,V1; NOTES N03 |
| I235 — B05 — placeholder, VERIFY V1 — Không đăng ký tên model chưa được kiểm chứng | §7.1; DECISIONS R3,V1; NOTES N03 |
| I236 — B06 — EfficientAD-M — Good-only train; mở rộng để so dung lượng anomaly | §7.1; §12 |
| I237 — B07 — FastFlow hoặc STFPM — Good-only train; thêm họ anomaly khi đủ ngân sách | §7.1; §12 |
| I238 — H03 — cấu hình biến thể có căn cứ — YOLO tốt nhất trên development → AD biến thể tốt nhất | §7.1 |
| I239 — H03 — cấu hình biến thể có căn cứ — Ghi rõ hai artifact; chỉ thực hiện khi thay đổi có căn cứ | §7.1 |
| I240 — Bảng tổng hợp và bảng phụ — ID, Model/mode, supervision; Balanced acc., Defect recall/precision, F1, MCC, FRR, Defect→GOOD, ECPB | §7.2; §7.8 |
| I241 — Bảng tổng hợp và bảng phụ — E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A, B04=N/A | §7.2; §7.8 |
| I242 — Bảng tổng hợp và bảng phụ — H00 đứng trước H01; B01/B02/B03/H00/H01/H02/B04 theo ma trận ưu tiên | §7.2; §7.8 |
| I243 — Bảng tổng hợp và bảng phụ — CSV đầy đủ: dataset_id, protocol_id, artifact/config hash, seed, thiết bị | §7.2; §7.8 |
| I244 — Bảng tổng hợp và bảng phụ — —=chưa đo; N/A=không áp dụng; không đưa điểm số nguồn vào bảng dự án | §7.2; §7.8 |
| I245 — Bảng tổng hợp và bảng phụ — Bảng detection sáu lớp và bảng localization riêng; §7.8 là nhãn giữ số, không có nội dung bổ sung | §7.2; §7.8 |

## Metrics, tài nguyên, bất định, chọn model, heatmap→box (§7.3–7.7)

| Tên phần tử | § nguồn |
| --- | --- |
| I246 — Một evaluator dùng chung — CLI / UI / benchmark cùng evaluator; chỉ tại đây prediction gặp ground truth | §4.2; §5 Bước 6; §8 |
| I247 — Một evaluator dùng chung — Input: output chuẩn + nhãn hợp lệ; adapter model không nhận GT | §4.2; §5 Bước 6; §8 |
| I248 — Một evaluator dùng chung — load / predict / describe nhất quán; chưa có nhãn chỉ tổng hợp prediction, không tạo accuracy | §4.2; §5 Bước 6; §8 |
| I249 — Metrics mức ảnh — DEFECT=1, GOOD=0; TP/TN/FP/FN và số FN tuyệt đối | §7.2–7.3 |
| I250 — Metrics mức ảnh — Balanced accuracy=(TPR+TNR)/2; Accuracy trùng trên 50/50 nên bỏ cột chính | §7.2–7.3 |
| I251 — Metrics mức ảnh — Defect recall / TPR=TP/(TP+FN); Defect→GOOD / miss rate=FN/(TP+FN) | §7.2–7.3 |
| I252 — Metrics mức ảnh — FRR=FP/(FP+TN); precision=TP/(TP+FP) | §7.2–7.3 |
| I253 — Metrics mức ảnh — F1: trung bình điều hòa precision/recall; MCC: tương quan từ confusion matrix | §7.2–7.3 |
| I254 — Metrics mức ảnh — Image AUROC / AUPRC chỉ từ score liên tục có nghĩa, không từ nhãn cứng | §7.2–7.3 |
| I255 — Metrics mức ảnh — Cascade ban đầu AUROC/AUPRC=N/A; score hợp nhất phải là biến thể hiệu chỉnh riêng | §7.2–7.3 |
| I256 — Metrics mức ảnh — Per-class image recall: một ảnh có thể thuộc nhiều lớp; không thay box recall | §7.2–7.3 |
| I257 — ECPB: chi phí mô phỏng — ECPB=(C_FN×FN_sim+C_FP×FP_sim)/N_sim | §7.2–7.3; §10.2; DECISIONS R4,H2; NOTES N02 |
| I258 — ECPB: chi phí mô phỏng — N_sim=10.000: 100 defect, 9.900 good; đây là mô phỏng 1%/99% | §7.2–7.3; §10.2; DECISIONS R4,H2; NOTES N02 |
| I259 — ECPB: chi phí mô phỏng — C_FP=1; C_FN cần HUMAN/VERIFY H2, không gắn trọng số chưa duyệt như cam kết nhà máy | §7.2–7.3; §10.2; DECISIONS R4,H2; NOTES N02 |
| I260 — Giả định trọng số chưa chốt — C_FN=50×C_FP là đề xuất cần xác nhận theo DECISIONS | §7.3; DECISIONS H2; TRIAGE R4 |
| I261 — Giả định trọng số chưa chốt — Không tự chốt 100:1 từ plan; xem đối chiếu NOTES N02 | §7.3; DECISIONS H2; TRIAGE R4 |
| I262 — Metrics detection và heatmap — Bảng detection: mAP@0.5, mAP@0.5:0.95, box recall tại ngưỡng, AP/recall sáu lớp, FP box/ảnh good | §7.2–7.3,7.7; contract Step 4 |
| I263 — Metrics detection và heatmap — YOLO output của cascade báo riêng; AD N/A cho AP sáu lớp, không ép heatmap thành class | §7.2–7.3,7.7; contract Step 4 |
| I264 — Heatmap → candidate boxes R8 — Heatmap stitched → tau_pixel=0.5 → Morph Open k=3 → Connected Components | §7.7; contract Step 4 |
| I265 — Heatmap → candidate boxes R8 — Area lớn hơn min_area=20 → Bounding Rect → Global NMS IoU=0.45 | §7.7; contract Step 4 |
| I266 — Heatmap → candidate boxes R8 — Box confidence=max pixel trong component; xyxy_original clip ảnh gốc | §7.7; contract Step 4 |
| I267 — Metrics detection và heatmap — Point-in-Box: argmax(Map) thuộc BBox_GT và Area_component ≤3×Area_GT | §7.2–7.3,7.7; contract Step 4 |
| I268 — Metrics detection và heatmap — Class-agnostic localization ở bảng phụ; không thay mAP hay gọi là segmentation accuracy | §7.2–7.3,7.7; contract Step 4 |
| I269 — Metrics hybrid và Iso-FRR — route_rate=ảnh thực chạy AD/tổng ảnh; thêm riêng good/defect | §7.4 |
| I270 — Metrics hybrid và Iso-FRR — rescued_defects: YOLO FN được AD cứu; rescue_rate=rescued_defects/FN_YOLO; mẫu số 0 → N/A | §7.4 |
| I271 — Metrics hybrid và Iso-FRR — added_false_rejects: good YOLO cho qua nhưng AD báo lỗi; không cứu được FP sẵn có của YOLO | §7.4 |
| I272 — Metrics hybrid và Iso-FRR — unclassified_positive_rate=AD DEFECT chưa có loại/tổng DEFECT của pipeline | §7.4 |
| I273 — Metrics hybrid và Iso-FRR — delta_recall(H01−H00), delta_FRR(H01−H00), delta_latency(H01−H00), cùng checkpoint | §7.4 |
| I274 — Metrics hybrid và Iso-FRR — DET/ROC và Recall @ FRR={1%,3%,5%} cho B01/B03/H00/H01 | §7.4 |
| I275 — Metrics hybrid và Iso-FRR — Operating point chọn trên calibration; không chỉnh holdout cuối để ép FRR bằng nhau | §7.4 |
| I276 — Đo latency và tài nguyên — Tách model load/cold start và steady-state; warm-up 30, timing 440 ảnh ×3 lượt | §7.5; §5 Bước 7,9 |
| I277 — Đo latency và tài nguyên — N chất lượng vẫn 440, không phải 1.320; batch=1 cho tương tác, throughput batch lớn là track riêng | §7.5; §5 Bước 7,9 |
| I278 — Đo latency và tài nguyên — Đồng bộ GPU/XPU khi cần; E2E=decode→preprocess→tile→model thực chạy→hậu xử lý | §7.5; §5 Bước 7,9 |
| I279 — Đo latency và tài nguyên — UI/encode/render/network đo riêng; không dùng cache score để đo latency routing | §7.5; §5 Bước 7,9 |
| I280 — Đo latency và tài nguyên — Mean/p50/p95, từng tầng, peak RAM/VRAM, artifact size, train/fit time, throughput thực | §7.5; §5 Bước 7,9 |
| I281 — Đo latency và tài nguyên — Công bố device/backend/precision/input size/tile count; không so 256 và 640 như cùng input | §7.5; §5 Bước 7,9 |
| I282 — Đo latency và tài nguyên — Cascade mean≈T_yolo+route_rate×E[T_ad\|routed]+overhead; p95 phải đo trực tiếp | §7.5; §5 Bước 7,9 |
| I283 — Đo latency và tài nguyên — OR≈T_yolo+T_ad; latency riêng good/defect, prevalence khác chỉ mô phỏng có chú thích | §7.5; §5 Bước 7,9 |
| I284 — Bất định và giới hạn so sánh — Ít nhất ba seed42/43/44: từng seed và mean±std; seed std không là CI tổng quát hóa | §1.4; §7.6 |
| I285 — Bất định và giới hạn so sánh — TP/TN/FP/FN theo source_group và worst-group recall/FRR | §1.4; §7.6 |
| I286 — Bất định và giới hạn so sánh — 1 FP hoặc FN trên nhóm 220 ảnh tương ứng khoảng 0,45 điểm phần trăm | §1.4; §7.6 |
| I287 — Bất định và giới hạn so sánh — Calibration chọn checkpoint lẫn ngưỡng có thể lạc quan; fusion sau selection không còn độc lập | §1.4; §7.6 |
| I288 — Bất định và giới hạn so sánh — Không gọi quy trình là hiệu chuẩn có bảo đảm thống kê; N=2 không chứng minh chuyển miền | §1.4; §7.6 |
| I289 — Bootstrap pair-level nếu dùng — Chỉ sampling variance nội bộ 2 nhóm; không ngoại suy domain shift | §7.6; DECISIONS R1 |
| I290 — Bootstrap pair-level nếu dùng — Không thay thế đa dạng dữ liệu; không LOGO-CV 11 groups | §7.6; DECISIONS R1 |
| I291 — Quy tắc chọn model — Trên development: đạt ràng buộc FRR trước, tối đa recall, rồi p95 và memory thấp | §7.6; §2.2 |
| I292 — Quy tắc chọn model — Pareto chất lượng–latency–memory, không tự gán điểm tổng hợp thiếu cơ sở | §7.6; §2.2 |
| I293 — Quy tắc chọn model — Mốc kế hoạch recall ≥95%, FRR ≤5%; khóa ngân sách latency sau biết phần cứng demo | §7.6; §2.2 |
| I294 — Quy tắc chọn model — Muốn đủ sáu lớp/định vị: cần detector AP/box recall, không thay bằng AD-only binary | §7.6; §2.2 |
| I295 — Quy tắc chọn model — Nếu không đạt: báo khoảng cách/trade-off; hybrid không đủ lợi ích thì chọn baseline | §7.6; §2.2 |
| I296 — Quy tắc chọn model — So H00/H01 tại cùng điều kiện; giữ kết quả âm, không mặc định hybrid tốt nhất | §7.6; §2.2 |

## Prediction record và registry (§8)

| Tên phần tử | § nguồn |
| --- | --- |
| I297 — Prediction record: định danh và provenance — run_id, sample_id, image_sha256, dataset_id, split, source_group | §8 |
| I298 — Prediction record: định danh và provenance — model_id, model_version, checkpoint_hashes, protocol_id, config_hash | §8 |
| I299 — Prediction record: định danh và provenance — mode, final_status, decision_source, routing_reason, error_reason | §8 |
| I300 — Prediction record: thực thi và điểm — yolo_executed, efficientad_executed, yolo_image_score, anomaly_score | §8 |
| I301 — Prediction record: thực thi và điểm — tau_yolo, tau_ad, score_definition, preprocessing_version, tiling_params | §8 |
| I302 — Prediction record: thực thi và điểm — Tầng không chạy: score và model timing=null, không phải 0 | §8 |
| I303 — Prediction record: hình học và output — detections[{class_id,class_name,confidence,xyxy_original}] | §8 |
| I304 — Prediction record: hình học và output — anomaly_regions, anomaly_map_path, annotated_image_path | §8 |
| I305 — Prediction record: hình học và output — ground_truth_available, ground_truth_reference; GT chỉ dùng tại evaluator | §8 |
| I306 — Prediction record: timing và runtime — timing_ms{decode,preprocess,yolo,efficientad,stitch,nms,postprocess,total_core} | §8 |
| I307 — Prediction record: timing và runtime — device, backend, precision, input_size, tile_count, timestamp | §8 |
| I308 — Raw predictions Step 3/4 — run_id,sample_id,image_sha256,split,source_group,model_id,checkpoint_sha256,config_hash | §5 Bước 3–4; §8; contract Step 3–4 |
| I309 — Raw predictions Step 3/4 — preprocessing_version; YOLO inference_params,detections,yolo_image_score | §5 Bước 3–4; §8; contract Step 3–4 |
| I310 — Raw predictions Step 3/4 — AD tile_params,anomaly_score,boxes[{xyxy,confidence,class_name}] | §5 Bước 3–4; §8; contract Step 3–4 |
| I311 — Raw predictions Step 3/4 — error_reason; timing_ms=null; schema raw khác record đích §8, không tự đổi tên khóa | §5 Bước 3–4; §8; contract Step 3–4 |
| I312 — Registry và tính tương thích — Giữ checkpoint, threshold, class_order, normalization AD và benchmark đúng artifact | §4.1; §8; contract Step 3–4 |
| I313 — Registry và tính tương thích — load / from_artifact kiểm checkpoint SHA-256, schema/class order, smoke opt-in | §4.1; §8; contract Step 3–4 |
| I314 — Registry và tính tương thích — ArtifactMismatchError / ClassOrderError / SmokeArtifactError; không trộn threshold/cache/report khác nguồn | §4.1; §8; contract Step 3–4 |
| I315 — Registry và tính tương thích — Config hash canonical không chứa seed; preprocessing version/hash có nguồn | §4.1; §8; contract Step 3–4 |
| I316 — Registry và tính tương thích — Không nạp checkpoint PCB cũ thiếu split/schema; pretrained phổ thông có nguồn, version, hash | §4.1; §8; contract Step 3–4 |
| I317 — Artifacts và run theo seed — runs/yolo hoặc runs/efficientad: model_id/seed, run_manifest.json, env.json, train.log, checkpoints | §4.1; §5 Bước 3–4,7; contract Step 3–4 |
| I318 — Artifacts và run theo seed — artifacts/yolo: best.pt,last.pt,artifact.json,model_card.md,calibration_per_class.json,preds_calibration/fusion | §4.1; §5 Bước 3–4,7; contract Step 3–4 |
| I319 — Artifacts và run theo seed — artifacts/efficientad: model.pt,artifact.json,model_card.md,calibration_stats.json,preds_calibration/fusion | §4.1; §5 Bước 3–4,7; contract Step 3–4 |
| I320 — Artifacts và run theo seed — run manifest: schema/run/model/recipe/seed/smoke, git_commit/dirty, versions/device, dataset release hash, args_used | §4.1; §5 Bước 3–4,7; contract Step 3–4 |
| I321 — Artifacts và run theo seed — epochs/best_epoch/early stop/resume nếu có, train_time, peak RAM/VRAM, checkpoint hashes | §4.1; §5 Bước 3–4,7; contract Step 3–4 |
| I322 — Artifacts và run theo seed — artifact metadata: normalization/inference/tile params hoặc class_order, config hash, run_manifest path/hash, commit tạo | §4.1; §5 Bước 3–4,7; contract Step 3–4 |
| I323 — Artifacts và run theo seed — validation report và model card tiếng Việt; scope, supervision, hạn chế N=2, không dùng test | §4.1; §5 Bước 3–4,7; contract Step 3–4 |

## Ablation và stress test (§9)

| Tên phần tử | § nguồn |
| --- | --- |
| I324 — Ablation phát triển — YOLO-only / AD-only / OR / cascade: cùng checkpoint/split/log score | §9; §5 Bước 6 |
| I325 — Ablation phát triển — YOLO11n vs YOLO11s: cùng recipe, công bố ngân sách | §9; §5 Bước 6 |
| I326 — Ablation phát triển — AD tile-256 vs resize-256 bắt buộc; đo bbox nhỏ, khóa stride/blend/NMS | §9; §5 Bước 6 |
| I327 — Ablation phát triển — Confidence + AD threshold: 2-D grid search calibration | §9; §5 Bước 6 |
| I328 — Ablation phát triển — Một seed vs ba seed: cùng split và ngân sách | §9; §5 Bước 6 |
| I329 — Ablation phát triển — Ưu tiên baseline→routing→threshold→resolution; không đổi mọi biến cùng lúc | §9; §5 Bước 6 |
| I330 — Ablation pooling mở rộng — Default max score vs top-k: biến thể riêng để kiểm nhiễu cực đại | §5 Bước 4; §9 |
| I331 — Ablation pooling mở rộng — Calibration riêng cho mỗi pooling, lưu version riêng | §5 Bước 4; §9 |
| I332 — Robustness / stress riêng — Xoay, mờ, đổi sáng, nén: tập phát triển, giữ liên kết ảnh gốc, không trộn benchmark chính | §9; DECISIONS R10 |
| I333 — Robustness / stress riêng — specular_glare — lóa phản xạ; motion_blur — rung lệch trục; defocus_blur — mất nét | §9; DECISIONS R10 |
| I334 — Robustness / stress riêng — alignment_jitter: dx/dy trong [−2,2] px, dtheta trong [−0.5°,0.5°] | §9; DECISIONS R10 |
| I335 — Robustness / stress riêng — Đánh giá jitter riêng B03/H01, kiểm false positive dọc biên do lệch template | §9; DECISIONS R10 |
| I336 — Lỗi chưa thấy — protocol riêng — Leave-one-defect-type-out chỉ khi cần; không chặn MVP sáu lớp | §2.2; §9 |
| I337 — Lỗi chưa thấy — protocol riêng — Ảnh đa lỗi chứa lớp giữ lại phải loại khỏi toàn dữ liệu train/tuning có thể lộ lớp | §2.2; §9 |
| I338 — Lỗi chưa thấy — protocol riêng — Không chỉ xóa bbox rồi coi là nền; kiểm số ảnh còn lại trước chạy | §2.2; §9 |

## Các màn hình giao diện (§10)

| Tên phần tử | § nguồn |
| --- | --- |
| I339 — Màn hình kiểm thử một ảnh — Upload PNG/JPG/JPEG; kích thước, dung lượng, ảnh gốc, cảnh báo khác domain/kích thước | §10.1 |
| I340 — Màn hình kiểm thử một ảnh — Chọn artifact đăng ký + mode; nút chạy/tiến độ; không load lại model khi chỉnh hiển thị | §10.1 |
| I341 — Màn hình kiểm thử một ảnh — Ảnh gốc/bbox/heatmap stitched 640×640 nếu AD chạy; bật/tắt overlay, opacity, zoom lỗi nhỏ | §10.1 |
| I342 — Màn hình kiểm thử một ảnh — GOOD / DEFECT / ERROR và REVIEW nếu bật; lý do, decision_source, tuyến xử lý | §10.1 |
| I343 — Màn hình kiểm thử một ảnh — AD không chạy: hiển thị Không chạy, không heatmap giả | §10.1 |
| I344 — Màn hình kiểm thử một ảnh — Bảng bbox: class/confidence/xyxy/kích thước; AD score/tau/vùng chưa xác định loại | §10.1 |
| I345 — Màn hình kiểm thử một ảnh — Model/version/hash rút gọn/tau/input size/tile count/stride/timing/device/backend/khóa hay thử | §10.1 |
| I346 — Màn hình kiểm thử một ảnh — Tải ảnh kết quả và prediction JSON | §10.1 |
| I347 — Hiển thị độ chính xác đúng nghĩa — YOLO confidence không là accuracy ảnh hay xác suất đã hiệu chuẩn; anomaly score không tự thành % đúng | §10.2; DECISIONS R4,H2 |
| I348 — Hiển thị độ chính xác đúng nghĩa — Benchmark từ report khóa kèm dataset/split/N/model version; thiếu report ghi Chưa đánh giá | §10.2; DECISIONS R4,H2 |
| I349 — Hiển thị độ chính xác đúng nghĩa — Upload không nhãn: Chưa có nhãn chuẩn để đánh giá đúng/sai | §10.2; DECISIONS R4,H2 |
| I350 — Hiển thị độ chính xác đúng nghĩa — Có GT: so ảnh/box; một ảnh đúng không là 100% accuracy hệ thống | §10.2; DECISIONS R4,H2 |
| I351 — Hiển thị độ chính xác đúng nghĩa — Thẻ Balanced accuracy, recall, FRR, F1, ECPB giả định được duyệt, p95 | §10.2; DECISIONS R4,H2 |
| I352 — Hiển thị độ chính xác đúng nghĩa — Chỉnh ngưỡng: thẻ lịch sử vẫn thuộc cấu hình gốc; không gán cho cấu hình mới | §10.2; DECISIONS R4,H2 |
| I353 — Màn hình kiểm thử batch — Nhiều ảnh hoặc dataset/manifest đăng ký; filename không là GT | §10.3 |
| I354 — Màn hình kiểm thử batch — Tiến độ, GOOD/DEFECT/REVIEW/ERROR, route_rate, thời gian, bảng từng ảnh | §10.3 |
| I355 — Màn hình kiểm thử batch — Có nhãn hợp lệ: confusion matrix/metrics/lọc FP/FN; thiếu nhãn chỉ tổng hợp dự đoán | §10.3 |
| I356 — Màn hình kiểm thử batch — Xem lại, tải CSV/JSON/overlay; upload không ghi vào split train/test | §10.3 |
| I357 — Màn hình benchmark và so sánh — Bảng §7 lọc dataset/protocol/device/backend/seed/mode; không xếp điều kiện không tương thích | §10.4 |
| I358 — Màn hình benchmark và so sánh — H00 vs H01 cùng bảng; cùng sample ID, disagreement, rescued defect, added false reject | §10.4 |
| I359 — Màn hình benchmark và so sánh — Confusion matrix, AP/recall từng lớp, worst-group, route_rate, DET/ROC | §10.4 |
| I360 — Màn hình benchmark và so sánh — Latency–recall OR/cascade; chất lượng–latency–memory; xuất bảng/hình | §10.4 |
| I361 — Màn hình benchmark và so sánh — Benchmark chính thức đọc frozen config; thử nghiệm có run_id riêng | §10.4 |
| I362 — Màn hình model và lịch sử — Artifact/schema/dữ liệu và ngày train/ngưỡng/số seed/trạng thái benchmark | §10.5 |
| I363 — Màn hình model và lịch sử — Tra lịch sử ảnh/run theo thời gian/model/kết quả/lỗi hệ thống | §10.5 |
| I364 — Màn hình model và lịch sử — Nhãn nhập tay tách GT xác minh; không tự đưa leaderboard hay retrain | §10.5 |

## Kiểm thử và nghiệm thu (§11)

| Tên phần tử | § nguồn |
| --- | --- |
| I365 — Kiểm thử ưu tiên — Data isolation: AD good-only, không auto-split trộn partition/pair/group | §11.1; contract Step 2–4 |
| I366 — Kiểm thử ưu tiên — Routing: YOLO+ không gọi AD kể cả FP; OR gọi hai; equality threshold; failure không GOOD | §11.1; contract Step 2–4 |
| I367 — Kiểm thử ưu tiên — TileManager linear stitch không sọc, tọa độ/clip/NMS .45 đúng; unit riêng | §11.1; contract Step 2–4 |
| I368 — Kiểm thử ưu tiên — Resize/padding/tiling round-trip đúng tọa độ gốc | §11.1; contract Step 2–4 |
| I369 — Kiểm thử ưu tiên — Metric với TP/TN/FP/FN tính tay; mẫu số0/khôngscore/khôngmask=N/A; Point-in-Box có area constraint | §11.1; contract Step 2–4 |
| I370 — Kiểm thử ưu tiên — Provenance: không trộn threshold/checkpoint, cache/preprocess, report/dataset | §11.1; contract Step 2–4 |
| I371 — Kiểm thử ưu tiên — Reload/export trong tolerance; báo sai khác export riêng | §11.1; contract Step 2–4 |
| I372 — Kiểm thử ưu tiên — UI integration khớp CLI: YOLO-positive, AD-positive, OR, GOOD, ERROR; chỉnh tau không ghi đè benchmark, ảnh hỏng/không nhãn thông báo đúng | §11.1; contract Step 2–4 |
| I373 — Kiểm thử ưu tiên — Ưu tiên lỗi làm sai kết luận; không viết test phản chiếu từng dòng UI | §11.1; contract Step 2–4 |
| I374 — Nghiệm thu cuối — Audit đạt, giữ sáu lớp và bốn partition | §11.2 |
| I375 — Nghiệm thu cuối — YOLO / AD / H00 / H01 chạy độc lập với artifact đầy đủ; log routing đúng | §11.2 |
| I376 — Nghiệm thu cuối — Có baseline đơn + benchmark chung + OR, không chỉ một số hybrid | §11.2 |
| I377 — Nghiệm thu cuối — Ba seed cho kết luận chính hoặc giải thích giới hạn; provenance truy được, không tuning test | §11.2 |
| I378 — Nghiệm thu cuối — UI upload/batch/box/heatmap/routing/version/latency/export hoạt động | §11.2 |
| I379 — Nghiệm thu cuối — Accuracy có nguồn; không gọi score là accuracy, không pixel metric khi không mask, không gán lớp cho AD | §11.2 |
| I380 — Nghiệm thu cuối — Limitations N=2/dedicated line/AD khóa khi YOLO+; README tái chạy, trade-off và chọn model hoặc chưa đạt | §11.2 |

## Gate 0, ba cổng, lộ trình và thứ tự ưu tiên (§12)

| Tên phần tử | § nguồn |
| --- | --- |
| I381 — Gate 0 — tiền điều kiện cloud GPU — Colab / Kaggle, CUDA 12.x; smoke-train 3 epoch thành công | §5 Bước 0; §12 |
| I382 — Gate 0 — tiền điều kiện cloud GPU — Chưa đạt: chưa đếm 4–6 tuần, hoãn train nặng Bước 3–7 | §5 Bước 0; §12 |
| I383 — Gate 0 — tiền điều kiện cloud GPU — Local CLI/UI/EDA/audit; smoke CPU kỹ thuật được contract Step 3/4 cho phép riêng | §5 Bước 0; §12 |
| I384 — Cổng 1 — Sau audit — Dữ liệu đủ hợp lệ để train thì qua cổng | §12; §5 Bước 1 |
| I385 — Cổng 1 — Sau audit — Không đạt: sửa phiên bản dữ liệu mới, không âm thầm thay release khóa | §12; §5 Bước 1 |
| I386 — Cổng 2 — Sau development — Hybrid có lợi ích đáng chi phí? So trực tiếp H00 / H01 | §12; §7.6; DECISIONS S4,H3 |
| I387 — Cổng 2 — Sau development — Không đủ lợi ích: giữ kết quả âm; chọn OR hoặc baseline phù hợp hơn | §12; §7.6; DECISIONS S4,H3 |
| I388 — Cổng 2 — Sau development — Không mặc định giả thuyết hybrid đúng; trước khóa danh sách đánh giá cuối | §12; §7.6; DECISIONS S4,H3 |
| I389 — Cổng 3 — Sau test — Kết luận trong DeepPCB / protocol / phần cứng đã đo | §12; §1.4 |
| I390 — Cổng 3 — Sau test — Chuyển sang camera: cần thu thập và xác nhận dữ liệu thực riêng | §12; §1.4 |
| I391 — Lộ trình 4–6 tuần sau Gate 0 — Tuần0: Colab/Kaggle CUDA12.x + smoke3epoch | §12 |
| I392 — Lộ trình 4–6 tuần sau Gate 0 — Tuần1: Bước0–2 local, smoke, sườn UI, TileManager | §12 |
| I393 — Lộ trình 4–6 tuần sau Gate 0 — Tuần2: cloud YOLO11n + AD-S; Bước3–5, cascade seed42, tau phát triển, UI một ảnh | §12 |
| I394 — Lộ trình 4–6 tuần sau Gate 0 — Tuần3: YOLO11s, anomaly đối chứng, OR, tile/resize, development và shortlist | §12 |
| I395 — Lộ trình 4–6 tuần sau Gate 0 — Tuần4: seed43/44, freeze, test cuối, worst-group, benchmark chính thức | §12 |
| I396 — Lộ trình 4–6 tuần sau Gate 0 — Tuần5: UI batch/compare/DET, profiling, export khi cần | §12 |
| I397 — Lộ trình 4–6 tuần sau Gate 0 — Tuần6: dự phòng, tái lập, phân tích lỗi, tài liệu/demo | §12 |
| I398 — Lộ trình 4–6 tuần sau Gate 0 — Lịch tổ chức phụ thuộc GPU/kinh nghiệm/biến thể, không là thời gian train đã đo | §12 |
| I399 — Thứ tự ưu tiên khi thiếu thời gian — Audit → YOLO11n → EfficientAD-S tile256 → OR H00 → cascade H01 → evaluator → UI → ba seed/test | §12 |
| I400 — Thứ tự ưu tiên khi thiếu thời gian — Lùi B04/B06/B07, export và nghiên cứu lỗi mới; B05 không thuộc ma trận bắt buộc | §12 |
| I401 — Các mục HUMAN / VERIFY còn mở — HUMAN H1: LOGO-CV còn câu hỏi lịch sử; quyết định R1 hiện hành không làm | §1.4; §7.6; §12; DECISIONS §3–4 |
| I402 — Các mục HUMAN / VERIFY còn mở — HUMAN H2: trọng số ECPB; HUMAN H3: chấp nhận hybrid có thể thua YOLO | §1.4; §7.6; §12; DECISIONS §3–4 |
| I403 — Các mục HUMAN / VERIFY còn mở — VERIFY V1: detector mở rộng; V2: bbox nhỏ tile vs resize; V3: route_rate thực | §1.4; §7.6; §12; DECISIONS §3–4 |
| I404 — Các mục HUMAN / VERIFY còn mở — HUMAN H1–H3 của DECISIONS khác giả thuyết H1–H5 của plan; không tự chốt | §1.4; §7.6; §12; DECISIONS §3–4 |

## Nguồn kỹ thuật và cách sử dụng (§13)

| Tên phần tử | § nguồn |
| --- | --- |
| I405 — Nguồn kỹ thuật và trạng thái kế hoạch — DeepPCB của tác giả (commit nguồn pin trong dataset card); bài báo EfficientAD arXiv:2303.14535 | §13 |
| I406 — Nguồn kỹ thuật và trạng thái kế hoạch — API anomalib EfficientAD và Ultralytics YOLO11: đối chiếu phiên bản thư viện đã khóa | §13 |
| I407 — Nguồn kỹ thuật và trạng thái kế hoạch — Nguồn dùng kiểm dữ liệu/thuật toán/API; không lấy điểm số nguồn làm kết quả PCB dự án | §13 |
| I408 — Nguồn kỹ thuật và trạng thái kế hoạch — Cấu trúc/UI/protocol/tiêu chí là thiết kế; sơ đồ không tuyên bố đã có ứng dụng hay benchmark | §13 |
| I409 — Nguồn kỹ thuật và trạng thái kế hoạch — Đọc PLAN.md §1–13, DECISIONS, TRIAGE, contracts Step2/3/4; không dùng docs/archive | §13 |

Tổng kiểm kê: **409 mục**, thuộc **99 nút nội dung**. Các node có thể gộp nhiều dòng nhưng từng dòng đều được ánh xạ trong COVERAGE.
