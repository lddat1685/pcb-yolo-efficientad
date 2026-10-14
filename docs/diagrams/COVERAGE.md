# Đối chiếu độ bao phủ PLAN_OVERVIEW

Đối chiếu từng dòng của [INVENTORY.md](INVENTORY.md) với nhãn nút trong [PLAN_OVERVIEW.mmd](PLAN_OVERVIEW.mmd). Nút nhóm được phép chứa nhiều phần tử; tên biến, metrics và trường record giữ nguyên. Căn cứ và khác biệt nguồn xem [NOTES.md](NOTES.md).

| Phần tử trong INVENTORY | Nằm ở nút nào | Có/không |
| --- | --- | --- |
| I001 — Nguồn dữ liệu canonical — DATASET_ROOT = ../DatasetVer4_Public; chỉ đọc | SOURCE | có |
| I002 — Nguồn dữ liệu canonical — DeepPCB: 2.998 ảnh RGB/grayscale 640×640; 1.498 good + 1.500 defect; 11 source groups | SOURCE | có |
| I003 — Nguồn dữ liệu canonical — README, DATASET_CARD, status.json; manifest samples.jsonl; classes.json; holdout.json; protocol.json | SOURCE | có |
| I004 — Nguồn dữ liệu canonical — build_summary.json, verification.json, release.json; 18/18 tệp release đã kiểm SHA-256 trong plan | SOURCE | có |
| I005 — Nguồn dữ liệu canonical — Không glob, không chia ngẫu nhiên theo ảnh; không tự ghép template; hash ảnh/pixel/group/pair không giao partition | SOURCE | có |
| I006 — Nguồn dữ liệu canonical — Không sửa protocol/release nguồn; output, cache và YAML nằm trong dự án riêng | SOURCE | có |
| I007 — Partition train — 1.792 ảnh = 895 good + 897 defect; 5 nhóm nguồn | TR | có |
| I008 — Partition train — YoloTrainSet: toàn bộ train; AdTrainSet: chỉ 895 good | TR | có |
| I009 — Partition train — Đọc manifest, sắp sample_id tất định; mọi tile của cùng ảnh giữ cùng partition | TR | có |
| I010 — Partition calibration — 460 ảnh = 230 good + 230 defect; 2 nhóm nguồn | CA | có |
| I011 — Partition calibration — Chọn checkpoint và ngưỡng; 230 good để chuẩn hóa map AD | CA | có |
| I012 — Partition calibration — Toàn bộ 230 defect đi qua AD độc lập để hiệu chỉnh tau_ad; không chỉ mẫu routed | CA | có |
| I013 — Partition fusion — 306 ảnh = 153 good + 153 defect; 2 nhóm nguồn | FU | có |
| I014 — Partition fusion — Alias development_selection; cùng mẫu và cùng thứ tự | FU | có |
| I015 — Partition fusion — So sánh, chọn cấu hình sau calibration; không fit logistic fusion head trong baseline | FU | có |
| I016 — Taxonomy sáu lớp — 0 open_circuit — đứt mạch; bbox train/final: 1.216/298 | CLASSES | có |
| I017 — Taxonomy sáu lớp — 1 short — chập mạch; bbox train/final: 973/215 | CLASSES | có |
| I018 — Taxonomy sáu lớp — 2 mouse_bite — khuyết mép; bbox train/final: 1.228/250 | CLASSES | có |
| I019 — Taxonomy sáu lớp — 3 spur — gai đồng; bbox train/final: 981/228 | CLASSES | có |
| I020 — Taxonomy sáu lớp — 4 spurious_copper — đồng thừa; bbox train/final: 884/194 | CLASSES | có |
| I021 — Taxonomy sáu lớp — 5 pin_hole — lỗ nhỏ; bbox train/final: 865/222 | CLASSES | có |
| I022 — Taxonomy sáu lớp — Đa box, đa lớp trong một ảnh; pin_hole không đồng nhất missing_hole PKU | CLASSES | có |
| I023 — Giới hạn dữ liệu và kết luận — Chỉ bbox, không mask chuẩn: pixel AUROC / Dice / IoU segmentation / AUPRO không được công bố | LIMITS | có |
| I024 — Giới hạn dữ liệu và kết luận — Good template đã làm sạch, defect có bổ sung thủ công; kiểm shortcut và dấu vết xử lý | LIMITS | có |
| I025 — Giới hạn dữ liệu và kết luận — source_group chưa xác minh là bo/design vật lý; đơn vị kết quả là ảnh/crop | LIMITS | có |
| I026 — Giới hạn dữ liệu và kết luận — N=2 nhóm cho mỗi tập ngoài train; dedicated line, không universal zero-shot AOI | LIMITS | có |
| I027 — Giới hạn dữ liệu và kết luận — Bootstrap không chứng minh domain shift; không LOGO-CV 11 groups | LIMITS | có |
| I028 — Giới hạn dữ liệu và kết luận — Hybrid chưa chứng minh tốt hơn YOLO đơn | LIMITS | có |
| I029 — Phạm vi và sản phẩm cần bàn giao — Core AI kiểm thử ảnh PCB trần có sẵn; không yêu cầu camera, robot hoặc phân loại vật lý | SCOPE | có |
| I030 — Phạm vi và sản phẩm cần bàn giao — Không phát triển lại kiến trúc ba nhánh; YOLO11n+AD-S là cấu hình xuất phát chưa xác minh tốt nhất | SCOPE | có |
| I031 — Phạm vi và sản phẩm cần bàn giao — Dataset adapter + báo cáo audit tái chạy; baseline checkpoint + cấu hình đầy đủ | SCOPE | có |
| I032 — Phạm vi và sản phẩm cần bàn giao — OR/cascade hai tầng + ngưỡng khóa; benchmark chung + prediction từng ảnh | SCOPE | có |
| I033 — Phạm vi và sản phẩm cần bàn giao — UI upload/bbox/heatmap/routing/batch/compare; gói chạy lại, hướng dẫn, báo cáo chọn model | SCOPE | có |
| I034 — Giả thuyết H1–H5 cần kiểm chứng — H1: AD cứu một phần defect YOLO không có box đạt ngưỡng | HYPOTH | có |
| I035 — Giả thuyết H1–H5 cần kiểm chứng — H2: recall tăng đáng chi phí false reject và tầng hai | HYPOTH | có |
| I036 — Giả thuyết H1–H5 cần kiểm chứng — H3: resolution/tiling ảnh hưởng lỗi nhỏ, phải đo | HYPOTH | có |
| I037 — Giả thuyết H1–H5 cần kiểm chứng — H4: cân bằng trên nhóm nguồn chưa thấy và nhiều seed | HYPOTH | có |
| I038 — Giả thuyết H1–H5 cần kiểm chứng — H5: H01 recall tương đương H00 nhưng latency và/hoặc FRR tốt hơn — mục tiêu chưa chứng minh | HYPOTH | có |
| I039 — Giả thuyết H1–H5 cần kiểm chứng — Sáu lớp đã biết không chứng minh mọi lỗi mới; mở rộng cần protocol riêng | HYPOTH | có |
| I040 — Một core — InferenceService — Chọn mode: yolo_only / efficientad_only / or / cascade / diagnostic_both | CORE | có |
| I041 — Một core — InferenceService — Nạp artifact đã đăng ký; preprocessing và output thống nhất cho CLI, UI, benchmark | CORE | có |
| I042 — Một core — InferenceService — YOLO không có box đạt ngưỡng không đồng nghĩa ảnh không lỗi; box thấp lưu được để phân tích | CORE | có |
| I043 — YOLO inference — Engine.infer RGB uint8 HWC → N×6 (xyxy, confidence, class) | YINFER | có |
| I044 — YOLO inference — YoloDetector.predict trả Detection tọa độ gốc, sort confidence/class_id/xyxy | YINFER | có |
| I045 — YOLO inference — accepted_boxes: đúng schema và confidence ≥ tau_yolo | YINFER | có |
| I046 — YOLO inference — yolo_image_score = max confidence trước ngưỡng; 0 khi không có candidate | YINFER | có |
| I047 — EfficientAD inference — EfficientAdEngine.infer_tiles: RGB uint8 NHWC → float32 [N,256,256] | ADINFER | có |
| I048 — EfficientAD inference — Teacher / student / autoencoder + stats đã khóa; ghép toàn ảnh qua TileManager | ADINFER | có |
| I049 — EfficientAD inference — Không cần template tham chiếu; không chọn ROI bằng ground truth | ADINFER | có |
| I050 — EfficientAD inference — anomaly_score = max(stitched map); DEFECT nếu anomaly_score ≥ tau_ad | ADINFER | có |
| I051 — EfficientAD inference — Vùng bất thường: anomaly_unclassified, class_id=None; không phải lớp GT thứ bảy | ADINFER | có |
| I052 — H01 / H02 — Cascade — H01 = B01 + B03; H02 = B02 + B03, bắt buộc vòng sàng lọc | CASCADE | có |
| I053 — H01 / H02 — Cascade — Cùng checkpoint/ngưỡng: giữ mọi ảnh YOLO đã báo lỗi; recall ảnh không chứng minh đủ mọi box | CASCADE | có |
| I054 — H01 / H02 — Cascade — Chỉ gọi AD khi accepted_boxes rỗng; không sửa được false positive của YOLO | CASCADE | có |
| I055 — YOLO dương → DEFECT — decision_source=YOLO; efficientad_executed=false | YEXIT | có |
| I056 — YOLO dương → DEFECT — Kể cả FP: kết thúc cascade, KHÔNG gọi EfficientAD | YEXIT | có |
| I057 — YOLO dương → DEFECT — AD bị khóa nên không cứu good bị báo nhầm, không tìm lỗi thứ hai bị sót | YEXIT | có |
| I058 — H00 — OR chạy cả hai — Mỗi ảnh gọi cả YOLO và AD; cùng checkpoint B01+B03 như H01 | ORRUN | có |
| I059 — H00 — OR chạy cả hai — DEFECT nếu YOLO dương HOẶC anomaly_score ≥ tau_ad; ngược lại GOOD | ORRUN | có |
| I060 — H00 — OR chạy cả hai — decision_source=OR; AD route=100%; không early-exit | ORRUN | có |
| I061 — H00 — OR chạy cả hai — Giá trị recall / FRR / latency là kết quả cần đo; H5 không là kết luận đã xác minh | ORRUN | có |
| I062 — diagnostic_both — Chạy cả YOLO và AD mọi ảnh để phân tích bù trừ đa lỗi | DIAG | có |
| I063 — diagnostic_both — Không lấy latency diagnostic_both làm latency cascade | DIAG | có |
| I064 — Nghiên cứu AD ngoài bbox — Background-only / vùng ngoài box là optional hoặc ablation | BGDIAG | có |
| I065 — Nghiên cứu AD ngoài bbox — Không thay routing toàn ảnh của cascade cơ bản | BGDIAG | có |
| I066 — Quyết định từ AD — Score ≥ tau_ad: DEFECT, decision_source=EFFICIENTAD | ADDEC | có |
| I067 — Quyết định từ AD — Score dưới tau_ad: GOOD — chưa phát hiện bất thường | ADDEC | có |
| I068 — Kết quả và lỗi — GOOD / DEFECT cùng decision_source, routing_reason, score/ngưỡng và thời gian | RESULT | có |
| I069 — Kết quả và lỗi — Đọc ảnh hỏng, thiếu weights, timeout, inference lỗi: ERROR; không trả GOOD | RESULT | có |
| I070 — Kết quả và lỗi — Không dùng max/trung bình confidence YOLO với anomaly score thô làm score cascade | RESULT | có |
| I071 — REVIEW nếu được bật — Ảnh ngoài phạm vi có thể cảnh báo hoặc REVIEW; anomaly score không mặc nhiên đáng tin cho OOD | REVIEW | có |
| I072 — REVIEW nếu được bật — Coverage, review rate riêng good/defect, defect accepted GOOD, ma trận ba trạng thái | REVIEW | có |
| I073 — REVIEW nếu được bật — Không loại REVIEW để làm đẹp accuracy; giữ bảng binary baseline riêng | REVIEW | có |
| I074 — Sample và loader chuẩn — Sample frozen: sample_id, source_group, pair_id, partition, image_path, width, height, is_defect | SAMPLE | có |
| I075 — Sample và loader chuẩn — boxes[Box(class_id,class_name,xyxy)], mask_status, sha256; load_image trả RGB uint8 HWC | SAMPLE | có |
| I076 — Sample và loader chuẩn — ManifestDataset / YoloTrainSet / AdTrainSet / EvalSet; mỗi ảnh một lần | SAMPLE | có |
| I077 — Sample và loader chuẩn — allow_test=false: metadata được đọc, truy cập pixel partition cuối bị PermissionError | SAMPLE | có |
| I078 — Nhánh YOLO 640 — Letterbox target=640, pad_value=114; lưu scale, pad_left/top, orig_w/h | LETTER | có |
| I079 — Nhánh YOLO 640 — Ảnh 640×640: scale=1, pad=0; unletterbox_boxes + clip về ảnh gốc | LETTER | có |
| I080 — Nhánh YOLO 640 — Round-trip tọa độ sai số không quá 1 px | LETTER | có |
| I081 — Nhánh AD native tile-256 — TileManager: tile_size=256, stride=224, overlap=32, pad_mode=reflect; không tile-320 | TILE | có |
| I082 — Nhánh AD native tile-256 — Số bước: H lớn hơn 256 thì ceil((H−256)/224)+1; còn lại 1 | TILE | có |
| I083 — Nhánh AD native tile-256 — 640×640 thành 9 tile; x0/y0 = 0,224,448; row-major; canvas 704×704, đệm phải/dưới | TILE | có |
| I084 — Nhánh AD native tile-256 — TileSpec: tile_id,row,col,x0,y0,size; split giữ nguyên pixel vùng không đệm | TILE | có |
| I085 — Nhánh AD native tile-256 — AdTrainSet.num_tiles theo plan hình học: 895 × 9 = 8.055 tile | TILE | có |
| I086 — Nhánh AD native tile-256 — Engine RGB uint8 [N,256,256,3]; dataset tensor [9,3,256,256] | TILE | có |
| I087 — TileManager hậu xử lý — stitch float32 tile maps: Linear Blending / feathering, tổng trọng số=1, không chia 0 | STITCH | có |
| I088 — TileManager hậu xử lý — Crop padding: anomaly_map float32 trở về 640×640; image score = max(stitched map) | STITCH | có |
| I089 — TileManager hậu xử lý — boxes_to_global: cộng offset, clip, bỏ box rỗng, giữ score | STITCH | có |
| I090 — TileManager hậu xử lý — Global NMS IoU=0.45; loại khi IoU lớn hơn ngưỡng; thứ tự score giảm dần | STITCH | có |
| I091 — Augmentation chỉ train — Recipe aug_v1 / yolo_aug_v1 có phiên bản; nhẹ và phù hợp PCB, không phá bản chất lỗi | AUG | có |
| I092 — Augmentation chỉ train — Seed tất định; biến đổi bbox đồng bộ ảnh; không mutate input; split khác bị từ chối | AUG | có |
| I093 — Augmentation chỉ train — Thay recipe chỉ qua ablation có tên | AUG | có |
| I094 — Nhánh ablation resize-256 — resize_for_ad bilinear 256; upsample_map bilinear không overshoot | RESIZE | có |
| I095 — Nhánh ablation resize-256 — Bắt buộc đối chứng tile-256 vs resize-256; không thay luồng B03 native | RESIZE | có |
| I096 — Nguồn ảnh và interface chung — BaseImageSource.get_frame trả (sample_id,image); FileImageSource / DirectoryWatcherSource / MockFrameGrabber | IMAGEAPI | có |
| I097 — Nguồn ảnh và interface chung — BaseDetector / BaseAnomalyDetector: load(artifact), predict(image), describe() | IMAGEAPI | có |
| I098 — Nguồn ảnh và interface chung — Detection: class_id, class_name, confidence, xyxy_original | IMAGEAPI | có |
| I099 — Nguồn ảnh và interface chung — EfficientAdResult: anomaly_score, anomaly_map, boxes; AD class_id=None | IMAGEAPI | có |
| I100 — Nguồn ảnh và interface chung — CLI / UI gọi cùng InferenceService; evaluator chỉ nhận output chuẩn, GT không vào inference | IMAGEAPI | có |
| I101 — Cấu trúc dự án: cấu hình và core — PLAN.md, plan.v2.md, plan.v3.md, Review.md, review2.md; docs/TRIAGE.md, docs/DECISIONS.md | TREE1 | có |
| I102 — Cấu trúc dự án: cấu hình và core — README.md, pyproject.toml, requirements khóa theo môi trường | TREE1 | có |
| I103 — Cấu trúc dự án: cấu hình và core — configs/dataset.yaml, protocol.yaml, models/, experiments/ | TREE1 | có |
| I104 — Cấu trúc dự án: cấu hình và core — src/pcb_lab/data/, models/, inference/ (TileManager), evaluation/, registry/ | TREE1 | có |
| I105 — Cấu trúc dự án: ứng dụng và output — app/ Streamlit; scripts/ audit/train/calibrate/benchmark/export; tests/ dữ liệu/routing/metrics | TREE2 | có |
| I106 — Cấu trúc dự án: ứng dụng và output — data_refs/ snapshot manifest và chữ ký, không nhân bản dataset | TREE2 | có |
| I107 — Cấu trúc dự án: ứng dụng và output — artifacts/, runs/, reports/ audit/bảng/hình/phân tích lỗi | TREE2 | có |
| I108 — Cấu trúc dự án: ứng dụng và output — uploads/ ảnh thử riêng không tự vào train; exports/ ZIP, CSV/JSON, ảnh kết quả | TREE2 | có |
| I109 — Runner GPU khi dùng từ xa — notebooks/ Colab/Kaggle hoặc package riêng dự án | NOTEBOOK | có |
| I110 — Runner GPU khi dùng từ xa — Resume và kiểm artifact tải về; không mặc định runner ba nhánh cũ | NOTEBOOK | có |
| I111 — Công nghệ và môi trường — Python, PyTorch, Ultralytics; AD dùng anomalib hoặc standalone | TECH | có |
| I112 — Công nghệ và môi trường — Streamlit tiếng Việt; JSON/JSONL config/metadata/prediction; CSV benchmark | TECH | có |
| I113 — Công nghệ và môi trường — Venv mới; khóa Python, torch/torchvision, CUDA/driver, anomalib, Ultralytics, UI; không dùng latest | TECH | có |
| I114 — Công nghệ và môi trường — Không thay môi trường cũ; kiểm tác dụng phụ loader/audit/packaging nguồn trước tái sử dụng | TECH | có |
| I115 — Công nghệ và môi trường — Runner riêng phục vụ baseline/cascade, output root riêng; không chạy lại kiến trúc ba nhánh | TECH | có |
| I116 — FastAPI khi cần nhiều client — Chỉ bổ sung khi nhiều client hoặc tích hợp hệ thống ngoài | FASTAPI | có |
| I117 — FastAPI khi cần nhiều client — Không bắt buộc frontend/backend riêng | FASTAPI | có |
| I118 — SQLite khi JSONL bất tiện — Lịch sử UI có thể chuyển sang SQLite khi cần | SQLITE | có |
| I119 — Bước 0 — Protocol và môi trường — Vào: release nguồn + ma trận baseline + phần cứng thực có | STEP0 | có |
| I120 — Bước 0 — Protocol và môi trường — Protocol deeppcb_cascade_holdout_v1; fusion → development_selection ghi trong protocol mới | STEP0 | có |
| I121 — Bước 0 — Protocol và môi trường — Đăng ký trước baseline/metrics/seed42,43,44/ngưỡng mục tiêu/ngân sách; experiment manifest | STEP0 | có |
| I122 — Bước 0 — Protocol và môi trường — Kiểm CPU/GPU, RAM/VRAM, dung lượng đĩa, framework/device thực; khóa môi trường local và cloud | STEP0 | có |
| I123 — Bước 0 — Protocol và môi trường — Lưu dataset hash, code revision/snapshot, environment lock; không sửa dự án cũ | STEP0 | có |
| I124 — Bước 0 — Protocol và môi trường — Ra: configs/protocol.yaml, reports/environment.json, reports/experiment_register.md | STEP0 | có |
| I125 — Bước 1 — Audit và EDA — Vào: manifest + release + ảnh/nhãn phát triển; holdout cuối chỉ metadata | AUDIT | có |
| I126 — Bước 1 — Audit và EDA — Kiểm decode/hash/kích thước, ID lớp, bbox, nhãn good rỗng, giao group/pair/hash mọi cặp partition | AUDIT | có |
| I127 — Bước 1 — Audit và EDA — Thống kê ảnh/box/lớp/số lỗi/kích thước/nhóm/tỷ lệ; đo pin_hole và mouse_bite thực tế | AUDIT | có |
| I128 — Bước 1 — Audit và EDA — Overlay train/calibration đủ sáu lớp, lỗi nhỏ và biên; giữ good nền trắng hợp lệ | AUDIT | có |
| I129 — Bước 1 — Audit và EDA — Không dùng filename/split làm feature; không suy luận nhãn từ cặp template/test nguồn | AUDIT | có |
| I130 — Bước 1 — Audit và EDA — Ra: reports/data_audit.json, reports/data_profile.md, overlay phát triển | AUDIT | có |
| I131 — Bước 1 — Audit và EDA — Lỗi nghiêm trọng hoặc thay dữ liệu: tạo phiên bản dữ liệu mới trước khi tiếp tục | AUDIT | có |
| I132 — Bước 2 — Tiền xử lý chung — Vào: Sample / ảnh canonical; L, LA, RGBA chuyển RGB nhất quán, không lẫn BGR | PRE | có |
| I133 — Bước 2 — Tiền xử lý chung — prepare_input: yolo / ad_tile / ad_resize; PreparedInput.pixels uint8 | PRE | có |
| I134 — Bước 2 — Tiền xử lý chung — meta: preprocessing_version/hash, mode, orig_hw, scale/pad hoặc tile_specs, color_order, normalized | PRE | có |
| I135 — Bước 2 — Tiền xử lý chung — normalize float32 đúng một lần; gọi lại dữ liệu normalized thì ValueError | PRE | có |
| I136 — Bước 2 — Tiền xử lý chung — Ra: configs/preprocessing.yaml (prep_v1), adapter, TileManager, reports/loader_check.json | PRE | có |
| I137 — Bước 2 — Tiền xử lý chung — check_loaders.py kiểm n_samples/shape/dtype/min/max/color/normalized/hash; prep_preview.py khớp API/CLI | PRE | có |
| I138 — Bước 3 — Baseline YOLO — Vào: train good + defect bbox; calibration chọn checkpoint bằng ultralytics_fitness | YOLOTRAIN | có |
| I139 — Bước 3 — Baseline YOLO — Pretrained phổ thông có nguồn/phiên bản/SHA-256; YOLO11n trước, YOLO11s đối chứng dung lượng | YOLOTRAIN | có |
| I140 — Bước 3 — Baseline YOLO — imgsz=640, epochs=100, patience=20, AdamW; batch theo VRAM, seed=42 ở vòng đầu | YOLOTRAIN | có |
| I141 — Bước 3 — Baseline YOLO — B01/B02 cùng recipe, chỉ khác model_id/architecture/pretrained; khác batch phải ghi lý do | YOLOTRAIN | có |
| I142 — Bước 3 — Baseline YOLO — Null lr0/batch/workers/augment hoặc khóa lạ: ConfigError; resume rõ ràng, không ghi đè run | YOLOTRAIN | có |
| I143 — Bước 3 — Baseline YOLO — YOLO view: train + calibration, hardlink/copy; labels từ manifest, good rỗng, data.yaml không có test | YOLOTRAIN | có |
| I144 — Bước 3 — Baseline YOLO — Ra: best.pt + last.pt, artifact.json, model_card.md, calibration_per_class.json, validation.md; log loss/mAP/recall từng lớp | YOLOTRAIN | có |
| I145 — Bước 3 — Baseline YOLO — Validation precision/recall tại confidence tối đa F1 có pr_definition/pr_conf; không nhầm ngưỡng vận hành | YOLOTRAIN | có |
| I146 — Bước 4 — B03 EfficientAD-S — Vào: chỉ 895 train good, 0 defect; DataLeakError nếu lẫn defect | ADTRAIN | có |
| I147 — Bước 4 — B03 EfficientAD-S — Teacher pretrained + student + autoencoder; học sai khác đặc trưng và tái tạo | ADTRAIN | có |
| I148 — Bước 4 — B03 EfficientAD-S — PyTorch; anomalib hoặc standalone; model_size=small, batch=1, 70 epoch tham khảo, AdamW | ADTRAIN | có |
| I149 — Bước 4 — B03 EfficientAD-S — Teacher mean/std chỉ train good; map quantiles 0.90/0.995 chỉ 230 calibration good | ADTRAIN | có |
| I150 — Bước 4 — B03 EfficientAD-S — Không chuẩn hóa ImageNet hai lần; tắt auto-threshold / auto-split / cập nhật ngưỡng tự động | ADTRAIN | có |
| I151 — Bước 4 — B03 EfficientAD-S — Null lr/weight_decay/device chưa duyệt hoặc khóa lạ: từ chối; seed=42, deterministic=true | ADTRAIN | có |
| I152 — B03 — EfficientAD-S (tile 256) — model.pt chứa weights teacher/student/autoencoder, normalization_stats và tile/inference params | B03 | có |
| I153 — B03 — EfficientAD-S (tile 256) — artifact.json: normalization_stats; model_card.md; calibration_stats.json: phân phối score good/defect; báo cáo false positive | B03 | có |
| I154 — Bước 4 — B03 EfficientAD-S — Fit → save → reload → predict: map/score sai số không quá 1e-5; ghi optimizer steps thực | ADTRAIN | có |
| I155 — Tài nguyên regularization khi dùng anomalib — Kiểm pretrained teacher và ImageNette theo triển khai đã khóa | IMAGENETTE | có |
| I156 — Tài nguyên regularization khi dùng anomalib — ImageNette phục vụ regularization, không cộng vào 895 PCB good | IMAGENETTE | có |
| I157 — Trích dự đoán thô phát triển — preds_calibration.jsonl: 460 dòng; preds_fusion.jsonl: 306 dòng; mỗi sample một dòng, sort sample_id | EXTRACT | có |
| I158 — Trích dự đoán thô phát triển — YOLO conf_floor=0.001, NMS iou=0.7, max_det=300, agnostic_nms=false, half=false | EXTRACT | có |
| I159 — Trích dự đoán thô phát triển — Candidate floor độc lập tau_yolo; mAP confidence sweep tách recall tại ngưỡng vận hành | EXTRACT | có |
| I160 — Trích dự đoán thô phát triển — Test bị PermissionError; lỗi giữ dòng/error_reason, score=null, không thành GOOD/0 | EXTRACT | có |
| I161 — Trích dự đoán thô phát triển — timing_ms=null trong raw extraction; không dùng cache prediction để đo latency | EXTRACT | có |
| I162 — Bước 5 — Hiệu chỉnh có điều kiện — Vào: score toàn calibration, gồm toàn bộ defect qua AD độc lập | CALGRID | có |
| I163 — Bước 5 — Hiệu chỉnh có điều kiện — 2-D Grid Search (tau_yolo,tau_ad): [0.1,0.9] × [mu_ad,mu_ad+3sigma_ad] | CALGRID | có |
| I164 — Bước 5 — Hiệu chỉnh có điều kiện — mu_ad/sigma_ad từ phân phối defect calibration; lưu toàn bộ điểm lưới | CALGRID | có |
| I165 — Bước 5 — Hiệu chỉnh có điều kiện — Tối đa Defect Recall với FRR ≤5%; mốc 1% là phân tích phụ; hòa thì ưu tiên latency thấp | CALGRID | có |
| I166 — Bước 5 — Hiệu chỉnh có điều kiện — Ngưỡng tái tạo được; không fit logistic fusion head | CALGRID | có |
| I167 — Phân biệt chuẩn hóa và ngưỡng — normalization_stats: source_split=calibration, n_good_images=230, quantile_min/max, mean/std | NORMAUDIT | có |
| I168 — Phân biệt chuẩn hóa và ngưỡng — Map quantiles 0.90/0.995 tính SAU khi fit trên 230 calibration good; tau_ad dùng good/defect để calibration | NORMAUDIT | có |
| I169 — Phân biệt chuẩn hóa và ngưỡng — Theo dõi riêng good routed, cỡ mẫu và phân phối; không ước lượng cực trị quá mức | NORMAUDIT | có |
| I170 — Artifact ngưỡng và routing — Ra: thresholds.json, routing_policy.json, báo cáo calibration/development selection | THRESH | có |
| I171 — Artifact ngưỡng và routing — Fusion chạy H00 và H01 thật: delta_recall, delta_FRR, delta_latency; route_rate good/defect | THRESH | có |
| I172 — Artifact ngưỡng và routing — Đổi checkpoint/resize/tile/score pooling/hậu xử lý: calibration lại và tăng version | THRESH | có |
| I173 — Artifact ngưỡng và routing — Giữ tau_pixel sinh candidate độc lập với tau_ad quyết định ảnh | THRESH | có |
| I174 — Bước 6 — Development và shortlist — Vào: calibration/fusion + evaluator chung + baseline/OR/cascade đã hiệu chỉnh | SHORTLIST | có |
| I175 — Bước 6 — Development và shortlist — Vòng seed42: B01/B02/B03/H00/H01; H02 thuộc ma trận sàng lọc | SHORTLIST | có |
| I176 — Bước 6 — Development và shortlist — Ưu tiên ablation dung lượng YOLO, tile vs resize, ngưỡng, rescue và added false reject | SHORTLIST | có |
| I177 — Bước 6 — Development và shortlist — Chọn shortlist có lý do; chạy thêm seed43/44, không chọn seed đẹp nhất | SHORTLIST | có |
| I178 — Bước 6 — Development và shortlist — Tập báo cáo cuối tối thiểu B01/B03/H00/H01; đối thủ mạnh bổ sung theo ngân sách | SHORTLIST | có |
| I179 — Bước 6 — Development và shortlist — Ra: bảng development, bảng ablation, shortlist và ngân sách thực đã dùng | SHORTLIST | có |
| I180 — Bước 7 — Khóa trước đánh giá cuối — Vào: shortlist và cấu hình demo đã chọn bằng development | FREEZE | có |
| I181 — Bước 7 — Khóa trước đánh giá cuối — frozen_evaluation.json: checkpoint hash, tau, schema, split hash, preprocessing | FREEZE | có |
| I182 — Bước 7 — Khóa trước đánh giá cuối — runtime/backend, seed, metrics, quy tắc chọn model; danh sách phương pháp gồm H00/H01 | FREEZE | có |
| I183 — Bước 7 — Khóa trước đánh giá cuối — Chỉ sau khóa mới mở luồng test; không dùng test để chọn checkpoint/ngưỡng | FREEZE | có |
| I184 — Partition test — chỉ đánh giá cuối — 440 ảnh = 220 good + 220 defect; 220 pair; 2 nhóm group44000/group92000 | TESTFINAL | có |
| I185 — Partition test — chỉ đánh giá cuối — Cùng danh sách canonical cho mọi model đã khóa; mỗi ảnh đúng một lần | TESTFINAL | có |
| I186 — Partition test — chỉ đánh giá cuối — Xuất từng seed / source_group / worst-group; đo pipeline routing thật | TESTFINAL | có |
| I187 — Partition test — chỉ đánh giá cuối — Ra: benchmark_summary.csv, benchmark_per_class.csv, benchmark_per_group.csv, predictions.jsonl | TESTFINAL | có |
| I188 — Partition test — chỉ đánh giá cuối — Không đạt thì báo không đạt; không sửa tau rồi ghi đè; nghiên cứu tiếp cần protocol/tập xác nhận mới | TESTFINAL | có |
| I189 — Partition test — chỉ đánh giá cuối — Phân tích lỗi test sau đánh giá chỉ giải thích hạn chế, không tuyên bố cải tiến trên cùng test | TESTFINAL | có |
| I190 — Bước 8 — Streamlit nội bộ — Vào: core đã kiểm Bước 5 + registry; sườn UI sớm có dữ liệu minh họa được đánh dấu | UIBUILD | có |
| I191 — Bước 8 — Streamlit nội bộ — UI gọi InferenceService qua BaseImageSource, không đọc file trực tiếp | UIBUILD | có |
| I192 — Bước 8 — Streamlit nội bộ — Chỉ load model đăng ký; cache giữa ảnh, trạng thái tải/chạy/lỗi | UIBUILD | có |
| I193 — Bước 8 — Streamlit nội bộ — Benchmark khóa tách thử ngưỡng; mỗi chỉnh tham số tạo run mới, không ghi đè artifact | UIBUILD | có |
| I194 — Bước 8 — Streamlit nội bộ — Ảnh mẫu mặc định calibration/fusion; gallery cuối chỉ trong chế độ xem báo cáo sau khóa | UIBUILD | có |
| I195 — Bước 8 — Streamlit nội bộ — Ra: ứng dụng upload/batch/benchmark; overlay và số liệu khớp core/CLI | UIBUILD | có |
| I196 — Bước 9 — Profile sau baseline đúng — Vào: baseline đúng + artifact đã khóa | RUNTIME | có |
| I197 — Bước 9 — Profile sau baseline đúng — Profile decode/preprocess/YOLO/AD/stitch/NMS/hậu xử lý/render/load | RUNTIME | có |
| I198 — Bước 9 — Profile sau baseline đúng — Giữ model trong RAM, tránh copy tensor thừa; cascade chỉ gọi AD khi routed | RUNTIME | có |
| I199 — Bước 9 — Profile sau baseline đúng — Kiểm backend Intel Arc XPU / CUDA cloud; giữ CPU demo nếu khả thi | RUNTIME | có |
| I200 — Bước 9 — Profile sau baseline đúng — Ra: bảng accuracy–latency–memory theo backend; lợi ích phải đo, sai khác trong mức đã định | RUNTIME | có |
| I201 — Export runtime khi hỗ trợ — FP16 / ONNX / OpenVINO: thử theo phần cứng và khả năng export thực | EXPORTOPT | có |
| I202 — Export runtime khi hỗ trợ — So PyTorch với export trên cùng ảnh: score/box/decision và E2E | EXPORTOPT | có |
| I203 — Export runtime khi hỗ trợ — Artifact version riêng; calibration lại nếu cần rồi khóa lại trước đánh giá | EXPORTOPT | có |
| I204 — Bước 10 — Đóng gói bàn giao — Vào: checkpoint + metadata + benchmark + UI đã kiểm | DELIVERY | có |
| I205 — Bước 10 — Đóng gói bàn giao — README tiếng Việt, model card, dataset/protocol card, limitations và hướng dẫn tái lập | DELIVERY | có |
| I206 — Bước 10 — Đóng gói bàn giao — Demo phát triển: YOLO TP, AD rescue, good đúng, FP/FN tiêu biểu nếu có | DELIVERY | có |
| I207 — Bước 10 — Đóng gói bàn giao — Benchmark OR/cascade, confusion matrix, PR/ROC/DET phù hợp, sáu lớp, Pareto latency–recall, ablation | DELIVERY | có |
| I208 — Bước 10 — Đóng gói bàn giao — Ra: gói chạy / ZIP, báo cáo và demo tái lập; checkpoint đúng quyền sử dụng | DELIVERY | có |
| I209 — Bước 10 — Đóng gói bàn giao — Kiểm cài/chạy từ thư mục dự án; người khác theo README đối chiếu được kết quả | DELIVERY | có |
| I210 — CLI dùng core và evaluator chung — Lệnh audit / train / calibrate / benchmark / UI; cấu hình mẫu | CLI | có |
| I211 — CLI dùng core và evaluator chung — Inference gọi InferenceService; benchmark gọi cùng evaluator như UI | CLI | có |
| I212 — CLI dùng core và evaluator chung — Công bố tác vụ cần cloud GPU và tài nguyên cần tải | CLI | có |
| I213 — Smoke CPU theo hợp đồng — Step 3: 1 epoch, view nhỏ, random từ kiến trúc, không cần pretrained | SMOKE | có |
| I214 — Smoke CPU theo hợp đồng — Step 4: 1 epoch, limit good=4, CPU; không tạo artifact chính thức | SMOKE | có |
| I215 — Smoke CPU theo hợp đồng — runs/smoke, smoke=true; không thay baseline, không chứng nhận Gate 0 cloud | SMOKE | có |
| I216 — Đánh giá công bằng: dữ liệu và lựa chọn — Cùng danh sách canonical, không thêm template riêng và không bỏ ảnh model lỗi | FAIR | có |
| I217 — Đánh giá công bằng: dữ liệu và lựa chọn — Công bố supervision: detector bbox+good/defect, anomaly good-only, OR/cascade kế thừa hai nhánh | FAIR | có |
| I218 — Đánh giá công bằng: dữ liệu và lựa chọn — Calibration chọn checkpoint/ngưỡng; fusion chọn cấu hình; holdout cuối không tham gia lựa chọn | FAIR | có |
| I219 — Đánh giá công bằng: dữ liệu và lựa chọn — Công bố số thử, GPU-hours, external/pretrained data và ngân sách tuning từng model | FAIR | có |
| I220 — Đánh giá công bằng: dữ liệu và lựa chọn — Cùng device/backend/precision/batch/timing mỗi bảng; khác điều kiện tách bảng | FAIR | có |
| I221 — Đánh giá công bằng: dữ liệu và lựa chọn — Công bố input size, tile count/stride; resolution ablation tách kiến trúc khỏi lượng pixel | FAIR | có |
| I222 — Đánh giá công bằng: báo cáo — Upload không nhãn: chỉ prediction/score; accuracy phải kèm dataset/split/N/artifact | FAIRDATA | có |
| I223 — Đánh giá công bằng: báo cáo — Error rate + số mẫu thành công; run thiếu dự đoán không được xếp hạng như run đầy đủ | FAIRDATA | có |
| I224 — Đánh giá công bằng: báo cáo — Dataset khác domain/schema/protocol có bảng riêng; phân biệt frozen zero-shot transfer và retrain/recalibration | FAIRDATA | có |
| I225 — Đánh giá công bằng: báo cáo — Dataset adapter mới khai báo taxonomy, split/group, nhãn good, annotation, nguồn | FAIRDATA | có |
| I226 — Đánh giá công bằng: báo cáo — Tập chỉ defect không đủ FRR / calibrate normal; thiếu điều kiện ghi N/A | FAIRDATA | có |
| I227 — Đánh giá công bằng: báo cáo — Không gộp missing_hole với pin_hole để tạo taxonomy giả | FAIRDATA | có |
| I228 — B01 — YOLO11n — Detector sáu lớp, supervision good + defect bbox; baseline nhanh bắt buộc | B01 | có |
| I229 — B01 — YOLO11n — Checkpoint B01 dùng lại cho H00 và H01, không train lại để đổi đối chứng | B01 | có |
| I230 — B02 — YOLO11s — Detector sáu lớp, supervision good + defect bbox; bắt buộc vòng sàng lọc | B02 | có |
| I231 — B02 — YOLO11s — Checkpoint B02 dùng lại cho H02 | B02 | có |
| I232 — B04 — PatchCore — Good-only train; đối chứng memory bank, nên có nếu đủ ngân sách | B04 | có |
| I233 — B04 — PatchCore — PaDiM đã smoke ở nguồn; tham khảo nếu PatchCore không chạy được | B04 | có |
| I234 — B05 — placeholder, VERIFY V1 — Loại khỏi ma trận bắt buộc; chỉ tái đưa detector mở rộng khi được xác minh | B05 | có |
| I235 — B05 — placeholder, VERIFY V1 — Không đăng ký tên model chưa được kiểm chứng | B05 | có |
| I236 — B06 — EfficientAD-M — Good-only train; mở rộng để so dung lượng anomaly | B06 | có |
| I237 — B07 — FastFlow hoặc STFPM — Good-only train; thêm họ anomaly khi đủ ngân sách | B07 | có |
| I238 — H03 — cấu hình biến thể có căn cứ — YOLO tốt nhất trên development → AD biến thể tốt nhất | H03 | có |
| I239 — H03 — cấu hình biến thể có căn cứ — Ghi rõ hai artifact; chỉ thực hiện khi thay đổi có căn cứ | H03 | có |
| I240 — Bảng tổng hợp và bảng phụ — ID, Model/mode, supervision; Balanced acc., Defect recall/precision, F1, MCC, FRR, Defect→GOOD, ECPB | TABLE | có |
| I241 — Bảng tổng hợp và bảng phụ — E2E p95 ms, RAM/VRAM MB, AD route%; B03/H00=100, YOLO-only=N/A, B04=N/A | TABLE | có |
| I242 — Bảng tổng hợp và bảng phụ — H00 đứng trước H01; B01/B02/B03/H00/H01/H02/B04 theo ma trận ưu tiên | TABLE | có |
| I243 — Bảng tổng hợp và bảng phụ — CSV đầy đủ: dataset_id, protocol_id, artifact/config hash, seed, thiết bị | TABLE | có |
| I244 — Bảng tổng hợp và bảng phụ — —=chưa đo; N/A=không áp dụng; không đưa điểm số nguồn vào bảng dự án | TABLE | có |
| I245 — Bảng tổng hợp và bảng phụ — Bảng detection sáu lớp và bảng localization riêng; §7.8 là nhãn giữ số, không có nội dung bổ sung | TABLE | có |
| I246 — Một evaluator dùng chung — CLI / UI / benchmark cùng evaluator; chỉ tại đây prediction gặp ground truth | EVALUATOR | có |
| I247 — Một evaluator dùng chung — Input: output chuẩn + nhãn hợp lệ; adapter model không nhận GT | EVALUATOR | có |
| I248 — Một evaluator dùng chung — load / predict / describe nhất quán; chưa có nhãn chỉ tổng hợp prediction, không tạo accuracy | EVALUATOR | có |
| I249 — Metrics mức ảnh — DEFECT=1, GOOD=0; TP/TN/FP/FN và số FN tuyệt đối | METRIC | có |
| I250 — Metrics mức ảnh — Balanced accuracy=(TPR+TNR)/2; Accuracy trùng trên 50/50 nên bỏ cột chính | METRIC | có |
| I251 — Metrics mức ảnh — Defect recall / TPR=TP/(TP+FN); Defect→GOOD / miss rate=FN/(TP+FN) | METRIC | có |
| I252 — Metrics mức ảnh — FRR=FP/(FP+TN); precision=TP/(TP+FP) | METRIC | có |
| I253 — Metrics mức ảnh — F1: trung bình điều hòa precision/recall; MCC: tương quan từ confusion matrix | METRIC | có |
| I254 — Metrics mức ảnh — Image AUROC / AUPRC chỉ từ score liên tục có nghĩa, không từ nhãn cứng | METRIC | có |
| I255 — Metrics mức ảnh — Cascade ban đầu AUROC/AUPRC=N/A; score hợp nhất phải là biến thể hiệu chỉnh riêng | METRIC | có |
| I256 — Metrics mức ảnh — Per-class image recall: một ảnh có thể thuộc nhiều lớp; không thay box recall | METRIC | có |
| I257 — ECPB: chi phí mô phỏng — ECPB=(C_FN×FN_sim+C_FP×FP_sim)/N_sim | COST | có |
| I258 — ECPB: chi phí mô phỏng — N_sim=10.000: 100 defect, 9.900 good; đây là mô phỏng 1%/99% | COST | có |
| I259 — ECPB: chi phí mô phỏng — C_FP=1; C_FN cần HUMAN/VERIFY H2, không gắn trọng số chưa duyệt như cam kết nhà máy | COST | có |
| I260 — Giả định trọng số chưa chốt — C_FN=50×C_FP là đề xuất cần xác nhận theo DECISIONS | COSTPENDING | có |
| I261 — Giả định trọng số chưa chốt — Không tự chốt 100:1 từ plan; xem đối chiếu NOTES N02 | COSTPENDING | có |
| I262 — Metrics detection và heatmap — Bảng detection: mAP@0.5, mAP@0.5:0.95, box recall tại ngưỡng, AP/recall sáu lớp, FP box/ảnh good | BOXMETRIC | có |
| I263 — Metrics detection và heatmap — YOLO output của cascade báo riêng; AD N/A cho AP sáu lớp, không ép heatmap thành class | BOXMETRIC | có |
| I264 — Heatmap → candidate boxes R8 — Heatmap stitched → tau_pixel=0.5 → Morph Open k=3 → Connected Components | R8 | có |
| I265 — Heatmap → candidate boxes R8 — Area lớn hơn min_area=20 → Bounding Rect → Global NMS IoU=0.45 | R8 | có |
| I266 — Heatmap → candidate boxes R8 — Box confidence=max pixel trong component; xyxy_original clip ảnh gốc | R8 | có |
| I267 — Metrics detection và heatmap — Point-in-Box: argmax(Map) thuộc BBox_GT và Area_component ≤3×Area_GT | BOXMETRIC | có |
| I268 — Metrics detection và heatmap — Class-agnostic localization ở bảng phụ; không thay mAP hay gọi là segmentation accuracy | BOXMETRIC | có |
| I269 — Metrics hybrid và Iso-FRR — route_rate=ảnh thực chạy AD/tổng ảnh; thêm riêng good/defect | HYBRIDMETRIC | có |
| I270 — Metrics hybrid và Iso-FRR — rescued_defects: YOLO FN được AD cứu; rescue_rate=rescued_defects/FN_YOLO; mẫu số 0 → N/A | HYBRIDMETRIC | có |
| I271 — Metrics hybrid và Iso-FRR — added_false_rejects: good YOLO cho qua nhưng AD báo lỗi; không cứu được FP sẵn có của YOLO | HYBRIDMETRIC | có |
| I272 — Metrics hybrid và Iso-FRR — unclassified_positive_rate=AD DEFECT chưa có loại/tổng DEFECT của pipeline | HYBRIDMETRIC | có |
| I273 — Metrics hybrid và Iso-FRR — delta_recall(H01−H00), delta_FRR(H01−H00), delta_latency(H01−H00), cùng checkpoint | HYBRIDMETRIC | có |
| I274 — Metrics hybrid và Iso-FRR — DET/ROC và Recall @ FRR={1%,3%,5%} cho B01/B03/H00/H01 | HYBRIDMETRIC | có |
| I275 — Metrics hybrid và Iso-FRR — Operating point chọn trên calibration; không chỉnh holdout cuối để ép FRR bằng nhau | HYBRIDMETRIC | có |
| I276 — Đo latency và tài nguyên — Tách model load/cold start và steady-state; warm-up 30, timing 440 ảnh ×3 lượt | TIMING | có |
| I277 — Đo latency và tài nguyên — N chất lượng vẫn 440, không phải 1.320; batch=1 cho tương tác, throughput batch lớn là track riêng | TIMING | có |
| I278 — Đo latency và tài nguyên — Đồng bộ GPU/XPU khi cần; E2E=decode→preprocess→tile→model thực chạy→hậu xử lý | TIMING | có |
| I279 — Đo latency và tài nguyên — UI/encode/render/network đo riêng; không dùng cache score để đo latency routing | TIMING | có |
| I280 — Đo latency và tài nguyên — Mean/p50/p95, từng tầng, peak RAM/VRAM, artifact size, train/fit time, throughput thực | TIMING | có |
| I281 — Đo latency và tài nguyên — Công bố device/backend/precision/input size/tile count; không so 256 và 640 như cùng input | TIMING | có |
| I282 — Đo latency và tài nguyên — Cascade mean≈T_yolo+route_rate×E[T_ad\|routed]+overhead; p95 phải đo trực tiếp | TIMING | có |
| I283 — Đo latency và tài nguyên — OR≈T_yolo+T_ad; latency riêng good/defect, prevalence khác chỉ mô phỏng có chú thích | TIMING | có |
| I284 — Bất định và giới hạn so sánh — Ít nhất ba seed42/43/44: từng seed và mean±std; seed std không là CI tổng quát hóa | UNCERT | có |
| I285 — Bất định và giới hạn so sánh — TP/TN/FP/FN theo source_group và worst-group recall/FRR | UNCERT | có |
| I286 — Bất định và giới hạn so sánh — 1 FP hoặc FN trên nhóm 220 ảnh tương ứng khoảng 0,45 điểm phần trăm | UNCERT | có |
| I287 — Bất định và giới hạn so sánh — Calibration chọn checkpoint lẫn ngưỡng có thể lạc quan; fusion sau selection không còn độc lập | UNCERT | có |
| I288 — Bất định và giới hạn so sánh — Không gọi quy trình là hiệu chuẩn có bảo đảm thống kê; N=2 không chứng minh chuyển miền | UNCERT | có |
| I289 — Bootstrap pair-level nếu dùng — Chỉ sampling variance nội bộ 2 nhóm; không ngoại suy domain shift | BOOT | có |
| I290 — Bootstrap pair-level nếu dùng — Không thay thế đa dạng dữ liệu; không LOGO-CV 11 groups | BOOT | có |
| I291 — Quy tắc chọn model — Trên development: đạt ràng buộc FRR trước, tối đa recall, rồi p95 và memory thấp | SELECT | có |
| I292 — Quy tắc chọn model — Pareto chất lượng–latency–memory, không tự gán điểm tổng hợp thiếu cơ sở | SELECT | có |
| I293 — Quy tắc chọn model — Mốc kế hoạch recall ≥95%, FRR ≤5%; khóa ngân sách latency sau biết phần cứng demo | SELECT | có |
| I294 — Quy tắc chọn model — Muốn đủ sáu lớp/định vị: cần detector AP/box recall, không thay bằng AD-only binary | SELECT | có |
| I295 — Quy tắc chọn model — Nếu không đạt: báo khoảng cách/trade-off; hybrid không đủ lợi ích thì chọn baseline | SELECT | có |
| I296 — Quy tắc chọn model — So H00/H01 tại cùng điều kiện; giữ kết quả âm, không mặc định hybrid tốt nhất | SELECT | có |
| I297 — Prediction record: định danh và provenance — run_id, sample_id, image_sha256, dataset_id, split, source_group | RECIDENT | có |
| I298 — Prediction record: định danh và provenance — model_id, model_version, checkpoint_hashes, protocol_id, config_hash | RECIDENT | có |
| I299 — Prediction record: định danh và provenance — mode, final_status, decision_source, routing_reason, error_reason | RECIDENT | có |
| I300 — Prediction record: thực thi và điểm — yolo_executed, efficientad_executed, yolo_image_score, anomaly_score | RECSCORE | có |
| I301 — Prediction record: thực thi và điểm — tau_yolo, tau_ad, score_definition, preprocessing_version, tiling_params | RECSCORE | có |
| I302 — Prediction record: thực thi và điểm — Tầng không chạy: score và model timing=null, không phải 0 | RECSCORE | có |
| I303 — Prediction record: hình học và output — detections[{class_id,class_name,confidence,xyxy_original}] | RECGEOM | có |
| I304 — Prediction record: hình học và output — anomaly_regions, anomaly_map_path, annotated_image_path | RECGEOM | có |
| I305 — Prediction record: hình học và output — ground_truth_available, ground_truth_reference; GT chỉ dùng tại evaluator | RECGEOM | có |
| I306 — Prediction record: timing và runtime — timing_ms{decode,preprocess,yolo,efficientad,stitch,nms,postprocess,total_core} | RECTIME | có |
| I307 — Prediction record: timing và runtime — device, backend, precision, input_size, tile_count, timestamp | RECTIME | có |
| I308 — Raw predictions Step 3/4 — run_id,sample_id,image_sha256,split,source_group,model_id,checkpoint_sha256,config_hash | RAWSCHEMA | có |
| I309 — Raw predictions Step 3/4 — preprocessing_version; YOLO inference_params,detections,yolo_image_score | RAWSCHEMA | có |
| I310 — Raw predictions Step 3/4 — AD tile_params,anomaly_score,boxes[{xyxy,confidence,class_name}] | RAWSCHEMA | có |
| I311 — Raw predictions Step 3/4 — error_reason; timing_ms=null; schema raw khác record đích §8, không tự đổi tên khóa | RAWSCHEMA | có |
| I312 — Registry và tính tương thích — Giữ checkpoint, threshold, class_order, normalization AD và benchmark đúng artifact | REGISTRY | có |
| I313 — Registry và tính tương thích — load / from_artifact kiểm checkpoint SHA-256, schema/class order, smoke opt-in | REGISTRY | có |
| I314 — Registry và tính tương thích — ArtifactMismatchError / ClassOrderError / SmokeArtifactError; không trộn threshold/cache/report khác nguồn | REGISTRY | có |
| I315 — Registry và tính tương thích — Config hash canonical không chứa seed; preprocessing version/hash có nguồn | REGISTRY | có |
| I316 — Registry và tính tương thích — Không nạp checkpoint PCB cũ thiếu split/schema; pretrained phổ thông có nguồn, version, hash | REGISTRY | có |
| I317 — Artifacts và run theo seed — runs/yolo hoặc runs/efficientad: model_id/seed, run_manifest.json, env.json, train.log, checkpoints | ARTIFACTS | có |
| I318 — Artifacts và run theo seed — artifacts/yolo: best.pt,last.pt,artifact.json,model_card.md,calibration_per_class.json,preds_calibration/fusion | ARTIFACTS | có |
| I319 — Artifacts và run theo seed — artifacts/efficientad: model.pt,artifact.json,model_card.md,calibration_stats.json,preds_calibration/fusion | ARTIFACTS | có |
| I320 — Artifacts và run theo seed — run manifest: schema/run/model/recipe/seed/smoke, git_commit/dirty, versions/device, dataset release hash, args_used | ARTIFACTS | có |
| I321 — Artifacts và run theo seed — epochs/best_epoch/early stop/resume nếu có, train_time, peak RAM/VRAM, checkpoint hashes | ARTIFACTS | có |
| I322 — Artifacts và run theo seed — artifact metadata: normalization/inference/tile params hoặc class_order, config hash, run_manifest path/hash, commit tạo | ARTIFACTS | có |
| I323 — Artifacts và run theo seed — validation report và model card tiếng Việt; scope, supervision, hạn chế N=2, không dùng test | ARTIFACTS | có |
| I324 — Ablation phát triển — YOLO-only / AD-only / OR / cascade: cùng checkpoint/split/log score | ABLATE | có |
| I325 — Ablation phát triển — YOLO11n vs YOLO11s: cùng recipe, công bố ngân sách | ABLATE | có |
| I326 — Ablation phát triển — AD tile-256 vs resize-256 bắt buộc; đo bbox nhỏ, khóa stride/blend/NMS | ABLATE | có |
| I327 — Ablation phát triển — Confidence + AD threshold: 2-D grid search calibration | ABLATE | có |
| I328 — Ablation phát triển — Một seed vs ba seed: cùng split và ngân sách | ABLATE | có |
| I329 — Ablation phát triển — Ưu tiên baseline→routing→threshold→resolution; không đổi mọi biến cùng lúc | ABLATE | có |
| I330 — Ablation pooling mở rộng — Default max score vs top-k: biến thể riêng để kiểm nhiễu cực đại | POOLOPT | có |
| I331 — Ablation pooling mở rộng — Calibration riêng cho mỗi pooling, lưu version riêng | POOLOPT | có |
| I332 — Robustness / stress riêng — Xoay, mờ, đổi sáng, nén: tập phát triển, giữ liên kết ảnh gốc, không trộn benchmark chính | STRESS | có |
| I333 — Robustness / stress riêng — specular_glare — lóa phản xạ; motion_blur — rung lệch trục; defocus_blur — mất nét | STRESS | có |
| I334 — Robustness / stress riêng — alignment_jitter: dx/dy trong [−2,2] px, dtheta trong [−0.5°,0.5°] | STRESS | có |
| I335 — Robustness / stress riêng — Đánh giá jitter riêng B03/H01, kiểm false positive dọc biên do lệch template | STRESS | có |
| I336 — Lỗi chưa thấy — protocol riêng — Leave-one-defect-type-out chỉ khi cần; không chặn MVP sáu lớp | NOVEL | có |
| I337 — Lỗi chưa thấy — protocol riêng — Ảnh đa lỗi chứa lớp giữ lại phải loại khỏi toàn dữ liệu train/tuning có thể lộ lớp | NOVEL | có |
| I338 — Lỗi chưa thấy — protocol riêng — Không chỉ xóa bbox rồi coi là nền; kiểm số ảnh còn lại trước chạy | NOVEL | có |
| I339 — Màn hình kiểm thử một ảnh — Upload PNG/JPG/JPEG; kích thước, dung lượng, ảnh gốc, cảnh báo khác domain/kích thước | UI1 | có |
| I340 — Màn hình kiểm thử một ảnh — Chọn artifact đăng ký + mode; nút chạy/tiến độ; không load lại model khi chỉnh hiển thị | UI1 | có |
| I341 — Màn hình kiểm thử một ảnh — Ảnh gốc/bbox/heatmap stitched 640×640 nếu AD chạy; bật/tắt overlay, opacity, zoom lỗi nhỏ | UI1 | có |
| I342 — Màn hình kiểm thử một ảnh — GOOD / DEFECT / ERROR và REVIEW nếu bật; lý do, decision_source, tuyến xử lý | UI1 | có |
| I343 — Màn hình kiểm thử một ảnh — AD không chạy: hiển thị Không chạy, không heatmap giả | UI1 | có |
| I344 — Màn hình kiểm thử một ảnh — Bảng bbox: class/confidence/xyxy/kích thước; AD score/tau/vùng chưa xác định loại | UI1 | có |
| I345 — Màn hình kiểm thử một ảnh — Model/version/hash rút gọn/tau/input size/tile count/stride/timing/device/backend/khóa hay thử | UI1 | có |
| I346 — Màn hình kiểm thử một ảnh — Tải ảnh kết quả và prediction JSON | UI1 | có |
| I347 — Hiển thị độ chính xác đúng nghĩa — YOLO confidence không là accuracy ảnh hay xác suất đã hiệu chuẩn; anomaly score không tự thành % đúng | UIACCURACY | có |
| I348 — Hiển thị độ chính xác đúng nghĩa — Benchmark từ report khóa kèm dataset/split/N/model version; thiếu report ghi Chưa đánh giá | UIACCURACY | có |
| I349 — Hiển thị độ chính xác đúng nghĩa — Upload không nhãn: Chưa có nhãn chuẩn để đánh giá đúng/sai | UIACCURACY | có |
| I350 — Hiển thị độ chính xác đúng nghĩa — Có GT: so ảnh/box; một ảnh đúng không là 100% accuracy hệ thống | UIACCURACY | có |
| I351 — Hiển thị độ chính xác đúng nghĩa — Thẻ Balanced accuracy, recall, FRR, F1, ECPB giả định được duyệt, p95 | UIACCURACY | có |
| I352 — Hiển thị độ chính xác đúng nghĩa — Chỉnh ngưỡng: thẻ lịch sử vẫn thuộc cấu hình gốc; không gán cho cấu hình mới | UIACCURACY | có |
| I353 — Màn hình kiểm thử batch — Nhiều ảnh hoặc dataset/manifest đăng ký; filename không là GT | UIBATCH | có |
| I354 — Màn hình kiểm thử batch — Tiến độ, GOOD/DEFECT/REVIEW/ERROR, route_rate, thời gian, bảng từng ảnh | UIBATCH | có |
| I355 — Màn hình kiểm thử batch — Có nhãn hợp lệ: confusion matrix/metrics/lọc FP/FN; thiếu nhãn chỉ tổng hợp dự đoán | UIBATCH | có |
| I356 — Màn hình kiểm thử batch — Xem lại, tải CSV/JSON/overlay; upload không ghi vào split train/test | UIBATCH | có |
| I357 — Màn hình benchmark và so sánh — Bảng §7 lọc dataset/protocol/device/backend/seed/mode; không xếp điều kiện không tương thích | UICOMPARE | có |
| I358 — Màn hình benchmark và so sánh — H00 vs H01 cùng bảng; cùng sample ID, disagreement, rescued defect, added false reject | UICOMPARE | có |
| I359 — Màn hình benchmark và so sánh — Confusion matrix, AP/recall từng lớp, worst-group, route_rate, DET/ROC | UICOMPARE | có |
| I360 — Màn hình benchmark và so sánh — Latency–recall OR/cascade; chất lượng–latency–memory; xuất bảng/hình | UICOMPARE | có |
| I361 — Màn hình benchmark và so sánh — Benchmark chính thức đọc frozen config; thử nghiệm có run_id riêng | UICOMPARE | có |
| I362 — Màn hình model và lịch sử — Artifact/schema/dữ liệu và ngày train/ngưỡng/số seed/trạng thái benchmark | UIHISTORY | có |
| I363 — Màn hình model và lịch sử — Tra lịch sử ảnh/run theo thời gian/model/kết quả/lỗi hệ thống | UIHISTORY | có |
| I364 — Màn hình model và lịch sử — Nhãn nhập tay tách GT xác minh; không tự đưa leaderboard hay retrain | UIHISTORY | có |
| I365 — Kiểm thử ưu tiên — Data isolation: AD good-only, không auto-split trộn partition/pair/group | TESTS | có |
| I366 — Kiểm thử ưu tiên — Routing: YOLO+ không gọi AD kể cả FP; OR gọi hai; equality threshold; failure không GOOD | TESTS | có |
| I367 — Kiểm thử ưu tiên — TileManager linear stitch không sọc, tọa độ/clip/NMS .45 đúng; unit riêng | TESTS | có |
| I368 — Kiểm thử ưu tiên — Resize/padding/tiling round-trip đúng tọa độ gốc | TESTS | có |
| I369 — Kiểm thử ưu tiên — Metric với TP/TN/FP/FN tính tay; mẫu số0/khôngscore/khôngmask=N/A; Point-in-Box có area constraint | TESTS | có |
| I370 — Kiểm thử ưu tiên — Provenance: không trộn threshold/checkpoint, cache/preprocess, report/dataset | TESTS | có |
| I371 — Kiểm thử ưu tiên — Reload/export trong tolerance; báo sai khác export riêng | TESTS | có |
| I372 — Kiểm thử ưu tiên — UI integration khớp CLI: YOLO-positive, AD-positive, OR, GOOD, ERROR; chỉnh tau không ghi đè benchmark, ảnh hỏng/không nhãn thông báo đúng | TESTS | có |
| I373 — Kiểm thử ưu tiên — Ưu tiên lỗi làm sai kết luận; không viết test phản chiếu từng dòng UI | TESTS | có |
| I374 — Nghiệm thu cuối — Audit đạt, giữ sáu lớp và bốn partition | ACCEPT | có |
| I375 — Nghiệm thu cuối — YOLO / AD / H00 / H01 chạy độc lập với artifact đầy đủ; log routing đúng | ACCEPT | có |
| I376 — Nghiệm thu cuối — Có baseline đơn + benchmark chung + OR, không chỉ một số hybrid | ACCEPT | có |
| I377 — Nghiệm thu cuối — Ba seed cho kết luận chính hoặc giải thích giới hạn; provenance truy được, không tuning test | ACCEPT | có |
| I378 — Nghiệm thu cuối — UI upload/batch/box/heatmap/routing/version/latency/export hoạt động | ACCEPT | có |
| I379 — Nghiệm thu cuối — Accuracy có nguồn; không gọi score là accuracy, không pixel metric khi không mask, không gán lớp cho AD | ACCEPT | có |
| I380 — Nghiệm thu cuối — Limitations N=2/dedicated line/AD khóa khi YOLO+; README tái chạy, trade-off và chọn model hoặc chưa đạt | ACCEPT | có |
| I381 — Gate 0 — tiền điều kiện cloud GPU — Colab / Kaggle, CUDA 12.x; smoke-train 3 epoch thành công | GATE0 | có |
| I382 — Gate 0 — tiền điều kiện cloud GPU — Chưa đạt: chưa đếm 4–6 tuần, hoãn train nặng Bước 3–7 | GATE0 | có |
| I383 — Gate 0 — tiền điều kiện cloud GPU — Local CLI/UI/EDA/audit; smoke CPU kỹ thuật được contract Step 3/4 cho phép riêng | GATE0 | có |
| I384 — Cổng 1 — Sau audit — Dữ liệu đủ hợp lệ để train thì qua cổng | GATE1 | có |
| I385 — Cổng 1 — Sau audit — Không đạt: sửa phiên bản dữ liệu mới, không âm thầm thay release khóa | GATE1 | có |
| I386 — Cổng 2 — Sau development — Hybrid có lợi ích đáng chi phí? So trực tiếp H00 / H01 | GATE2 | có |
| I387 — Cổng 2 — Sau development — Không đủ lợi ích: giữ kết quả âm; chọn OR hoặc baseline phù hợp hơn | GATE2 | có |
| I388 — Cổng 2 — Sau development — Không mặc định giả thuyết hybrid đúng; trước khóa danh sách đánh giá cuối | GATE2 | có |
| I389 — Cổng 3 — Sau test — Kết luận trong DeepPCB / protocol / phần cứng đã đo | GATE3 | có |
| I390 — Cổng 3 — Sau test — Chuyển sang camera: cần thu thập và xác nhận dữ liệu thực riêng | GATE3 | có |
| I391 — Lộ trình 4–6 tuần sau Gate 0 — Tuần0: Colab/Kaggle CUDA12.x + smoke3epoch | ROADMAP | có |
| I392 — Lộ trình 4–6 tuần sau Gate 0 — Tuần1: Bước0–2 local, smoke, sườn UI, TileManager | ROADMAP | có |
| I393 — Lộ trình 4–6 tuần sau Gate 0 — Tuần2: cloud YOLO11n + AD-S; Bước3–5, cascade seed42, tau phát triển, UI một ảnh | ROADMAP | có |
| I394 — Lộ trình 4–6 tuần sau Gate 0 — Tuần3: YOLO11s, anomaly đối chứng, OR, tile/resize, development và shortlist | ROADMAP | có |
| I395 — Lộ trình 4–6 tuần sau Gate 0 — Tuần4: seed43/44, freeze, test cuối, worst-group, benchmark chính thức | ROADMAP | có |
| I396 — Lộ trình 4–6 tuần sau Gate 0 — Tuần5: UI batch/compare/DET, profiling, export khi cần | ROADMAP | có |
| I397 — Lộ trình 4–6 tuần sau Gate 0 — Tuần6: dự phòng, tái lập, phân tích lỗi, tài liệu/demo | ROADMAP | có |
| I398 — Lộ trình 4–6 tuần sau Gate 0 — Lịch tổ chức phụ thuộc GPU/kinh nghiệm/biến thể, không là thời gian train đã đo | ROADMAP | có |
| I399 — Thứ tự ưu tiên khi thiếu thời gian — Audit → YOLO11n → EfficientAD-S tile256 → OR H00 → cascade H01 → evaluator → UI → ba seed/test | PRIORITY | có |
| I400 — Thứ tự ưu tiên khi thiếu thời gian — Lùi B04/B06/B07, export và nghiên cứu lỗi mới; B05 không thuộc ma trận bắt buộc | PRIORITY | có |
| I401 — Các mục HUMAN / VERIFY còn mở — HUMAN H1: LOGO-CV còn câu hỏi lịch sử; quyết định R1 hiện hành không làm | OPEN | có |
| I402 — Các mục HUMAN / VERIFY còn mở — HUMAN H2: trọng số ECPB; HUMAN H3: chấp nhận hybrid có thể thua YOLO | OPEN | có |
| I403 — Các mục HUMAN / VERIFY còn mở — VERIFY V1: detector mở rộng; V2: bbox nhỏ tile vs resize; V3: route_rate thực | OPEN | có |
| I404 — Các mục HUMAN / VERIFY còn mở — HUMAN H1–H3 của DECISIONS khác giả thuyết H1–H5 của plan; không tự chốt | OPEN | có |
| I405 — Nguồn kỹ thuật và trạng thái kế hoạch — DeepPCB của tác giả (commit nguồn pin trong dataset card); bài báo EfficientAD arXiv:2303.14535 | REFERENCES | có |
| I406 — Nguồn kỹ thuật và trạng thái kế hoạch — API anomalib EfficientAD và Ultralytics YOLO11: đối chiếu phiên bản thư viện đã khóa | REFERENCES | có |
| I407 — Nguồn kỹ thuật và trạng thái kế hoạch — Nguồn dùng kiểm dữ liệu/thuật toán/API; không lấy điểm số nguồn làm kết quả PCB dự án | REFERENCES | có |
| I408 — Nguồn kỹ thuật và trạng thái kế hoạch — Cấu trúc/UI/protocol/tiêu chí là thiết kế; sơ đồ không tuyên bố đã có ứng dụng hay benchmark | REFERENCES | có |
| I409 — Nguồn kỹ thuật và trạng thái kế hoạch — Đọc PLAN.md §1–13, DECISIONS, TRIAGE, contracts Step2/3/4; không dùng docs/archive | REFERENCES | có |

## Phương pháp tự kiểm

- Đọc đủ PLAN.md §1–§13; DECISIONS, TRIAGE; toàn bộ contracts Step 2/3/4 hiện có. §7.8 và §13 cũng có mục riêng.
- Đối chiếu tên và nội dung 409 dòng, không chỉ dò số §. Mỗi dòng phải nằm nguyên ý trong nhãn node đã ghi; không có hàng `không`.
- Rà một flowchart, ID không trùng, mọi đầu cạnh đã định nghĩa, cân bằng subgraph/end, toàn bộ nhãn được quote; dấu góc chỉ dùng cho thẻ b/i/br có chủ ý.
- Rà độc lập các cạnh: train→AD chỉ good; calibration→checkpoint/normalization/tau; fusion→selection; FREEZE→TESTFINAL; YOLO-positive→YEXIT→RESULT không gọi AD; mode OR gọi cả hai nhánh; H00/H01 dùng cùng B01+B03; CLI/UI cùng CORE và EVALUATOR; ground truth chỉ vào evaluator.
- Thành phần tùy chọn có class optional nét đứt và mọi cạnh nối thành phần đó cũng nét đứt. R8 sinh bbox nằm ở inference; đánh giá Point-in-Box và GT nằm ở evaluator.
- **Chưa kiểm cú pháp bằng công cụ Mermaid:** `npx` không được nhận diện, nên không có SVG/PNG và chưa kiểm được bố cục sau render. Rà cấu trúc bằng PowerShell không được coi là parser Mermaid. Xem log và kết quả đối chiếu ở `export/`.

## Đối chiếu các nút/cạnh sau review

| Mục | Nút/cạnh đã đối chiếu | Có/không | Kết quả |
| --- | --- | --- | --- |
| 1 | B04/B06/B07 → CORE; RESULT → EVALUATOR | có | Ba model tùy chọn đi nét đứt vào core, “khi đủ ngân sách”; prediction chuẩn đi vào evaluator. |
| 2 | IMAGENETTE → ADTRAIN | có | Dữ liệu regularization đầu vào, nét đứt khi triển khai cần. |
| 3 | RESIZE → ADINFER; TILE → ADINFER | có | Thêm cạnh ablation resize-256 nét đứt; đường chính B03 native tile-256 giữ nguyên. |
| 4′ | ADTRAIN/CA → NORMAUDIT → B03; I168 | có | Quantiles 0.90/0.995 sau fit trên 230 calibration good; không còn NORMAUDIT → ADTRAIN. Teacher mean/std vẫn từ train good. |
| 5 | ADTRAIN → B03 → ADINFER/EXTRACT/ORRUN/CASCADE; I152–I153 | có | B03 là artifact riêng và nguồn duy nhất đưa checkpoint + normalization vào ADINFER; artifact.json chứa normalization_stats, calibration_stats.json báo cáo phân phối score. |
| 6 | TABLE; I241 | có | AD route% bổ sung B04=N/A. |
| 7′ | YEXIT → RESULT; CORE → ADINFER | có | Tầng không chạy có score và timing=null; efficientad_only tách cạnh khỏi or/diagnostic_both. |
| 8a–b | TESTFINAL → EVALUATOR/CORE | có | GT/sample_id chỉ sang evaluator; ảnh qua core với prepare_input; không thêm TESTFINAL → PRE. |
| 8c | TESTFINAL tới sáu đích yêu cầu | có | 56 đường đơn phân biệt theo cạnh tới SHORTLIST (41 chuỗi nút); 0 đường tới năm đích khác. Chỉ báo cáo, chưa sửa liên thông qua node dùng chung. |

Đường đi, nhãn cạnh và nút dùng chung được liệt kê đầy đủ tại [TESTFINAL_PATHS.md](TESTFINAL_PATHS.md). Độ bao phủ nội dung không thay thế kiểm tra liên thông: các đường đi cấu trúc tới SHORTLIST vẫn đang chờ người dùng quyết định.

**đủ 409/409 mục**
