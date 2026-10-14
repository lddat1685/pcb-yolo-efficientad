# Step 3 — câu hỏi giao diện của VERIFIER

Ngày: 2026-09-25. Chỉ ghi vấn đề; không sửa `step3-contract.md` hay `WORKFLOW.md`.

## Các câu hỏi đã được giám sát giải đáp trong hội thoại Giai đoạn A

1. **Đường dẫn plan:** `docs/plan.md` không tồn tại. Giám sát xác nhận **`PLAN.md` tại gốc repo**
   là kế hoạch chính thức. Đã đọc §1.2–1.3, §3.1, §5 Bước 3, §6, §7.3, §8, §11.
2. **Config API:** dùng `pcb_lab.models.yolo.train._merge_configs(paths)`, `_config_hash(cfg)`,
   `_check_nulls(cfg)`, `ConfigError`. Tám khóa cấp cao hợp lệ: `model_id`, `recipe_id`,
   `architecture`, `pretrained`, `train`, `selection`, `infer`, `classes_ref`.
   Hash là SHA-256 của JSON canonical sắp khóa, không chứa seed.
3. **Detector metadata:** `model_id`, `checkpoint_sha256`, `class_order`, `infer_params`,
   `input_color="rgb"`. `describe()` bảo toàn các trường này.
4. **Fake engine:** tiêm trực tiếp `YoloDetector(engine, meta)`; engine có `class_names` và
   `infer(RGB uint8 HWC) -> ndarray[N,6]` trong tọa độ ảnh đầu vào.
   `from_artifact` kiểm `FileNotFoundError`, `SmokeArtifactError`, `ArtifactMismatchError`,
   `ClassOrderError` theo các điều kiện đã nêu trong hợp đồng/giải đáp.

Các điểm trên đã đóng; không cần thêm phê duyệt để viết test.

## Điểm schema còn chưa đặt tên chính xác, không chặn phần độc lập

- `partitions_used{train:n, val:"calibration", n}` chưa nêu tên khóa đếm calibration.
  Test chốt `train` là số nguyên, `val="calibration"`, không có fusion/test; tên khóa đếm
  còn lại cần đối chiếu schema được bàn giao ở B. Không tự sửa hợp đồng.
- `schema_version` chưa chốt kiểu string hay integer; test chấp nhận cả hai, loại bool.
- `peak_ram_mb` chưa chốt scalar hay object, hoặc tên trường method khi null. Test chấp nhận
  số không âm, object `{value, method}`, hoặc null. Ở C phải kiểm method được ghi khi null.
- Report calibration chỉ viết “infer params”, chưa chốt spelling. Test yêu cầu ít nhất một trong
  `infer_params` / `inference_params` / `infer` và giá trị bằng artifact, không tự bỏ qua trường này.
- Constructor `UltralyticsEngine` không được chốt. Test chặn nạp trọng số giả bằng cách thay
  **chỉ constructor này** bằng callable `*args, **kwargs -> FakeEngine(class_names=...)`.
  Không mock `YoloDetector.from_artifact`, hash hay kiểm class order trong các test guard/extraction.
  Nếu implementation dùng factory khác, chỉ sửa binding test khi chứng minh không phải lỗi hành vi.

Trong snapshot dataset, ảnh test (và ảnh ngoài danh sách manifest cho phép) chỉ lấy stat;
không mở byte để hash, không decode. Các file còn lại so SHA-256 + mtime + size, đồng thời
kiểm kê mọi đường dẫn/cache mới. Đây là cách giữ lệnh cấm mở ảnh test trong WORKFLOW khi kiểm dataset chỉ đọc.

## Giai đoạn C — bằng chứng cần bổ sung để đóng Step 3 (2026-09-25)

Đã hoàn tất các phép kiểm độc lập có thể chạy; kết quả chi tiết ở `step3-verify.md` phần C.
Không tự sửa hợp đồng hoặc dữ liệu artifact để chấp nhận null.

- Cả hai run không có git_commit/git_dirty hoặc hash source bundle ràng buộc với run. Cần
  xác định gói mã nguồn thực tế đã chạy, mã băm của nó và cơ chế provenance được điều phối duyệt;
  không thể lấy HEAD hiện tại làm commit lịch sử chỉ vì checkpoint checksum đúng.
- Cần bản gốc data.yaml và train.log của Colab; gói hiện tại chỉ có args.yaml/env/results.csv,
  manifest, checkpoint và các plot. Nếu không còn bản gốc phải ghi rõ giới hạn xác minh lịch sử test.
- Cần thống nhất validation 640×640 với inference 640×640 trước khi tái xuất report/preds.
  `val(rect=True)` hiện tạo tensor 672×672; nguyên nhân AP pin_hole B01 lệch đã được tái lập.
- Null pr_conf/n_gt_boxes là lỗi lấy field, không phải không có số liệu: `nt_per_class` tồn tại;
  số GT calibration đúng là `[234,160,339,277,269,265]`, tổng 1.544. Test metadata nghiêm ngặt giữ nguyên.

## Step 4 — Giai đoạn A (2026-09-27)

Các mục Step 3 ở trên giữ nguyên để bảo toàn lịch sử. Không đọc `step4/impl`,
source EfficientAD hoặc `step4-impl.md`; không sửa hợp đồng.

1. **ĐÃ ĐÓNG — Engine và normalization schema, commit `a385e16`.** Điều phối đã
   chốt `Engine.infer_tiles(uint8 NHWC) -> float32 NHW`, Detection frozen dùng
   `xyxy_original`, `class_id=None`, EfficientAdResult và normalization_stats gồm
   `source_split`, `n_good_images`, `quantile_min`, `quantile_max`, `mean`, `std`.
   Test dùng FakeEngine theo đúng protocol, gọi constructor/predict thật để kiểm
   tiling/R8 mà không cần checkpoint hoặc torch. JSONL vẫn dùng `xyxy` như contract.
   Schema normalization kiểm đầy đủ các trường đã chốt, không suy luận nguồn dữ
   liệu từ việc schema hợp lệ. `tile_params` chưa chốt tên khóa con; kiểm mapping,
   đối chiếu metadata và kiểm hành vi tile thực tế qua engine.
2. **Normalization audit / fit-save hook.** Cần giao diện hoặc audit trace nêu
   sample ID dùng để tính teacher stats (train good) và map quantiles (calibration
   good); contract hiện chỉ chốt tên split trong config, không chốt trace/hook.
   Dừng phần kiểm tra runtime provenance của các thống kê tại A. Giai đoạn B phải
   thử cài defect vào nguồn thống kê và kiểm nguồn độc lập. Test reload hiện
   deserialize toàn payload, save lại rồi load qua adapter, so sánh 5 ảnh calibration
   gồm good/defect; chưa chứng minh dự đoán model đang fit trước lần save đầu tiên.
3. **Tiền xử lý:** Step 4 nhắc `prepare_input(image, "ad")`, trong khi contract Step 2
   công khai `ad_tile`/`ad_resize`. Chưa ép alias `ad` chưa được chốt; kiểm hành vi
   tiling 9×256×256, stride 224 và stitching qua interface đã khóa của TileManager.
4. **Các điểm chưa định nghĩa:** kiểu cụ thể `schema_version`, tên file checkpoint
   trong run, quy ước `>`/`>=` ở đúng tau_pixel, connectivity 4/8 và điều kiện biên
   morphology. Test không đoán các chi tiết này: nhận version string/int, không
   chốt tên checkpoint run, dùng giá trị cách tau, component không mơ hồ. Ca ảnh sát
   biên dùng vùng 10×10 mà cả padding phổ biến đều giữ nguyên bounding rectangle.
5. **Môi trường:** venv cũ có đúng Python 3.14.7 / torch 2.14.0 / pytest 9.1.1 theo
   metadata, nhưng import torch báo WinError 4551 (Application Control chặn DLL)
   cả trong và ngoài sandbox. Không tự thay môi trường hoặc tắt chính sách Windows.

## Step 4 — cập nhật Giai đoạn B (2026-09-27)

- **Mục A.2 đã đóng cho kiểm định CPU smoke.** Implementation sau merge công khai
  `teacher_statistics`, `map_statistics` và sample_ids trong audit. Verifier đã
  chạy 12 ca contamination, đối chiếu sample IDs thực sự decode với manifest,
  kiểm thống kê bằng oracle độc lập, và chụp predict của model đang fit trước
  lần `_save(model.pt)` đầu tiên rồi so với from_artifact trên 5 ảnh good/defect.
  Sai số map/score = 0.0. Test: `tests/test_efficientad_phase_b.py`; chi tiết và
  giới hạn số kênh teacher được kiểm nằm trong `step4-verify.md` phần B.
- **Mục A.5 đã đóng trên venv do điều phối chỉ định.** Python 3.12.10 /
  torch 2.6.0+cpu import và train thật được khi chạy ngoài sandbox; không thay
  Windows Application Control. Venv Step 3 không được sửa.
- **Binding extractor ở A đã điều chỉnh sau đọc code.** FakeDetector bổ sung
  thuộc tính meta từ fixture và run_id từ manifest; toàn bộ assertion giữ nguyên.
  Không thay hợp đồng Engine hoặc nguồn production.
- **Còn điều kiện nghiệm thu B03:** chưa có STEP4_ARTIFACT_DIR baseline, nên 5
  test artifact/predictions thật vẫn skip. Cần artifact sau train chính thức và
  preds_calibration/fusion đủ 460/306 dòng. Đây là đầu vào cho vòng nghiệm thu
  tiếp theo, không phải yêu cầu sửa contract hay cho phép dùng smoke làm baseline.
- Không có câu hỏi spec mới chặn phần CPU đã kiểm. Giữ nguyên các quy ước chưa
  chốt ở A.3/A.4; không tự sửa alias preprocessing, connectivity hoặc biên threshold.

---

# Step 5 verifier: specification discrepancy recorded in Phase A

- PLAN.md:319 fixes the AD grid domain to `[mu, mu + 3*sigma]`.
- step5-contract.md:12 uses `[min(p10, mu-sigma), max(p99.5, mu+3*sigma)]`.
- The contract explicitly says PLAN wins on conflict. The Phase A test therefore
  checks PLAN's domain. Neither source of truth is edited. Acceptance of the
  wider envelope is withheld; an approved specification revision would be needed.
- The policy JSON nesting and GridSearchResult point representation are not
  prescribed. Tests check specified content without inventing nesting rules.

Phase A isolation: source and handoff implementation files were not read.
A concurrent Git operation fast-forwarded the requested verifier worktree to
`f121f2c` during setup. Tests were authored in a temporary worktree at `dbad064`
on `step5/verifier-spec-first`, then will be integrated into `step5/tests`.

## Step 5 — Phase B clarifications (2026-09-28)

- User confirmed there is no other active agent and requested continuation.
  The actual Git history also contains a pre-existing acceptance report commit
  `fd12d9e`; the new report supersedes its PASS using this verifier's own runs.
- Two Phase A assertions over-specified artifact structure: `calibration_metrics`
  used `{...}` (no mandatory `defect_to_good` field), and the selected H01 policy
  need not name B02 literally, repeat all variant thresholds, or include OR.
  Phase B removes these assumptions and verifies all six actual B01/B02/B03
  SHA-256 checksums instead. API metric requirements remain unchanged.
- PLAN does not fix population/sample standard deviation. The AD-domain test
  accepts either convention and 4-decimal serialization tolerance, while still
  requiring PLAN's mean lower bound. Both domains are independently computed.
- Earlier Step 3/4 question history was restored verbatim in Phase B after
  the Phase A edit inadvertently replaced it. No source-of-truth file changed.
- No acceptance waiver is requested. The domain conflict and implementation
  defects are recorded as FAIL, with no changes to implementation or artifacts.

## Step 6 — Phase A specification notes (2026-09-28)

- Read Step 6 contract and PLAN section 5 / section 7 before implementation.
  Config schema/path, BenchmarkResult/table structure, and ablation column names
  are not prescribed. Phase A writes independent assertions and stdlib formulas;
  only `tests/helpers/step6_bindings.py` will bind those representations in B.
  Pending hooks fail explicitly in A; no guessed interface is treated as a bug.
- Canonical comparison signs in tests: B02-B01, H01-H00, secondary-primary. Any
  documented alternative direction is adapted explicitly in B, not silently.
- PLAN prioritizes AD tile-versus-resize ablation, while Step 6's enumerated
  ablations omit it and forbid additional GPU forwards. Do not invent resize
  measurements. Audit whether the report states this missing comparison; full
  completion of that PLAN item remains unverified without existing predictions.
- Whole-suite execution must not enable older integration tests that read the
  real combined manifest, including test metadata. Use an unavailable explicit
  DATASET_ROOT and leave optional real model inputs disabled; record actual skips
  and environment failures rather than claiming every legacy test passed.

## Step 6 — Phase B binding decisions

- `run_development_benchmark` adds a `repo_root` argument and uses the Step 5
  `fusion_common.yaml`. Tests bind these real inputs without mocking evaluation.
  The runner requires exactly 460/306 rows. Synthetic fixtures now expand the
  Phase A patterns to those counts with unique IDs; the same independent oracle
  computes expectations over those actual fixture rows.
- Table JSON nests `splits[split][method]`; CSV uses `method`. The adapter only
  flattens/labels these representations, selects the six primary rows, and does
  not manufacture missing routing/provenance values. Secondary rows are checked
  separately by the independent script. All comparison directions match Phase A.
- Ablation targets are parsed from its comparison label (5% and 1%); nested
  threshold coordinates and `saving_rate_early_exit` are representation bindings.
- Shortlist can legitimately include derived `saving_rate` even when the input
  row lacks that key; test now checks it against `1-route_rate`. Public return
  objects are converted only to inspect contents; separate tests retain the
  exact contract requirement that ablation and shortlist APIs return dictionaries.
- PLAN permits YOLO-only AD routing N/A. Oracle accepts blank CSV values there;
  B03 must still report 100% routing. It does not excuse missing required keys.
- B03 run manifest exists upstream and SHA-256 matches artifact metadata. Its
  `train_time_s=4682.796126892999` is a measured training elapsed duration on
  `cuda:0`; reporting it as unavailable is not supported. GPU utilization/billing
  was not measured, and the verifier does not claim those quantities.
