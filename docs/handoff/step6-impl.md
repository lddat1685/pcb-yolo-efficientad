# Step 6 IMPLEMENTER — Benchmark baseline và chọn shortlist

## Phase 0 — Diễn giải và thiết kế (2026-09-28)

Đã đọc `WORKFLOW.md`, `step6-contract.md`, `PLAN.md` (§5 Bước 6, §7.1–§7.3).
Worktree `../PCB_Lab_impl6`, nhánh `step6/impl`;
venv `.../.cache/step3-verifier-b/.venv/Scripts/python.exe`. Không có AGENTS.md/CLAUDE.md.

### Diễn giải

- Tái sử dụng engine Bước 5 (`src/pcb_lab/fusion/{rules,evaluate,metrics}.py`) qua package mới
  `src/pcb_lab/evaluation/`: `benchmark.py` (đánh giá thống nhất B01/B02/B03/H00/H01/H02
  trên cả `fusion` 306 ảnh và `calibration` 460 ảnh tại ngưỡng khóa), `ablation.py`
  (3 ablation bắt buộc), `shortlist.py` (luận cứ + ngân sách).
- Runner `scripts/benchmark_baseline.py` xuất `artifacts/benchmark/development_summary.{json,csv}`,
  `ablation_summary.{json,csv}`, `shortlist.json`, `reports/step6_baseline_benchmark_report.md`.
- Nguồn thật: 6 file preds (checksum khớp `routing_policy.json/source_checksums`),
  `thresholds.json` (H01 τ=(0.14, 1.4598), H02 τ=(0.38, 1.4598), `target_frr=0.05`),
  ngưỡng phụ FRR≤1% suy trực tiếp từ `grid_calibration.json` hiện tại bằng đúng
  luật chọn Bước 5 → H01 phụ (0.48, 1.4598), calib FRR 0.87% / recall 100%.
  (Số (0.48, 0.7879) trong `step5-impl.md` thuộc về lưới cũ trước khi verifier patch
  sweep về [μ, μ+3σ]; không dùng.)
- Lưu ý: τ_ad chính hiện tại là **1.4598** (= μ defect calib), không phải 0.7879 của bản
  step5-impl cũ — verifier đã patch sweep về `[μ, μ+3σ]` đúng PLAN §5. Dùng số hiện tại.

### File dự kiến

- `src/pcb_lab/evaluation/{__init__,benchmark,ablation,shortlist}.py` (mới).
- `scripts/benchmark_baseline.py` (mới).
- `tests/test_benchmark_impl.py` (mới): guard test-leak, determinism, schema artifact.
- `artifacts/benchmark/*`, `reports/step6_baseline_benchmark_report.md` (mới).
- `docs/handoff/step6-impl.md` (file này).

### Ngân sách thật đã dùng (cho shortlist.json)

- B01 YOLO11n seed 42: 33 epoch (best 13), train_time 1260.5s ≈ **0.35 GPU-h** (device 0,
  Colab); pretrained `yolo11n.pt` ultralytics-assets (sha `0ebbc80d…`); recipe AdamW lr 1e-3,
  batch 16, imgsz 640, tối đa 100 epoch patience 20.
- B02 YOLO11s seed 42: 43 epoch (best 23), train_time 1695.3s ≈ **0.47 GPU-h**;
  pretrained `yolo11s.pt` (sha `85a76fe8…`); cùng recipe.
- B03 EfficientAD-S seed 42: **70.000 optimizer steps**, 895 good train (8055 tile),
  tile 256/stride 224; teacher PDN-S pretrained frozen; AdamW, không ImageNette penalty.
  GPU-hours chi tiết xem step4-verify (ghi "không đo được chính xác" nếu thiếu).
- Bước 5 sweep: 2050 điểm/lưới × 2 lưới (H01/H02), CPU <1 phút — tuning budget đã bao gồm
  trong bảng (không fit thêm tham số nào ngoài grid search trên calibration).
- Vòng sàng lọc: **seed 42 duy nhất**; seed 43/44 dành sau shortlist (PLAN Bước 6).

### Quyết định mở

1. Preds `*.jsonl` bị `.gitignore` nên không theo worktree — đã copy từ repo chính vào
   impl6 và **xác minh SHA-256 khớp 6/6 checksum** trong `routing_policy.json`. File copy
   không commit (đúng gitignore); provenance ghi trong handoff này.
2. Ablation tile-256 vs resize-256 (PLAN) **không thực hiện được**: chỉ có preds tile-256
   B03, không có preds resize-256 — ghi rõ trong báo cáo thay vì bịa số.
3. Latency: preds tĩnh `timing_ms=null` toàn bộ → Δlatency = null, saving rate tính từ
   route_rate (1 − route_rate) thay vì đo ms.

### Câu hỏi chặn: không có.

## Phase 1 — Kết quả thực hiện (2026-09-28)

### File đã tạo

- `src/pcb_lab/evaluation/{__init__,benchmark,ablation,shortlist}.py` — tái dùng
  engine Bước 5 (`evaluate_split`), không code metric mới.
- `scripts/benchmark_baseline.py` — runner xuất 6 artifacts + báo cáo.
- `tests/test_benchmark_impl.py` — 4 passed (guard test-leak, luật chọn điểm,
  3 ablation, schema shortlist).
- `artifacts/benchmark/{development_summary,ablation_summary}.{json,csv}`,
  `shortlist.json`; `reports/step6_baseline_benchmark_report.md`.

### Số thật đã xác minh

Determinism: chạy runner 2 lần → SHA-256 3 file JSON giống hệt.
Provenance: 6/6 checksum preds khớp `routing_policy.json`; mọi số fusion khớp
`thresholds.json` từng điểm (check tự động >1e-9 fail); `test_access: false`;
counts 460/306 đúng.

Fusion 306: H01 (0.14, 1.4598) BalAcc 80.72% / recall 100% / FRR 38.56% /
ECPB 0.3818 / route 30.7%; H02 (0.38, 1.4598) BalAcc 92.16% / FRR 15.69% /
ECPB 0.1553 / route 42.2%; B03 recall 28.76% / FRR 0% / ECPB 0.7124.
A1 (B02−B01): Δrecall 0, ΔFRR −0.2288, ΔBalAcc +0.1144, ΔECPB −0.2265.
A2 (H01−H00): Δ = 0 toàn bộ; saving rate 69.28%; rescue N/A (yolo_fn 0); addedFR 0.
A3 (phụ−chính H01): Δrecall 0, ΔFRR −0.2157, ΔECPB −0.2135;
điểm phụ (0.48, 1.4598), calib FRR 0.87% / recall 100%.
Drift calib→fusion FRR H01 4.78%→38.56%: ghi trung thực, không chọn lại ngưỡng.
Ngân sách: YOLO đo được 0.82 GPU-h (B01 0.35 + B02 0.47); B03 GPU-h không đo
chính xác (70k steps, ghi rõ); hybrid tái dùng training, không cộng dồn.
