# Step 10 — Implementer: Đóng gói và bàn giao đồ án

## Phase 0 — Kế hoạch trước triển khai (2026-09-30)

- Căn cứ: yêu cầu Step 10 của người dùng, `PLAN.md` §5 Bước 10 (§373-382), §6 quy tắc
  công bằng, §7.6 bất định/chọn mô hình, §11.2 checklist nghiệm thu và `WORKFLOW.md`.
  Branch `step10/impl` tạo mới từ `main` (commit e0e336d) tại worktree
  `../PCB_Lab_impl10` (240 files).
- Ràng buộc: **không sửa đổi hay làm hỏng bất kỳ file source code, benchmark artifact,
  hay test suite nào của Bước 1–9.** Mọi thay đổi Step 10 chỉ gồm file mới
  (docs/scripts/exports) + 1 sửa cộng thêm trong `.gitignore`.
- Python kiểm chứng: `../pcb_thin_lab/.venv/Scripts/python.exe`
  (torch CPU). Không train/tune, không chạm `frozen_evaluation.json`, không đổi thresholds.

### Phạm vi Step 10

1. **Tài liệu & thẻ đặc tả (Markdown):** `README.md` (tiếng Việt hoàn chỉnh),
   `docs/MODEL_CARD.md` (B01/B02/B03/H00/H01/H02/H01_sec: kiến trúc, tham số,
   SHA-256, ngưỡng khóa), `docs/DATASET_CARD.md` (DeepPCB, 6 lớp, holdout canonical,
   2 nhóm test), `docs/LIMITATIONS.md` (N=2, bare-PCB, domain shift, AD bị khóa khi
   YOLO dương), `docs/REPRODUCIBILITY.md` (từng bước + checksum).
2. **Scripts:** `scripts/verify_reproducibility.py` (toàn vẹn repo: tồn tại + SHA-256
   checkpoints/frozen/CSV/runtime) và `scripts/export_thesis_assets.py`
   (5 bảng LaTeX + Markdown vào `exports/thesis/`).
3. **Handoff + commit:** file này và commit
   `[claude] step10: package project deliverables, documentation, cards and thesis export assets`.

---

## Kết quả triển khai và kiểm thử

### File đã tạo (8 mới + 1 sửa config)

| File | Nội dung |
|------|----------|
| `README.md` | Giới thiệu đề tài, kiến trúc Hybrid (Cascade H01 vs OR H00), cài đặt CPU, chạy UI (`scripts/run_ui.py`), CLI inspect, tái lập thí nghiệm, bảng test cuối, giới hạn, license dữ liệu |
| `docs/MODEL_CARD.md` | 7 cấu hình: B01 YOLO11n (~2.6M, `c0cfe917…`, τ=0.14), B02 YOLO11s (~9.4M, `aa079248…`, τ=0.38), B03 EfficientAD-S (~8.06M, `9453bc2c…`, τ=1.4598, tile 256/stride 224), H00 OR, H01 cascade (demo mặc định), H01_sec (τ=0.48), H02 + bảng tổng hợp |
| `docs/DATASET_CARD.md` | DeepPCB 2998 ảnh 640×640, 6 lớp (bbox train/test), holdout train 1792 / calib 460 / fusion 306 / test 440, test = group44000 + group92000, audit PASS 18/18, rò rỉ 0/6 cặp |
| `docs/LIMITATIONS.md` | N=2 test group [R1/v3-P0.4], bare-PCB không linh kiện, domain shift đã đo (calib FRR 4.78% → test 10%), AD bị khóa khi YOLO dương [S1], không mask → không pixel metrics, chênh <0.45pp không kết luận |
| `docs/REPRODUCIBILITY.md` | 9 bước tái lập (audit → train GPU → calibrate → benchmark → test → UI/CLI → runtime → đóng gói) + bảng 10 checksum |
| `scripts/verify_reproducibility.py` | 60 checks read-only: frozen SHA, operating points, checkpoints, sidecars, required files, CSV content (7 models, H01 spot-check), parity 4 models |
| `scripts/export_thesis_assets.py` | 5 bảng × (.md + .tex) + README index: overall, per-group, per-class, latency-memory, confusion |
| `exports/thesis/` (11 files) | Output của script trên, version riêng cho luận văn |
| `.gitignore` (sửa cộng thêm) | `exports/*` + `!exports/thesis/` — chỉ version bảng thesis, giữ ignore các export tạm khác (cùng pattern Step 9 đã dùng cho `artifacts/runtime/`) |

### Lệnh và kết quả thực tế

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Compile scripts | `python -m py_compile scripts/verify_reproducibility.py scripts/export_thesis_assets.py` | **PASS** |
| Export thesis | `python scripts/export_thesis_assets.py` | **PASS** — 5 bảng × (md+tex) + README, số liệu khớp CSV khóa (H01 BalAcc 95.00%, route 45.0%; H00 p50 ~12052 ms) |
| Verify toàn vẹn | `python scripts/verify_reproducibility.py` | **57 PASS, 0 FAIL, 3 MISSING** — MISSING là 3 checkpoint `.pt` (git-ignore theo thiết kế, báo trung thực kèm hướng dẫn khôi phục, không PASS giả) |
| Parity runtime | (trong verify) B01/B03/H00/H01 | **20/20 matches, maxDelta 0.0** cả 4 model |
| Frozen untouched | `sha256(artifacts/test/frozen_evaluation.json)` | **038aa4a3…** giữ nguyên |
| Step 9 tests | `pytest tests/test_step9_runtime.py` | **5 passed** |
| Step 8 UI tests | `pytest tests/test_step8_ui.py` | **18 passed, 1 failed** — `test_apptest_screens_and_rerun` lỗi `altair_chart() got an unexpected keyword argument 'width'` (xung đột phiên bản streamlit/altair trong venv, thuộc code UI Bước 8; Step 10 không sửa source nào nên không phải hồi quy) |
| Thu thập test | `pytest tests/ --collect-only` | **435 tests** collected, không lỗi import |

### Sửa lỗi trong quá trình làm

1. `Write` thiếu tham số (lỗi gọi tool) — ghi lại đầy đủ, không ảnh hưởng repo.
2. `UnicodeEncodeError` ký tự `Δ`/`—` trên console Windows (cp1252): `UnicodeEncodeError`
   là subclass của `ValueError` nên bị `except ValueError` nuốt thành FAIL giả cho 4 check
   parity — đã sửa thành ASCII (`maxDelta`, `-`) + fallback `ascii/replace` trong `record()`.
3. `exports/` bị ignore nên bảng thesis không version được — đã đổi sang `exports/*` +
   `!exports/thesis/` và bổ sung 11 file thesis vào verify script.

### Bằng chứng không phá vỡ Bước 1–9

- `git status --short`: chỉ `M .gitignore` + 7 file mới (`?? README.md`, 4 docs, 2 scripts)
  và `?? exports/` — **không có file modified nào trong `src/`, `artifacts/`, `tests/`**,
  không touched `frozen_evaluation.json` (hash xác minh giữ nguyên).
- Không test nào của Bước 1–9 bị sửa; 435 tests collect sạch; Step 9 suite 5/5 pass.

### Hạn chế & ghi chú trung thực

- 3 checkpoint `.pt` vắng trong worktree (git-ignore) → verify báo MISSING + exit 1;
  khôi phục từ bản bàn giao checkpoint rồi chạy lại để đạt 60/60.
- Bảng confusion trong thesis assets suy từ locked rates (không phải đếm trực tiếp từng
  ô từ predictions) — đã ghi rõ nguồn trong caption từng bảng.
- Số tham số YOLO (~2.6M/~9.4M) là giá trị chuẩn kiến trúc Ultralytics, không đếm lại
  từ checkpoint vắng mặt — đã ghi chú trong MODEL_CARD.
- 1 test UI fail do môi trường (streamlit/altair), ngoài phạm vi Step 10.

---

## Bàn giao

- Branch: `step10/impl`, nền `main` e0e336d (frozen hash `038aa4a3…`).
- Commit theo yêu cầu: `[claude] step10: package project deliverables, documentation, cards and thesis export assets`.
- Tái lập đóng gói:
  ```bash
  ../pcb_thin_lab/.venv/Scripts/python.exe scripts/verify_reproducibility.py
  ../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_thesis_assets.py
  ```
- Người tiếp theo: khôi phục 3 checkpoint `.pt` → verify 60/60 → đọc `docs/REPRODUCIBILITY.md`
  để chạy lại toàn bộ → dùng `exports/thesis/*.tex` dán trực tiếp vào luận văn.
