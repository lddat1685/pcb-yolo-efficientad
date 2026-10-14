# Biên bản Kiểm định Độc lập Step 10 — Đóng gói, Tài liệu & Tái lập

> **Worktree:** `../PCB_Lab_verify10` — branch `step10/tests`
> (Phase A `249c038` + merge `step10/impl` `58e1954` tại `a75a4b8`)
> **Ngày kiểm định:** 2026-09-30
> **Python:** `../pcb_thin_lab/.venv/Scripts/python.exe` — Python 3.11.16
> **Frozen hash:** `038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd`
> **Căn cứ:** yêu cầu Step 10 của người dùng, `PLAN.md` §5 Bước 10 (§373-382), §6 quy tắc
> công bằng, §11.2 checklist nghiệm thu, `docs/handoff/WORKFLOW.md`, `docs/handoff/step10-impl.md`

---

## 1. Kết luận (Final Verdict)

**KẾT LUẬN: PASS**

- **Spec-first 13 tests:** 13/13 PASS (`tests/test_step10_packaging.py`) —
  5 docs tồn tại + không rỗng, 2 scripts tồn tại, 4 mục LIMITATIONS bắt buộc,
  2 entry-point (latex/markdown, sha256/hash).
- **Verify toàn vẹn:** **57 PASS, 0 FAIL, 3 MISSING / 60 checks**
  (`scripts/verify_reproducibility.py`) — MISSING là đúng thiết kế: 3 checkpoint
  `.pt` git-ignored, báo trung thực kèm hướng dẫn khôi phục, không PASS giả.
- **Export thesis:** **PASS** — `scripts/export_thesis_assets.py` sinh
  5 bảng × (`.md` + `.tex`) + `README.md` vào `exports/thesis/` (11 files),
  chạy lại idempotent, mọi bảng có `tabular`/`\hline`/separator hợp lệ.
- **Sabotage audit 20 ca:** 20/20 PASS (`scripts/verify_step10_sabotage.py`,
  log `docs/handoff/step10-sabotage-log.json`).
- **2 số độc lập tự xác minh:** (a) `sha256(frozen_evaluation.json)` =
  `038aa4a3…` khớp byte-cuối; (b) H01 `BalAcc 0.95 / route 0.45` đọc độc lập từ
  `benchmark_summary.csv` khớp `exports/thesis/overall_comparison.tex` (`95.00\%`, `45.0\%`).

**Bảng phát hiện:** 0 Critical, 0 Major, 3 Minor (ghi nhận, không chặn nghiệm thu —
xem §4).

---

## 2. Phương pháp thẩm định độc lập

### 2.1. Verify toàn vẹn: chạy độc lập, không tin báo cáo impl

```bash
cd ../PCB_Lab_verify10
../pcb_thin_lab/.venv/Scripts/python.exe scripts/verify_reproducibility.py
# -> [verify] 57 PASS, 0 FAIL, 3 MISSING / 60 checks (exit 1 do MISSING, đúng thiết kế)
```

Đối chiếu đầu ra thực tế (head):

```
[PASS   ] frozen file sha256 - 038aa4a39ba5…
[PASS   ] frozen test_split_sha256 - 440-image canonical test split
[PASS   ] frozen checkpoint SHAs - B01/B02/B03 entries
[PASS   ] frozen operating points - H01 0.14/1.4598, H01_sec 0.48
[MISSING] checkpoint B01 best.pt - absent (git-ignored); restore from checkpoint handoff, then re-run
[MISSING] checkpoint B02 best.pt - (như trên)
[MISSING] checkpoint B03 model.pt - (như trên)
[PASS   ] sidecar artifacts/yolo/B01_yolo11n/seed42/artifact.json - checkpoint_best_sha256 matches
[PASS   ] sidecar artifacts/yolo/B02_yolo11s/seed42/artifact.json - checkpoint_best_sha256 matches
... (toàn bộ 24 required runtime files + 7 reports + 5 docs + 11 thesis assets PASS)
[PASS   ] benchmark models - B01/B02/B03/H00/H01/H02/H01_sec
[PASS   ] benchmark ranges - BalAcc/FRR in [0,1]
[PASS   ] benchmark H01 spot-check - BalAcc 0.95, recall 1.0
[PASS   ] parity B01/B03/H00/H01 - maxDelta=0.0, matches=20/20
```

Verifier xác nhận impl báo trung thực: đúng `57 PASS, 0 FAIL, 3 MISSING`,
không khai khống 60/60.

### 2.2. Export thesis: chạy lại độc lập + validate format

```bash
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_thesis_assets.py
# -> [Step 10] Done: 5 tables x (md + tex) + README.md
../pcb_thin_lab/.venv/Scripts/python.exe -m pytest tests/test_step10_packaging.py -q
# -> 13 passed in 0.09s
```

Kiểm tra thủ công 5 cặp bảng:

| Asset | `.md` | `.tex` | Nguồn khóa |
|---|---|---|---|
| `overall_comparison` | 7 models, 11 cột | `tabular{lccccccccc}` + `\hline` | `benchmark_summary.csv` |
| `per_group_breakdown` | Group/n/Recall/FRR/BalAcc | `tabular{llcccc}` | `benchmark_per_group.csv` (N=2) |
| `per_class_detection` | B01/B02/H01 × 6 lớp | escape `\_` | `benchmark_per_class.csv` |
| `latency_memory` | mean/p50/p95/peak RSS | `tabular{llcccc}` | `artifacts/runtime/*/profile.json` |
| `confusion_counts` | TP/FN/FP/TN | `tabular{lcccc}` | suy từ locked rates, đã ghi caption |

### 2.3. Hai số độc lập (đường tính riêng, không qua script impl)

```python
# (a) frozen SHA — hashlib trực tiếp
hashlib.sha256(open('artifacts/test/frozen_evaluation.json','rb').read()).hexdigest()
# -> 038aa4a39ba5632844fdbcfc09276065428340211e2686d68f18af1755997bcd (khớp tuyệt đối)

# (b) H01 từ CSV gốc — csv.DictReader thuần
# -> ('H01', '0.95', route '0.45', latency '1348.7899...')
#    khớp overall_comparison.tex: `H01 & 95.00\% ... & 1348.8 & 45.0\%`
# (c) phụ trợ: H01 profile.json total mean 4986.457 — khớp báo cáo Step 9
```

### 2.4. Sabotage audit 20 ca (S01–S20)

Chạy `scripts/verify_step10_sabotage.py` → **20/20 PASS**
(log `docs/handoff/step10-sabotage-log.json`).

- S01–S05 (thiếu thẻ tài liệu): README vắng/mất, MODEL_CARD thiếu `SHA-256`,
  DATASET_CARD thiếu `calibration`, REPRODUCIBILITY thiếu `sha-256` → đều bị từ chối.
- S06–S09 (thiếu giới hạn): LIMITATIONS mất `N=2` / `bare PCB` / `domain shift` /
  `AD locked` → đều bị từ chối (đã sửa guard case-insensitive `n=2` để mỗi ca
  fail đúng lý do, không fail ké).
- S10–S13 (giả mạo): frozen bị chèn byte, checkpoint giả, sidecar SHA `0…0`,
  `tau_yolo=0.99` giả → đều báo `Hash mismatch` / `threshold mismatch`.
- S14–S15 (format sai): `.tex` thiếu `tabular`, `.md` thiếu bảng pipe → bị từ chối;
  file thật đã qua positive-control trước khi phá.
- S16–S19 (giả mạo số liệu): confusion `TP+FN=225`, CSV thiếu `H01_sec`,
  `BalAcc=1.5`, parity `Δ=0.005` → đều bị chặn (`>=1e-3`).
- S20 (trung thực MISSING): 3 `.pt` vắng phải báo `MISSING` (không `PASS`),
  và `verify_reproducibility.py` phải exit 1 khi còn MISSING — đã gọi trực tiếp
  `vr.main()` để chứng minh, PASS.

---

## 3. Kiểm đối chiếu artifact & ràng buộc workflow

- `git diff --name-only main...HEAD | grep -E "^(src|artifacts|tests/test_step[1-9])"`
  → **trống** (chỉ chạm `tests/test_step10_packaging.py` là file mới của verifier).
  Toàn bộ diff Step 10 là file mới docs/scripts/exports + 1 sửa cộng thêm
  `.gitignore` (`exports/*` + `!exports/thesis/`) — **không sửa source, benchmark
  artifact hay suite Bước 1–9 nào**, `frozen_evaluation.json` giữ nguyên hash.
- Parities Step 9 còn nguyên: `export_report.json` 4 model đều
  `score_delta_max=0.0`, `decision_matches=20/20`, flips 0 (ngưỡng Δ<1e-3).
- Repo sẵn sàng nộp/bảo vệ: `README.md` (261 dòng, tiếng Việt) bao đủ cài đặt CPU,
  `DATASET_ROOT`, checkpoint SHA, UI (`run_ui.py`), CLI inspect, bảng tái lập 13 bước,
  bảng test cuối, giới hạn, license — người ngoài làm theo là chạy được
  (sau khi lấy 3 `.pt` từ bản bàn giao checkpoint theo `REPRODUCIBILITY.md` §2).

---

## 4. Bảng Phát hiện (Findings — Critical/Major/Minor)

| Mức | Vị trí | Phát hiện | Bằng chứng | Đề xuất | Trạng thái |
|---|---|---|---|---|---|
| Minor | `scripts/verify_reproducibility.py` + worktree | 3 checkpoint `.pt` vắng → verify exit 1 (57/60). Đúng thiết kế git-ignore, nhưng người chấm chạy verify sẽ thấy exit nonzero | `[MISSING] checkpoint B01/B02/B03` + dòng hướng dẫn restore | Khôi phục 3 `.pt` từ bản bàn giao checkpoint rồi chạy lại để đạt 60/60 trước buổi bảo vệ | **Ghi nhận** |
| Minor | `exports/thesis/confusion_counts.*` + caption | Bảng confusion suy từ locked rates (220/220), không đếm trực tiếp từng ô từ predictions | impl tự khai `step10-impl.md` §"Hạn chế"; caption bảng đã ghi `derived from locked test rates` | Giữ nguyên + trích dẫn kèm caption khi đưa vào luận văn; không gọi là đếm trực tiếp | **Ghi nhận** |
| Minor | `README.md:99` | Lệnh mẫu ghi `cd ..\PCB_Lab_impl10` (đường dẫn worktree impl) thay vì repo-root chung | `README.md:99` | Đổi thành `cd <repo-root>` khi nộp bản cuối để người ngoài clone về không nhầm | **Đề xuất** |
| - | — | Không có Critical/Major nào: docs đủ 5 file, LIMITATIONS đủ 4 mục, thesis đủ 5 bảng, hash frozen khớp, parity 0.0 | — | Nghiệm thu | PASS |

---

## 5. Lệnh & kết quả thực (đã chạy trong worktree này)

```bash
# Spec-first suite (Phase A, sau merge vẫn xanh)
../pcb_thin_lab/.venv/Scripts/python.exe -m pytest tests/test_step10_packaging.py -q
# -> 13 passed in 0.09s

# Verify toàn vẹn độc lập
../pcb_thin_lab/.venv/Scripts/python.exe scripts/verify_reproducibility.py
# -> [verify] 57 PASS, 0 FAIL, 3 MISSING / 60 checks

# Export thesis độc lập
../pcb_thin_lab/.venv/Scripts/python.exe scripts/export_thesis_assets.py
# -> [Step 10] Done: 5 tables x (md + tex) + README.md

# 2 số độc lập (csv/hashlib thuần, không qua script impl)
# -> frozen_sha 038aa4a3… (khớp); H01 BalAcc 0.95 / route 0.45 (khớp .tex)

# Sabotage 20 ca
../pcb_thin_lab/.venv/Scripts/python.exe scripts/verify_step10_sabotage.py
# -> 20/20 cases passed -> docs/handoff/step10-sabotage-log.json

# Không chạm Bước 1–9
git diff --name-only main...HEAD | grep -E "^(src|artifacts|tests/test_step[1-9])"
# -> (trống)
```

---

## 6. Bàn giao

- Handoff impl: `docs/handoff/step10-impl.md`
- Handoff verify (file này): `docs/handoff/step10-verify.md`
- Sabotage log: `docs/handoff/step10-sabotage-log.json`
- Script audit: `scripts/verify_step10_sabotage.py`
- Suite spec-first: `tests/test_step10_packaging.py`
- Thesis assets: `exports/thesis/` (5 bảng × md/tex + README)

*Verifier xác nhận: repo ở trạng thái sẵn sàng nộp đồ án và bảo vệ — người ngoài
tải repo, làm theo README ( + lấy 3 checkpoint `.pt` từ bản bàn giao theo
`docs/REPRODUCIBILITY.md` §2 ) là chạy được ngay; mọi số công bố truy ngược được
về `artifacts/test/frozen_evaluation.json` (`038aa4a3…`).*
