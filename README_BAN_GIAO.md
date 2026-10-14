# Bộ bàn giao PCB YOLO + EfficientAD

- File: `PCB_YOLO_EfficientAD_Lab_portable_20261005.zip` (340.7 MB).
- Mã nguồn tại commit `b98c9dab46db3d41dd3732bd6cb7126785c3942f`; kèm checkpoint gốc, TorchScript, artifacts,
  training runs, tests, tài liệu, báo cáo và bảng xuất luận văn.
- Không kèm `.venv`, Git, cache, scratch hoặc các ZIP thí nghiệm trùng lặp.
- Đã giải nén thử: toàn bộ 618 tệp khớp SHA-256, 9 artifact được bảo vệ
  giữ nguyên và kiểm tra reproducibility đạt **60/60 PASS**. Xem `verification.txt`.

## Cách chạy

1. Giải nén toàn bộ ZIP vào một thư mục có quyền ghi.
2. Chuẩn bị Python 3.11/3.12 64-bit và Internet để cài dependency.
3. Windows: chạy `setup_env.bat`, sau đó `run_ui.bat`.
4. Camera A3504MG100: c?i MV Viewer + IMV SDK x64 tr?n m?y n?i camera, ch?y `run_camera_check.bat`; xem `docs/CAMERA_DAHUA.md`. Camera v?t l? ch?a ???c ki?m th? t?i m?y ph?t tri?n.
5. Linux/macOS: `bash setup_env.sh`, sau đó `bash run_ui.sh`.

**Dataset chưa nằm trong ZIP.** Upload ảnh riêng vẫn dùng được. Để dùng Gallery,
parity hoặc kiểm thử cần dataset, đặt `DatasetVer4_Public` bên cạnh thư mục đã
giải nén, đặt nội dung dataset trong `data/`, hoặc cấu hình `DATASET_ROOT`.
Chi tiết yêu cầu nền tảng và cách kiểm thử nằm trong `README.md` của dự án.

Tệp `.sha256` bên cạnh ZIP dùng để kiểm tra tính toàn vẹn sau khi sao chép.
