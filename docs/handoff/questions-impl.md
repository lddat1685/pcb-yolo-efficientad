# Câu hỏi IMPLEMENTER — Step 2

## Q1 — Sample.pair_id khi canonical có nhiều pair alias (đã chốt)

- Contract `data/samples.py` yêu cầu `Sample.pair_id` nhưng chưa chốt kiểu/ánh xạ từ `pair_ids` của manifest.
- Bằng chứng đọc metadata qua adapter Step 1, không decode ảnh: `deeppcb_20085159_good` có `pair_ids=["20085159","20085167","20085168"]`.
- Đề xuất: `pair_id: str = min(pair_ids)` và thêm `pair_ids: tuple[str,...]` chứa toàn bộ alias theo thứ tự ổn định; mọi isolation/grouping cần đầy đủ quan hệ sẽ dùng pair_ids. Phương án khác: chốt pair_id là tuple toàn bộ alias.
- Người dùng đã trả lời: **"Chốt chuỗi đại diện + tuple đầy đủ"**. Áp dụng pair_id là chuỗi nhỏ nhất theo từ điển và pair_ids là tuple đầy đủ. Không còn chặn loader; không sửa contract.

## Q2 — Duyệt recipe aug_v1 (đã duyệt theo yêu cầu Phase 0.3)

- Đề xuất chi tiết tại `step2-impl.md`: train-only, k×90° (k đều 0..3) rồi horizontal flip p=0.5, seed cục bộ; hoán vị pixel/box chính xác, không nội suy/crop/photometric.
- Rủi ro: thay phân phối hướng layout. Không làm mất hoặc thay hình thái chi tiết lỗi do resampling, nhưng chưa chứng minh cải thiện model.
- Người dùng đã trả lời: **"Duyệt recipe hình học này"**, rồi yêu cầu **"tiếp tục"**. Đã duyệt đúng recipe nêu trên; không tự thêm photometric hoặc biến đổi khác.

## Step 4 — Q3: learning rate và weight decay (đã chốt)

Contract đặt train.lr và train.weight_decay = null (chưa duyệt), bắt buộc ConfigError cả smoke. Người dùng trả lời trong phiên 2026-09-27: **"Dùng lr=0.0001, weight_decay=0.00001 cho B03"**. Đã điền cấu hình chính theo câu trả lời này; kiểm tra ConfigError khi đưa về null vẫn giữ nguyên. Không còn chặn phần train.

## Step 4 — Q4: tên chế độ preprocessing (đã diễn giải)

Contract ghi prepare_input(image, "ad"), API Step 2 thực tế dùng "ad_tile". Dùng "ad_tile" để tái sử dụng đúng native TileManager; không thay đổi contract hoặc API Step 2.
