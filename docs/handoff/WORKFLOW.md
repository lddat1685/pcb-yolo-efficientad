\# Quy ước làm việc chung (IMPLEMENTER và VERIFIER)



\- Môi trường: Windows, PowerShell 5.1 (không dùng \&\&; dùng ; hoặc mỗi lệnh một dòng). Đọc AGENTS.md nếu có.

&#x20; $env:DATASET\_ROOT đã đặt.

\- Nguồn sự thật, chỉ đọc: docs/plan.md, docs/DECISIONS.md, docs/TRIAGE.md, hợp đồng của step

&#x20; (docs/handoff/stepN-contract.md). Không dùng docs/archive/ làm chỉ dẫn. Chỉ đọc docs/review.md khi cần lý do của một quyết định.

\- Hợp đồng là giao ước: được thêm khoá/hàm, KHÔNG đổi tên/kiểu/hành vi đã chốt và KHÔNG sửa file hợp đồng.

&#x20; Thấy sai/thiếu: ghi docs/handoff/questions-<impl|verify>.md và dừng phần bị ảnh hưởng.

\- Dataset chỉ đọc; đọc mẫu theo manifest, không glob; không hard-code số dataset trong code.

&#x20; Không decode/mở/xem ảnh test (chỉ metadata) trừ khi hợp đồng của step cho phép rõ.

\- Mỗi agent một branch/worktree, không sửa branch của nhau. Commit nhỏ, dạng "\[claude|codex] stepN: <việc>".

&#x20; Trước mỗi commit chạy git status; không commit dataset, ảnh, checkpoint, key/token.

&#x20; Không force push, không rebase, không merge vào main, không sửa docs/plan.md, DECISIONS.md, TRIAGE.md.

\- Bàn giao qua docs/handoff/stepN-impl.md và stepN-verify.md. Chỉ được nói "xong" khi đã chạy thật;

&#x20; ghi lệnh và kết quả thật, kể cả phần fail. Số liệu lệch plan: báo nguyên trạng, không chỉnh code hay kỳ vọng cho khớp.

\- IMPLEMENTER bắt đầu bằng Phase 0: diễn giải, chỗ mơ hồ, danh sách file, các quyết định mở. Mơ hồ chặn việc làm đúng thì DỪNG và hỏi.

\- VERIFIER viết test từ spec trước khi đọc code: không đọc src của branch impl và stepN-impl.md cho tới khi commit Giai đoạn A.

&#x20; Giai đoạn B bắt buộc có nhật ký phá hoại và tự xác minh ít nhất 2 con số bằng đường độc lập.

&#x20; Kết luận PASS / PASS-WITH-CONDITIONS / FAIL kèm bảng phát hiện (Critical/Major/Minor, file:dòng, bằng chứng, đề xuất sửa).

\- Test đỏ ở Giai đoạn A là bình thường; import module cần test bên trong fixture/hàm để pytest vẫn collect được.

