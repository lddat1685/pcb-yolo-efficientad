# B03 EfficientAD-S

Mục đích: baseline phát hiện bất thường PCB. Teacher PDN-S pretrained frozen, student 768 kênh và autoencoder.

Dữ liệu: 895 good train, 8055 tile, 0 defect; 70000 optimizer step. Tile 256x256, stride 224, reflect và linear blending; score=max map đã stitch.

Teacher stats từ train good; map norm từ 230 calibration good. Calibration defect chỉ dùng báo cáo sau khi khóa normalization. Không truy cập pixel test.

Phân phối score calibration:

```json
{
  "split": "calibration",
  "n_good": 230,
  "n_defect": 230,
  "good_scores": {
    "min": 0.0,
    "max": 0.7240502238273621,
    "mean": 0.191201680192076,
    "std": 0.14496944117635902,
    "p50": 0.14966683089733124,
    "p90": 0.4090312749147415,
    "p95": 0.5259406387805938,
    "p99": 0.6418012857437134
  },
  "defect_scores": {
    "min": 0.18060244619846344,
    "max": 9.916268348693848,
    "mean": 1.4598158236431038,
    "std": 1.5820265934356676,
    "p50": 0.990909218788147,
    "p90": 3.458925366401671,
    "p95": 4.778082108497619,
    "p99": 9.337356052398686
  },
  "source": "frozen_checkpoint_max_stitched_map; descriptive only; no auto-threshold"
}
```

Recipe standalone: AdamW, không ImageNette penalty, không augmentation, không auto-threshold; không khẳng định tái lập điểm số bài báo. Tau_pixel chỉ tạo candidate box, chưa có tau_ad quyết định ảnh.

Giới hạn: không phân loại sáu lớp lỗi (anomaly_unclassified); N=2 nhóm nguồn test, dedicated line; chưa chứng minh domain shift. Smoke chỉ kiểm pipeline, không đo chất lượng B03.
