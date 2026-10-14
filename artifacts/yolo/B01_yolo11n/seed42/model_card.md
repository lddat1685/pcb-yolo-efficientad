Model card — B01_yolo11n

Muc dich
Baseline phat hien 6 loai loi PCB (YOLO detect) dung kien truc yolo11n.
Day la baseline Buoc 3 cua do an; chi neu ho model da dung, khong so sanh voi model khac.

Du lieu dung
Train: 1792 anh (good 895 / defect 897).
Chon checkpoint bang validation tren calibration: 460 anh.
Khong dung tap test de chon epoch hay nguong.

Cau hinh
optimizer=AdamW, lr0=0.001, batch=16, imgsz=640, epochs=100, patience=20.
Augmentation: yolo_aug_v1 (hinh hoc thuan: xoay 90 x k + lat ngang; tat moi photometric/mosaic).
Infer: conf_floor=0.001, iou=0.7, max_det=300.

Phien ban
ultralytics 8.4.161, torch 2.14.0+cu130.
config_hash=1361dcc44534ff23c52e97949271dab1aa755fe8a90e85874c962db2ac9482f1, view_signature=d4e75ccc65d3e3665ce5cd6fbe03c4788f5b9b69daba60ae515b31f2b69d6bd1.

So lieu calibration (F1-optimal confidence, KHONG phai nguong van hanh)
mAP50=0.9341408318893346, mAP50-95=0.7240575938947142, precision=0.9200001954248777, recall=0.8945469545445234.

Theo lop
mouse_bite: ap50=0.9638803000948835, ap50_95=0.7511297757465137, precision=0.9191823533152107, recall=0.9174041297935103.
open_circuit: ap50=0.9863670110986422, ap50_95=0.6864870748855068, precision=0.9416210895919941, recall=0.9650139496327174.
pin_hole: ap50=0.8182132012674659, ap50_95=0.7402501771368164, precision=0.8063539646163647, recall=0.7886792452830189.
short: ap50=0.9085818610403151, ap50_95=0.5975648937099005, precision=0.9065298055793938, recall=0.8486344440622556.
spur: ap50=0.957774227710809, ap50_95=0.6996073434204076, precision=0.9621692959212941, recall=0.9181819287558606.
spurious_copper: ap50=0.9700283901238927, ap50_95=0.8693062984691406, precision=0.9841446635250081, recall=0.929368029739777.

Gioi han
Dataset chi co N=2 nhom nguon o moi partition (calibration/fusion/test);
so anh khong dong nghia so quan sat vat ly doc lap.
Khong co pixel mask chuan nen khong bao pixel AUROC/Dice/IoU segmentation.
Chua chay tren test; khong dung test chon cau hinh.

Metadata refresh: calibration only, rect=False, imgsz=640; ultralytics 8.4.161, torch 2.14.0+cpu, device=cpu.
pr_conf=0.4964964964964965; epochs_run=33, best_epoch=13, train_time_s=1260.5.
Training source commit: 7623c11d88e4502c879d6f0efa71aab4ae495839. Checkpoints and predictions preserved.
