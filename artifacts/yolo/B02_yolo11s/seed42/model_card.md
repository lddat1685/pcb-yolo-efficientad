Model card — B02_yolo11s

Muc dich
Baseline phat hien 6 loai loi PCB (YOLO detect) dung kien truc yolo11s.
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
config_hash=6070bffbb4b71c71929fce974acfa0e81148dc12d42882a40f226e45df3afa7b, view_signature=d4e75ccc65d3e3665ce5cd6fbe03c4788f5b9b69daba60ae515b31f2b69d6bd1.

So lieu calibration (F1-optimal confidence, KHONG phai nguong van hanh)
mAP50=0.9408849607293804, mAP50-95=0.7392858571501647, precision=0.910859149909308, recall=0.9197194136460188.

Theo lop
mouse_bite: ap50=0.9685997069734623, ap50_95=0.7710773840783082, precision=0.9552543990145346, recall=0.9446283486875141.
open_circuit: ap50=0.9909022013046781, ap50_95=0.6936441548516169, precision=0.97032823562929, recall=0.9914529914529915.
pin_hole: ap50=0.831403758424777, ap50_95=0.7537212621248008, precision=0.6085728377218853, recall=0.8566037735849057.
short: ap50=0.9343414545671933, ap50_95=0.6207301791238156, precision=0.9603912764208942, recall=0.90625.
spur: ap50=0.9787033241992871, ap50_95=0.73931783716583, precision=0.9747941192983912, recall=0.9350180505415162.
spurious_copper: ap50=0.9413593189068845, ap50_95=0.8572243255566161, precision=0.9958140313708532, recall=0.8843633176091859.

Gioi han
Dataset chi co N=2 nhom nguon o moi partition (calibration/fusion/test);
so anh khong dong nghia so quan sat vat ly doc lap.
Khong co pixel mask chuan nen khong bao pixel AUROC/Dice/IoU segmentation.
Chua chay tren test; khong dung test chon cau hinh.

Metadata refresh: calibration only, rect=False, imgsz=640; ultralytics 8.4.161, torch 2.14.0+cpu, device=cpu.
pr_conf=0.5795795795795796; epochs_run=43, best_epoch=23, train_time_s=1695.3.
Training source commit: 7623c11d88e4502c879d6f0efa71aab4ae495839. Checkpoints and predictions preserved.
