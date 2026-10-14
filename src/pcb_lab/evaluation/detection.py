"""Shared raw-box evaluator using the installed Ultralytics validation convention.

Class-aware, per-image IoU matching; AP integrates the precision envelope over
101 recall points at IoU .50:.05:.95. Operating recall is matched separately
after applying the locked confidence threshold. No detector inference is needed.
"""
from types import SimpleNamespace

import numpy as np
import torch


def evaluate_detections(rows, names, tau, *, model_id):
    import ultralytics
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    matcher = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    matches, scores, classes, targets = [], [], [], []
    counts = np.zeros(len(names), dtype=int)
    detected = counts.copy()
    operating_tp = counts.copy()
    for row in rows:
        gt = row.get("ground_truth_boxes", row.get("boxes", []))
        det = row["detections"]
        truth_cls = torch.tensor([b["class_id"] for b in gt], dtype=torch.float32)
        pred_cls = torch.tensor([b["class_id"] for b in det], dtype=torch.float32)
        truth_xy = torch.tensor([b["xyxy"] for b in gt], dtype=torch.float32).reshape(-1, 4)
        pred_xy = torch.tensor([b["xyxy_original"] for b in det], dtype=torch.float32).reshape(-1, 4)
        conf = np.asarray([b["confidence"] for b in det])
        iou = box_iou(truth_xy, pred_xy)
        correct = BaseValidator.match_predictions(matcher, pred_cls, truth_cls, iou).numpy()
        keep = conf >= tau
        op_match = BaseValidator.match_predictions(matcher, pred_cls[keep], truth_cls, iou[:, keep]).numpy()
        for cid in range(len(names)):
            counts[cid] += int((truth_cls == cid).sum())
            detected[cid] += int((pred_cls[keep] == cid).sum())
            operating_tp[cid] += int(op_match[pred_cls[keep].numpy() == cid, 0].sum())
        matches.append(correct)
        scores.extend(conf)
        classes.extend(pred_cls.tolist())
        targets.extend(truth_cls.tolist())
    ap_by_class = {}
    if scores and targets:
        metrics = ap_per_class(np.concatenate(matches), np.asarray(scores), np.asarray(classes),
                               np.asarray(targets), plot=False)
        ap_by_class = dict(zip(metrics[6], metrics[5]))
    return [dict(model_id=model_id, class_name=name, gt_box_count=int(counts[cid]),
                 detected_box_count=int(detected[cid]),
                 ap_50=float(ap_by_class.get(cid, np.zeros(10))[0]),
                 ap_50_95=float(np.mean(ap_by_class.get(cid, np.zeros(10)))),
                 class_recall=float(operating_tp[cid] / counts[cid]) if counts[cid] else 0.,
                 tau_yolo=tau, evaluator=f"ultralytics-{ultralytics.__version__}:iou-match/101-trapezoid")
            for cid, name in enumerate(names)]
