"""Independent stdlib AP50 oracle: no pcb_lab or Ultralytics metric imports.

Match predictions, in descending confidence order, to unmatched same-class GT
within each image at IoU >= 0.5. Interpolate the precision envelope at 101 recall
levels (0..1 inclusive) and average. Ties retain JSONL/image order because raw
confidence cannot be recovered after serialization. Coordinates are continuous.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def linear_101_area(recall, precision):
    """Independent piecewise-linear envelope integration, matching report convention.

    Ultralytics 8.4.161 uses a 101-point trapezoid and zero terminal precision,
    rather than the arithmetic mean of the 101 step-envelope values above.
    """
    if not recall:
        return 0.0
    x = [0.0, *recall, recall[-1], 1.0]
    y = [1.0, *precision, 0.0, 0.0]
    for i in range(len(y) - 2, -1, -1):
        y[i] = max(y[i], y[i + 1])
    interpolated = []
    for i in range(101):
        query = i / 100
        left = bisect_right(x, query) - 1
        if left == len(x) - 1:
            value = y[-1]
        else:
            fraction = (query - x[left]) / (x[left + 1] - x[left])
            value = y[left] + fraction * (y[left + 1] - y[left])
        interpolated.append(value)
    return sum((a + b) / 200 for a, b in zip(interpolated, interpolated[1:]))


def ap50_101(predictions, ground_truth, names):
    truth = defaultdict(list)
    for row in ground_truth:
        for box in row["boxes"]:
            truth[(row["sample_id"], box["class_id"])].append(tuple(box["xyxy"]))
    per_class = {}
    for cid, name in enumerate(names):
        total = sum(len(boxes) for (_, key), boxes in truth.items() if key == cid)
        candidates = [(d["confidence"], row["sample_id"], d["xyxy_original"])
                      for row in predictions for d in row["detections"] if d["class_id"] == cid]
        candidates.sort(key=lambda item: -item[0])  # stable within serialized ties
        matched = defaultdict(set)
        precision, recall = [], []
        tp = 0
        for rank, (_, sample_id, box) in enumerate(candidates, 1):
            key = (sample_id, cid)
            available = [(iou(box, gt), index) for index, gt in enumerate(truth[key])
                         if index not in matched[key]]
            best_iou, best_index = max(available, default=(-1.0, -1))
            if best_iou >= 0.5:
                matched[key].add(best_index)
                tp += 1
            precision.append(tp / rank)
            recall.append(tp / total if total else 0.0)
        envelope = precision[:]
        for index in range(len(envelope) - 2, -1, -1):
            envelope[index] = max(envelope[index], envelope[index + 1])
        samples = []
        for level in range(101):
            index = bisect_left(recall, level / 100)
            samples.append(envelope[index] if index < len(envelope) else 0.0)
        per_class[name] = {"n_gt_boxes": total, "n_predictions": len(candidates),
                           "tp_at_conf_floor": tp, "ap50": sum(samples) / 101 if total else None,
                           "ap50_linear101": linear_101_area(recall, precision) if total else None}
    scored = [m["ap50"] for m in per_class.values() if m["ap50"] is not None]
    return {"method": "IoU>=0.5; confidence-greedy one-to-one; precision envelope mean at 101 recall levels",
            "overall_map50": sum(scored) / len(scored) if scored else None,
            "overall_map50_linear101": sum(m["ap50_linear101"] for m in per_class.values() if m["ap50_linear101"] is not None) / len(scored) if scored else None,
            "per_class": per_class}
