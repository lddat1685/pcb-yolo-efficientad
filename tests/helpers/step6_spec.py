"""Step 6 spec-first oracle: standard library only; never imports pcb_lab."""
from collections import Counter
import csv
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("B01", "B02", "B03", "H00", "H01", "H02")
METRICS = ("balanced_accuracy", "defect_recall", "defect_precision", "frr",
           "defect_to_good", "f1", "mcc", "ecpb")
ROUTING = ("route_rate_total", "route_rate_good", "route_rate_defect",
           "rescued_defects", "rescue_rate", "added_false_rejects", "unclassified_positive_rate")


def plain(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def artifact(name):
    return json.loads((ROOT / "artifacts/benchmark" / name).read_text(encoding="utf-8"))


def binary(tp, tn, fp, fn):
    recall = tp / (tp + fn) if tp + fn else None
    frr = fp / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
        balanced_accuracy=(recall + 1 - frr) / 2 if recall is not None and frr is not None else None,
        defect_recall=recall, defect_precision=precision, frr=frr,
        defect_to_good=1-recall if recall is not None else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        mcc=(tp*tn-fp*fn)/math.sqrt(denom) if denom else 0.0,
        ecpb=(100*(100*fn/(tp+fn)) + 9900*fp/(tn+fp))/10000 if tp+fn and tn+fp else None)


def oracle(samples, model, ty, ta):
    """samples: (ground truth, YOLO-n score, YOLO-s score, AD score)."""
    counts = Counter(tp=0, tn=0, fp=0, fn=0)
    n_good = sum(not s[0] for s in samples)
    n_defect = len(samples) - n_good
    routed = routed_good = routed_defect = rescued = added = misses = unclassified = positive = 0
    for defect, yn, ys, ad in samples:
        y = ys if model in {"B02", "H02"} else yn
        yhit, ahit = y >= ty, ad >= ta
        pred = yhit if model in {"B01", "B02"} else ahit if model == "B03" else yhit or ahit
        route = model in {"B03", "H00"} or model in {"H01", "H02"} and not yhit
        counts[("tp" if pred else "fn") if defect else ("fp" if pred else "tn")] += 1
        routed += route
        routed_good += route and not defect
        routed_defect += route and defect
        positive += pred
        if model.startswith("H"):
            misses += defect and not yhit
            rescued += defect and not yhit and ahit
            added += not defect and not yhit and ahit
            unclassified += not yhit and ahit
        elif model == "B03":
            unclassified += ahit
    result = binary(**counts)
    result.update(route_rate_total=routed/len(samples),
                  route_rate_good=routed_good/n_good if n_good else None,
                  route_rate_defect=routed_defect/n_defect if n_defect else None,
                  rescued_defects=rescued if model.startswith("H") else None,
                  rescue_rate=rescued/misses if model.startswith("H") and misses else None,
                  added_false_rejects=added if model.startswith("H") else None,
                  unclassified_positive_rate=unclassified/positive if positive else None)
    return result


def synthetic_samples(scenario="mixed"):
    if scenario == "perfect":
        return [(bool(i%2), 0.9 if i%2 else 0.1, 0.9 if i%2 else 0.1, 0.9 if i%2 else 0.1) for i in range(8)]
    if scenario == "total_fail":
        return [(bool(i%2), 0.1 if i%2 else 0.9, 0.1 if i%2 else 0.9, 0.1 if i%2 else 0.9) for i in range(8)]
    return [(True,.9,.9,.1), (True,.1,.9,.9), (True,.1,.1,.1), (True,.9,.1,.9),
            (False,.9,.1,.1), (False,.1,.1,.9), (False,.1,.9,.1), (False,.1,.1,.1)]


def recursive(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from recursive(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive(child)


def csv_rows(name):
    with (ROOT / "artifacts/benchmark" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
