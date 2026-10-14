"""Verifier oracle written from Step 5 contract and PLAN, before reading impl.

Input score names come from Step 3/4 extraction contracts. Ground truth is
synthetic; no dataset manifest, image, or test-partition metadata is loaded.
"""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[2]
METRICS = {"balanced_accuracy", "defect_recall", "defect_precision", "frr",
           "defect_to_good", "f1", "mcc", "ecpb"}


def rows(sample_id="synthetic", *, split="calibration", defect=True,
         yolo=0.2, ad=0.8):
    common = {"sample_id": sample_id, "split": split, "is_defect": defect,
              "error_reason": None, "timing_ms": None}
    return (dict(common, yolo_image_score=yolo, detections=[]),
            dict(common, anomaly_score=ad, boxes=[]))


def config(*, small=False):
    return {
        "recipe_id": "fusion_recipe_v1", "calibration_split": "calibration",
        "development_split": "fusion",
        "grid_search": {"yolo_min": 0.10, "yolo_max": 0.90,
                        "yolo_step": 0.40 if small else 0.02,
                        "ad_num_points": 5 if small else 50},
        "constraints": {"primary_frr_max": 0.05, "secondary_frr_max": 0.01},
        "cost_matrix": {"c_fn": 100.0, "c_fp": 1.0,
                        "defect_prevalence": 0.01, "n_simulation": 10000},
    }


def plain(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def point_metrics(point):
    # The spec does not pin whether grid metrics are nested or flattened.
    data = plain(point)
    return data.get("metrics", data.get("calibration_metrics", data))


def load_artifact(name):
    return json.loads((ROOT / "artifacts" / "fusion" / name).read_text("utf-8"))


def walk(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def assert_routing_contract(decision):
    assert decision.decision in {"GOOD", "DEFECT"}
    assert decision.mode in {"CASCADE", "OR"}
    if decision.mode == "CASCADE" and not decision.route_ad:
        assert decision.ad_score is None, "Unrouted AD score must be None"
        assert decision.reason == "yolo_hit"
        assert decision.decision == "DEFECT"
        assert decision.ad_boxes == []
    if decision.mode == "OR":
        assert decision.route_ad is True
    if decision.route_ad:
        assert isinstance(decision.ad_score, (float, int))
        assert math.isfinite(decision.ad_score)


def confusion_oracle(yolo_rows, ad_rows, tau_yolo, tau_ad):
    ad_by_id = {row["sample_id"]: row for row in ad_rows}
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for row in yolo_rows:
        predicted = (row["yolo_image_score"] >= tau_yolo or
                     ad_by_id[row["sample_id"]]["anomaly_score"] >= tau_ad)
        key = ("tp" if predicted else "fn") if row["is_defect"] else (
            "fp" if predicted else "tn")
        counts[key] += 1
    return counts


def ec_pb(counts):
    # Explicit 10,000-board simulation, independent of implementation metrics.
    fn_sim = 100 * counts["fn"] / (counts["tp"] + counts["fn"])
    fp_sim = 9900 * counts["fp"] / (counts["tn"] + counts["fp"])
    return (100 * fn_sim + fp_sim) / 10000


def sweep_rows(*, yolo_false_positive=False):
    pairs = [rows(f"good-{i:02}", defect=False,
                  yolo=0.3 if yolo_false_positive and i == 0 else 0.0,
                  ad=1.4 if i == 0 else 0.0) for i in range(20)]
    pairs += [rows("defect-low", yolo=0.95, ad=0.2),
              rows("defect-high", yolo=0.95, ad=1.8)]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
