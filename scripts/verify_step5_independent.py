"""Step 5 verification using only the Python standard library and saved data.

No pcb_lab imports, no dataset-root access, no images/checkpoints, no test split.
The label convention is explicitly checked: sample ID suffix must be good/defect.
This does not independently re-audit dataset ground truth against its manifest.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"yolo_n": "yolo/B01_yolo11n", "yolo_s": "yolo/B02_yolo11s",
          "ad": "efficientad/B03_efficientad_s"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_allowed(model, split):
    if split not in {"calibration", "fusion"}:
        raise PermissionError(split)
    path = ROOT / "artifacts" / MODELS[model] / "seed42" / f"preds_{split}.jsonl"
    data = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    by_id = {}
    for row in data:
        assert row["split"] == split
        assert row["sample_id"] not in by_id
        assert row["error_reason"] is None
        score = row["anomaly_score"] if model == "ad" else row["yolo_image_score"]
        assert math.isfinite(score)
        assert row["sample_id"].rsplit("_", 1)[1] in {"good", "defect"}
        by_id[row["sample_id"]] = row
    return path, by_id


def summarize(yolo, ad, ty, ta):
    assert yolo.keys() == ad.keys()
    defects = {sid for sid in yolo if sid.rsplit("_", 1)[1] == "defect"}
    goods = set(yolo) - defects
    yh = {sid for sid, row in yolo.items() if row["yolo_image_score"] >= ty}
    ah = {sid for sid, row in ad.items() if row["anomaly_score"] >= ta}
    hits = yh | ah
    tp, tn, fp, fn = len(hits & defects), len(goods - hits), len(goods & hits), len(defects - hits)
    fn_sim = 100 * Fraction(fn, len(defects))
    fp_sim = 9900 * Fraction(fp, len(goods))
    ec = (100 * fn_sim + fp_sim) / 10000
    rescue = len((defects - yh) & ah)
    misses = len(defects - yh)
    return dict(tau_yolo=ty, tau_ad=ta, n=len(yolo), n_good=len(goods), n_defect=len(defects),
                tp=tp, tn=tn, fp=fp, fn=fn, fn_sim=float(fn_sim), fp_sim=float(fp_sim),
                ecpb=float(ec), ecpb_exact=str(ec), frr=fp / len(goods),
                defect_recall=tp / len(defects), early_exit=len(yh),
                early_exit_good=len(yh & goods), early_exit_defect=len(yh & defects),
                routed=len(yolo) - len(yh), route_rate=(len(yolo) - len(yh)) / len(yolo),
                route_rate_good=len(goods - yh) / len(goods),
                route_rate_defect=len(defects - yh) / len(defects),
                rescued_defects=rescue, yolo_fn_count=misses,
                rescue_rate=rescue / misses if misses else None,
                added_false_rejects=len((goods - yh) & ah),
                unclassified_positive_rate=len(ah - yh) / len(hits),
                yolo_only_fp=len(yh & goods))


def rank(point):
    return (-point["defect_recall"], point["tau_yolo"], point["ecpb"])


def independent_plan_grid(yolo, ad, *, sample_std):
    scores = [row["anomaly_score"] for sid, row in ad.items() if sid.endswith("_defect")]
    mu = statistics.mean(scores)
    sigma = statistics.stdev(scores) if sample_std else statistics.pstdev(scores)
    points = [summarize(yolo, ad, (10 + 2 * yi) / 100, mu + 3 * sigma * ai / 49)
              for yi in range(41) for ai in range(50)]
    result = {"ad_min": mu, "ad_max": mu + 3 * sigma, "std_ddof": int(sample_std)}
    for key, frr in (("primary", 0.05), ("secondary", 0.01)):
        eligible = [point for point in points if point["frr"] <= frr]
        result[key] = min(eligible, key=rank) if eligible else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/step5-independent.json")
    args = parser.parse_args()
    thresholds = json.loads((ROOT / "artifacts/fusion/thresholds.json").read_text("utf-8"))
    policy = json.loads((ROOT / "artifacts/fusion/routing_policy.json").read_text("utf-8"))
    grid = json.loads((ROOT / "artifacts/fusion/grid_calibration.json").read_text("utf-8"))
    loaded, sources = {}, {}
    for model in MODELS:
        for split in ("calibration", "fusion"):
            path, records = read_allowed(model, split)
            key = f"{model}_{split}"
            loaded[key] = records
            sha = digest(path)
            assert sha == policy["source_checksums"][key]
            sources[key] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha, rows=len(records))
    for split in ("calibration", "fusion"):
        ad = loaded[f"ad_{split}"]
        for model in ("yolo_n", "yolo_s"):
            yolo = loaded[f"{model}_{split}"]
            assert yolo.keys() == ad.keys()
            for sid in yolo:
                assert yolo[sid]["image_sha256"] == ad[sid]["image_sha256"]
                assert yolo[sid]["source_group"] == ad[sid]["source_group"]
    assert not (loaded["ad_calibration"].keys() & loaded["ad_fusion"].keys())

    output = {"oracle": "Python stdlib; no pcb_lab import", "sources": sources,
              "label_source": "validated good/defect suffix in upstream prediction sample_id; no dataset manifest read",
              "locked": {}, "artifact_value_mismatches": [], "grid_replay_mismatches": []}
    for variant, model in (("h01_yolo11n_cascade", "yolo_n"), ("h02_yolo11s_cascade", "yolo_s")):
        locked = thresholds[variant]
        output["locked"][variant] = {}
        for split in ("calibration", "fusion"):
            result = summarize(loaded[f"{model}_{split}"], loaded[f"ad_{split}"],
                               locked["tau_yolo"], locked["tau_ad"])
            output["locked"][variant][split] = result
            recorded = locked["calibration_metrics"] if split == "calibration" else locked["fusion_eval"]["h01_cascade"]
            for key in ("tp", "tn", "fp", "fn", "ecpb", "route_rate", "route_rate_good", "route_rate_defect"):
                if not math.isclose(result[key], recorded[key], abs_tol=1e-12):
                    output["artifact_value_mismatches"].append(dict(variant=variant, split=split, key=key,
                                                                   actual=recorded[key], expected=result[key]))
            if split == "fusion":
                for key in ("rescued_defects", "rescue_rate", "added_false_rejects"):
                    assert result[key] == recorded[key]
                for key in ("tp", "tn", "fp", "fn", "ecpb"):
                    assert result[key] == locked["fusion_eval"]["h00_or"][key]
    recounted = []
    for point in grid:
        independent = summarize(loaded["yolo_n_calibration"], loaded["ad_calibration"],
                                point["tau_yolo"], point["tau_ad"])
        recounted.append(independent)
        for key in ("tp", "tn", "fp", "fn"):
            if independent[key] != point[key]:
                output["grid_replay_mismatches"].append(dict(tau_yolo=point["tau_yolo"],
                    tau_ad=point["tau_ad"], key=key, actual=point[key], expected=independent[key]))
    output["saved_grid"] = {"count": len(grid), "ad_min": min(p["tau_ad"] for p in grid),
                            "ad_max": max(p["tau_ad"] for p in grid)}
    for key, frr in (("primary", 0.05), ("secondary", 0.01)):
        eligible = [p for p in recounted if p["frr"] <= frr]
        output["saved_grid"][key] = {"eligible": len(eligible), "optimal": min(eligible, key=rank)}
    output["plan_grid_oracle"] = {model: independent_plan_grid(loaded[f"{model}_calibration"],
                                  loaded["ad_calibration"], sample_std=True)
                                  for model in ("yolo_n", "yolo_s")}
    output["plan_grid_population_std_sensitivity"] = independent_plan_grid(
        loaded["yolo_n_calibration"], loaded["ad_calibration"], sample_std=False)
    output["replay_byte_identical"] = {}
    for filename in ("thresholds.json", "routing_policy.json", "grid_calibration.json"):
        replay = ROOT / ".cache/step5-replay/artifacts" / filename
        output["replay_byte_identical"][filename] = digest(replay) == digest(ROOT / "artifacts/fusion" / filename)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
