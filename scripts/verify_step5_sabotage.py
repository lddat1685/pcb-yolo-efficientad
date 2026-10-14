"""Reproducible, synthetic-only Step 5 sabotage log (Phase B)."""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from helpers.step5_spec import assert_routing_contract, config, rows, sweep_rows
from pcb_lab.fusion.evaluate import evaluate_fusion
from pcb_lab.fusion.metrics import compute_hybrid_metrics
from pcb_lab.fusion.rules import predict_cascade, predict_or
from pcb_lab.fusion.sweep import grid_search_2d


def observe(name, expected, action):
    try:
        value = action()
    except Exception as exc:
        return dict(case=name, expected=expected.__name__, observed=type(exc).__name__,
                    message=str(exc), passed=isinstance(exc, expected))
    return dict(case=name, expected=expected.__name__, observed="no exception",
                result=repr(value), passed=False)


def main():
    cases = []
    for function in (predict_cascade, predict_or):
        for side in ("yolo", "ad", "both"):
            for score in (0.1, 0.9):
                yolo, ad = rows(yolo=score)
                if side in {"yolo", "both"}:
                    yolo["split"] = "test"
                if side in {"ad", "both"}:
                    ad["split"] = "test"
                cases.append(observe(f"{function.__name__}: test in {side}; yolo={score}",
                    PermissionError, lambda: function(yolo, ad, 0.5, 0.5)))
        yolo, ad = rows()
        ad["sample_id"] = "mismatched-synthetic-id"
        cases.append(observe(f"{function.__name__}: swap sample_id", ValueError,
                             lambda: function(yolo, ad, 0.5, 0.5)))
    for side in ("yolo", "ad"):
        yolo, ad = sweep_rows()
        (yolo if side == "yolo" else ad)[0]["split"] = "test"
        cases.append(observe(f"grid: test in {side}", PermissionError,
                             lambda: grid_search_2d(yolo, ad, config(small=True))))
        cases.append(observe(f"evaluate: test in {side}", PermissionError,
                             lambda: evaluate_fusion(yolo, ad, {"tau_yolo": 0.5, "tau_ad": 0.5})))
    valid = predict_cascade(*rows(yolo=0.9), 0.5, 0.5)
    cases.append(observe("numeric AD score on an unrouted cascade decision", AssertionError,
                         lambda: assert_routing_contract(replace(valid, ad_score=0.73))))
    yolo, ad = rows(yolo=0.0, ad=0.0)
    yolo["error_reason"] = "synthetic inference failure"
    cases.append(observe("failed prediction is rejected", ValueError,
                         lambda: predict_cascade(yolo, ad, 0.5, 0.5)))
    metric_decisions = [predict_cascade(*rows("rescued", yolo=0.1, ad=0.9), 0.5, 0.5),
                        predict_cascade(*rows("missed", yolo=0.1, ad=0.1), 0.5, 0.5),
                        predict_cascade(*rows("false-reject", defect=False, yolo=0.1, ad=0.9), 0.5, 0.5)]
    metrics = compute_hybrid_metrics(metric_decisions)
    output = dict(synthetic_only=True, real_test_access=False, cases=cases,
                  hybrid_contract_call={"expected_rescue_rate": 0.5, "actual_rescue_rate": metrics["rescue_rate"],
                      "expected_added_false_rejects": 1, "actual_added_false_rejects": metrics["added_false_rejects"]})
    path = ROOT / "docs/handoff/step5-sabotage-log.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(cases)} observations; {sum(case['passed'] for case in cases)} met the required exception")
    print(path)


if __name__ == "__main__":
    main()
