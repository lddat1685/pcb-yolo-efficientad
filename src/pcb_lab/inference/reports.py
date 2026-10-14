"""Read-only Step 7 report loading and transparent comparisons."""
import csv
from dataclasses import dataclass
from io import StringIO
import math

from .registry import ModelRegistry, REPO_ROOT


@dataclass
class BenchmarkReport:
    frozen: dict
    summary: list[dict]
    per_group: list[dict]
    per_class: list[dict]
    csv_files: dict[str, bytes]

    def h00_h01(self):
        rows = {r["model_id"]: r for r in self.summary}
        before, after = rows["H00"], rows["H01"]
        return {"latency_reduction_pct": 100 * (1 - after["latency_mean_ms"] / before["latency_mean_ms"]),
                "recall_delta": after["defect_recall"] - before["defect_recall"],
                "frr_delta": after["frr"] - before["frr"],
                "ecpb_delta": after["ecpb"] - before["ecpb"]}


def load_benchmark(repo_root=REPO_ROOT):
    registry = ModelRegistry(repo_root)
    required = {
        "summary": {"model_id", "defect_recall", "frr", "ecpb", "latency_mean_ms", "latency_p95_ms", "balanced_accuracy", "f1"},
        "per_group": {"model_id", "source_group", "sample_count", "defect_recall", "frr", "worst_group_recall", "worst_group_frr"},
        "per_class": {"model_id", "class_name", "gt_box_count", "ap_50", "class_recall"},
    }
    tables, files = {}, {}
    for name, columns in required.items():
        filename = f"benchmark_{name}.csv"
        raw = (registry.repo_root / "artifacts/test" / filename).read_bytes()
        reader = csv.DictReader(StringIO(raw.decode("utf-8-sig")))
        if not columns <= set(reader.fieldnames or []):
            raise ValueError(f"Thiếu cột báo cáo: {filename}")
        rows = []
        for row in reader:
            converted = {}
            for key, value in row.items():
                if key in {"model_id", "class_name", "source_group"}:
                    converted[key] = value
                elif value in {"N/A", "", None}:
                    converted[key] = None
                else:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"Số liệu không hữu hạn: {filename}/{key}")
                    converted[key] = number
            if converted["model_id"] not in registry.specs:
                raise ValueError("Báo cáo chứa model ngoài registry đã khóa")
            if any(converted[k] is None for k in columns):
                raise ValueError(f"Thiếu số liệu bắt buộc: {filename}")
            rows.append(converted)
        if not rows:
            raise ValueError(f"Báo cáo rỗng: {filename}")
        identity = [tuple(r[k] for k in ("model_id",) + (("source_group",) if name == "per_group" else ("class_name",) if name == "per_class" else ())) for r in rows]
        if len(set(identity)) != len(identity):
            raise ValueError(f"Báo cáo lặp dòng: {filename}")
        tables[name], files[filename] = rows, raw
    if {r["model_id"] for r in tables["summary"]} != set(registry.specs):
        raise ValueError("Báo cáo thiếu model đã khóa")
    for row in tables["summary"]:
        if row["latency_mean_ms"] <= 0:
            raise ValueError("Latency benchmark phải dương")
        groups = [r for r in tables["per_group"] if r["model_id"] == row["model_id"]]
        if sum(r["sample_count"] for r in groups) != registry.frozen["sample_count"]:
            raise ValueError("Số mẫu nhóm không khớp freeze")
    return BenchmarkReport(registry.frozen, csv_files=files, **tables)


def pareto_models(rows):
    """Nondominated rows: maximize recall, minimize FRR/ECPB/mean latency."""
    def costs(r):
        return (-r["defect_recall"], r["frr"], r["ecpb"], r["latency_mean_ms"])
    result = []
    for row in rows:
        x = costs(row)
        if not any(all(a <= b for a, b in zip(costs(other), x)) and
                   any(a < b for a, b in zip(costs(other), x)) for other in rows):
            result.append(row["model_id"])
    return result
