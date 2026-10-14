"""Step 6 development benchmark: unified evaluation of B01/B02/B03/H00/H01/H02.

Loads static prediction files (calibration + fusion) and the locked Step 5
thresholds, then evaluates every mandatory baseline with the Step 5 engine
(``pcb_lab.fusion.evaluate.evaluate_split``). 100 % static, deterministic,
no GPU forward. Any ``test``-partition row raises ``PermissionError``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pcb_lab.fusion.evaluate import evaluate_split

# Mandatory baseline matrix (contract §1).
METHODS = ("B01", "B02", "B03", "H00", "H01", "H02")
SPLITS = ("calibration", "fusion")

PRED_FILENAMES = {
    "yolo_n": "artifacts/yolo/B01_yolo11n/seed42",
    "yolo_s": "artifacts/yolo/B02_yolo11s/seed42",
    "ad": "artifacts/efficientad/B03_efficientad_s/seed42",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load a static preds_*.jsonl file; refuse the test partition."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file missing: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") == "test":
                raise PermissionError(
                    f"Test split detected at {path}:{lineno}; "
                    "Step 6 must not access test (test_access: False)"
                )
            rows.append(row)
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def derive_optimum(
    grid_points: List[Dict[str, Any]],
    frr_max: float,
) -> Dict[str, Any]:
    """Pick the grid point with max Defect Recall s.t. FRR <= frr_max.

    Tie-break: lower tau_yolo (early-exit), then lower ECPB. Same rule as
    Step 5 sweep for the primary/secondary constraints.
    """
    feas = [p for p in grid_points if p["frr"] <= frr_max + 1e-12]
    if not feas:
        raise ValueError(f"No grid point satisfies FRR <= {frr_max}")
    feas.sort(key=lambda p: (-p["defect_recall"], p["tau_yolo"], p["ecpb"]))
    return feas[0]


@dataclass
class BenchmarkResult:
    """Full development benchmark: metrics per split x method."""

    recipe_id: str
    target_frr: float
    secondary_frr_max: float
    cost_matrix: Dict[str, float]
    operating_points: Dict[str, Dict[str, float]]
    splits: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    source_checksums: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    test_access: bool = False
    repo_root: Optional[Path] = None
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "recipe_id": self.recipe_id,
            "selection_split": "fusion",
            "calibration_split": "calibration",
            "target_frr": self.target_frr,
            "secondary_frr_max": self.secondary_frr_max,
            "test_access": self.test_access,
            "cost_matrix": self.cost_matrix,
            "operating_points": self.operating_points,
            "splits": self.splits,
            "source_checksums": self.source_checksums,
            "counts": self.counts,
        }

    def to_rows(self) -> List[Dict[str, Any]]:
        """Flatten to one row per (split, method) for CSV export."""
        rows: List[Dict[str, Any]] = []
        for split in SPLITS:
            for method in METHODS:
                entry = dict(self.splits.get(split, {}).get(method, {}))
                entry["split"] = split
                entry["method"] = method
                prov = self.provenance.get(method, {})
                entry["dataset_id"] = prov.get("dataset_id", "deeppcb")
                entry["protocol_id"] = prov.get("protocol_id", "holdout_v1")
                entry["seed"] = prov.get("seed", 42)
                entry["device"] = prov.get("device", "cpu")
                entry["artifact_hash"] = prov.get("artifact_hash", "")
                entry["config_hash"] = prov.get("config_hash", "")
                rows.append(entry)
        return rows


def _resolve_pred(repo_root: Path, subdir: str, split: str) -> Path:
    return repo_root / subdir / f"preds_{split}.jsonl"


def run_development_benchmark(
    config_path: str | Path,
    thresholds_path: str | Path,
    policy_path: str | Path,
    repo_root: Optional[str | Path] = None,
) -> BenchmarkResult:
    """Evaluate B01/B02/B03/H00/H01/H02 on calibration + fusion at locked points.

    ``config_path``/``thresholds_path``/``policy_path`` may be absolute or
    relative to ``repo_root`` (default: inferred from this file's location).
    Returns a :class:`BenchmarkResult`.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    def _abs(p: str | Path) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else root / pp

    cfg_path = _abs(config_path)
    thr_path = _abs(thresholds_path)
    pol_path = _abs(policy_path)
    if not pol_path.is_file():
        raise FileNotFoundError(f"Policy file missing: {pol_path}")
    with open(pol_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
    if policy.get("test_access") is not False:
        raise ValueError("Policy has test_access enabled; Step 6 requires test_access: False")

    if not thr_path.is_file():
        raise FileNotFoundError(f"Thresholds file missing: {thr_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(thr_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    cost_params = {
        "c_fn": float(config["cost_matrix"]["c_fn"]),
        "c_fp": float(config["cost_matrix"]["c_fp"]),
        "defect_prevalence": float(config["cost_matrix"]["defect_prevalence"]),
        "n_simulation": float(config["cost_matrix"]["n_simulation"]),
    }
    target_frr = float(config["constraints"]["primary_frr_max"])
    secondary_frr_max = float(config["constraints"]["secondary_frr_max"])

    h01 = thresholds["h01_yolo11n_cascade"]
    h02 = thresholds["h02_yolo11s_cascade"]
    h00 = thresholds["h00_yolo11n_or"]
    if not (h00["tau_yolo"] == h01["tau_yolo"] and h00["tau_ad"] == h01["tau_ad"]):
        raise ValueError("H00 must share the H01 operating point (contract)")

    if "tau_yolo" in policy and abs(float(policy["tau_yolo"]) - float(h01["tau_yolo"])) > 1e-6:
        raise ValueError(f"Policy tau_yolo ({policy['tau_yolo']}) does not match thresholds.json ({h01['tau_yolo']})")
    if "tau_ad" in policy and abs(float(policy["tau_ad"]) - float(h01["tau_ad"])) > 1e-6:
        raise ValueError(f"Policy tau_ad ({policy['tau_ad']}) does not match thresholds.json ({h01['tau_ad']})")

    operating_points = {
        "B01": {"tau_yolo": float(h01["tau_yolo"]), "tau_ad": None},
        "B02": {"tau_yolo": float(h02["tau_yolo"]), "tau_ad": None},
        "B03": {"tau_yolo": None, "tau_ad": float(h01["tau_ad"])},
        "H00": {"tau_yolo": float(h01["tau_yolo"]), "tau_ad": float(h01["tau_ad"])},
        "H01": {"tau_yolo": float(h01["tau_yolo"]), "tau_ad": float(h01["tau_ad"])},
        "H02": {"tau_yolo": float(h02["tau_yolo"]), "tau_ad": float(h02["tau_ad"])},
    }

    # Secondary operating point (FRR<=1%) from the locked Step 5 grid stored
    # next to thresholds.json; same selection rule, no new fitting.
    grid_path = thr_path.parent / "grid_calibration.json"
    if grid_path.is_file():
        with open(grid_path, "r", encoding="utf-8") as f:
            grid_points = json.load(f)
        sec = derive_optimum(grid_points, secondary_frr_max)
        operating_points["H01_secondary"] = {
            "tau_yolo": float(sec["tau_yolo"]),
            "tau_ad": float(sec["tau_ad"]),
        }
    else:
        sec = None

    # Load artifacts and manifests for provenance & validation
    metas: Dict[str, Dict[str, Any]] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    expected_models = {
        "yolo_n": "B01",
        "yolo_s": "B02",
        "ad": "B03",
    }

    for key, subdir in PRED_FILENAMES.items():
        meta_path = root / subdir / "artifact.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                metas[key] = json.load(f)
            man_rel = metas[key].get("run_manifest", {}).get("path")
            if man_rel and (root / man_rel).is_file():
                with open(root / man_rel, "r", encoding="utf-8") as f:
                    manifests[key] = json.load(f)
        else:
            man_path = root / "runs" / Path(subdir).parent.name / Path(subdir).name / "seed42" / "run_manifest.json"
            if man_path.is_file():
                with open(man_path, "r", encoding="utf-8") as f:
                    manifests[key] = json.load(f)

    # Load the six static prediction files.
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    checksums: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for key, subdir in PRED_FILENAMES.items():
        meta = metas.get(key, {})
        meta_model = meta.get("model_id")
        meta_ckpt = meta.get("checkpoint_sha256") or meta.get("checkpoint_best_sha256")
        meta_cfg = meta.get("config_hash")
        exp_prefix = expected_models[key]

        for split in SPLITS:
            p = _resolve_pred(root, subdir, split)
            rows = load_predictions(p)
            digest = sha256_file(p)
            checksums[f"{key}_{split}"] = digest
            counts[f"{key}_{split}"] = len(rows)
            loaded[f"{key}_{split}"] = rows

            # Validate policy checksum
            policy_chk = policy.get("source_checksums", {}).get(f"{key}_{split}")
            if policy_chk and policy_chk != digest:
                raise ValueError(
                    f"Checksum mismatch for {key}_{split}: policy={policy_chk}, actual={digest}"
                )

            # Validate model identity and checkpoint sha256
            for r in rows:
                mid = r.get("model_id", "")
                if not mid.startswith(exp_prefix):
                    raise ValueError(f"Model ID prefix mismatch: expected {exp_prefix}, got {mid}")
                if meta_model and mid != meta_model:
                    raise ValueError(f"Model ID mismatch: expected {meta_model}, got {mid}")
                if meta_ckpt and r.get("checkpoint_sha256") != meta_ckpt:
                    raise ValueError(f"Checkpoint sha256 mismatch for {mid}")
                if meta_cfg and r.get("config_hash") != meta_cfg:
                    raise ValueError(f"Config hash mismatch for {mid}")

    n_cal = len(loaded["yolo_n_calibration"])
    n_fus = len(loaded["yolo_n_fusion"])
    if n_cal == 0 or n_fus == 0:
        raise ValueError(f"Unexpected empty split sizes: calibration={n_cal}, fusion={n_fus}")
    for key in ("yolo_s", "ad"):
        if len(loaded[f"{key}_calibration"]) != n_cal:
            raise ValueError(f"Calibration row count mismatch for {key}: {len(loaded[f'{key}_calibration'])} vs {n_cal}")
        if len(loaded[f"{key}_fusion"]) != n_fus:
            raise ValueError(f"Fusion row count mismatch for {key}: {len(loaded[f'{key}_fusion'])} vs {n_fus}")

    # Build provenance metadata per PLAN §7.2
    provenance: Dict[str, Dict[str, Any]] = {}
    for m in METHODS:
        if m in ("B01", "H00", "H01"):
            src_k = "yolo_n"
        elif m in ("B02", "H02"):
            src_k = "yolo_s"
        else:
            src_k = "ad"
        meta = metas.get(src_k, {})
        man = manifests.get(src_k, {})
        art_hash = (
            meta.get("checkpoint_sha256")
            or meta.get("checkpoint_best_sha256")
            or checksums.get(f"{src_k}_fusion", "")
        )
        cfg_hash = meta.get("config_hash") or checksums.get(f"{src_k}_calibration", "")
        dev = str(man.get("device", "cuda:0" if m == "B03" else "cpu"))
        if dev == "0":
            dev = "cuda:0"
        provenance[m] = {
            "dataset_id": "deeppcb",
            "protocol_id": "holdout_v1",
            "seed": int(man.get("seed", 42)),
            "device": dev,
            "artifact_hash": art_hash,
            "config_hash": cfg_hash,
        }

    result = BenchmarkResult(
        recipe_id=config.get("recipe_id", "fusion_recipe_v1"),
        target_frr=target_frr,
        secondary_frr_max=secondary_frr_max,
        cost_matrix=cost_params,
        operating_points=operating_points,
        source_checksums=checksums,
        counts=counts,
        repo_root=root,
        provenance=provenance,
    )

    for split in SPLITS:
        yn = loaded[f"yolo_n_{split}"]
        ys = loaded[f"yolo_s_{split}"]
        ad = loaded[f"ad_{split}"]

        ev_n = evaluate_split(
            yn, ad, h01["tau_yolo"], h01["tau_ad"], cost_params
        )
        ev_s = evaluate_split(
            ys, ad, h02["tau_yolo"], h02["tau_ad"], cost_params
        )

        b01_dict = dict(ev_n["yolo_only"])
        b01_dict.update(
            tau_yolo=h01["tau_yolo"],
            tau_ad=None,
            route_rate_total=0.0,
            route_rate=0.0,
            route_rate_good=0.0,
            route_rate_defect=0.0,
            rescued_defects=None,
            rescue_rate=None,
            added_false_rejects=None,
            unclassified_positive_rate=0.0,
            mean_latency_ms=None,
        )

        b02_dict = dict(ev_s["yolo_only"])
        b02_dict.update(
            tau_yolo=h02["tau_yolo"],
            tau_ad=None,
            route_rate_total=0.0,
            route_rate=0.0,
            route_rate_good=0.0,
            route_rate_defect=0.0,
            rescued_defects=None,
            rescue_rate=None,
            added_false_rejects=None,
            unclassified_positive_rate=0.0,
            mean_latency_ms=None,
        )

        b03_dict = dict(ev_n["ad_only"])
        b03_pos = b03_dict["tp"] + b03_dict["fp"]
        b03_dict.update(
            tau_yolo=None,
            tau_ad=h01["tau_ad"],
            route_rate_total=1.0,
            route_rate=1.0,
            route_rate_good=1.0,
            route_rate_defect=1.0,
            rescued_defects=None,
            rescue_rate=None,
            added_false_rejects=None,
            unclassified_positive_rate=1.0 if b03_pos > 0 else None,
            mean_latency_ms=None,
        )

        h00_dict = dict(ev_n["h00_or"])
        h00_dict.update(
            tau_yolo=h01["tau_yolo"],
            tau_ad=h01["tau_ad"],
            route_rate_total=h00_dict.get("route_rate", 1.0),
        )

        h01_dict = dict(ev_n["h01_cascade"])
        h01_dict.update(
            tau_yolo=h01["tau_yolo"],
            tau_ad=h01["tau_ad"],
            route_rate_total=h01_dict.get("route_rate", 0.0),
        )

        h02_dict = dict(ev_s["h01_cascade"])
        d_rec = h02_dict["defect_recall"] - h00_dict["defect_recall"]
        d_frr = h02_dict["frr"] - h00_dict["frr"]
        d_ecpb = h02_dict["ecpb"] - h00_dict["ecpb"]
        deltas_h02_h00 = {
            "delta_recall": d_rec,
            "delta_frr": d_frr,
            "delta_ecpb": d_ecpb,
            "delta_recall_H01_minus_H00": d_rec,
            "delta_FRR_H01_minus_H00": d_frr,
            "delta_ECPB_H01_minus_H00": d_ecpb,
            "delta_latency_ms_H01_minus_H00": None,
        }
        h02_dict.update(
            tau_yolo=h02["tau_yolo"],
            tau_ad=h02["tau_ad"],
            route_rate_total=h02_dict.get("route_rate", 0.0),
            deltas_H02_minus_H00=deltas_h02_h00,
        )

        methods: Dict[str, Dict[str, Any]] = {
            "B01": b01_dict,
            "B02": b02_dict,
            "B03": b03_dict,
            "H00": h00_dict,
            "H01": h01_dict,
            "H02": h02_dict,
        }
        # Consistency cross-check vs the locked Step 5 fusion numbers.
        if split == "fusion":
            for lock_key, mkey, sub in (
                ("h01_yolo11n_cascade", "H01", "h01_cascade"),
                ("h01_yolo11n_cascade", "H00", "h00_or"),
                ("h02_yolo11s_cascade", "H02", "h01_cascade"),
            ):
                locked = thresholds[lock_key]["fusion_eval"][sub]
                for metric in ("balanced_accuracy", "defect_recall",
                               "defect_precision", "frr", "f1", "mcc", "ecpb"):
                    if abs(methods[mkey][metric] - locked[metric]) > 1e-9:
                        raise ValueError(
                            f"Provenance mismatch {split}/{mkey}/{metric}: "
                            f"{methods[mkey][metric]} vs locked {locked[metric]}"
                        )
        if sec is not None:
            ev_sec = evaluate_split(
                yn, ad, sec["tau_yolo"], sec["tau_ad"], cost_params
            )
            methods["H01_secondary"] = dict(
                ev_sec["h01_cascade"],
                tau_yolo=sec["tau_yolo"], tau_ad=sec["tau_ad"],
                route_rate_total=ev_sec["h01_cascade"].get("route_rate", 0.0),
            )
            methods["H00_secondary"] = dict(
                ev_sec["h00_or"],
                tau_yolo=sec["tau_yolo"], tau_ad=sec["tau_ad"],
                route_rate_total=ev_sec["h00_or"].get("route_rate", 1.0),
            )
        result.splits[split] = methods

    return result
