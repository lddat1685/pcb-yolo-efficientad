"""Independent Phase B statistics, fault injection, and pre-save runtime audit.

Set STEP4_TEACHER_PATH to a cached pretrained PDN-S to enable the real CPU smoke.
No download is allowed. The dataset is read-only; test pixels are never opened.
"""
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def linear_quantile(values, probability):
    """Sorted order-statistic oracle, independent of np.quantile/torch.quantile."""
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return float(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))


@pytest.mark.parametrize("statistic,split", [("teacher_statistics", "train"),
                                            ("map_statistics", "calibration")])
@pytest.mark.parametrize("corruption", ["defect", "boxes", "test"])
@pytest.mark.parametrize("late", [False, True], ids=["before_iteration", "during_iteration"])
def test_statistics_reject_contamination(statistic, split, corruption, late):
    import pcb_lab.models.efficientad.train as module
    from pcb_lab.models.efficientad.dataset import DataLeakError
    good = SimpleNamespace(sample_id="good", partition=split, is_defect=False, boxes=())
    bad = SimpleNamespace(**vars(good))
    if corruption == "defect":
        bad.is_defect = True
    elif corruption == "boxes":
        bad.boxes = (object(),)
    else:
        bad.partition = "test"

    def iterate():
        if not late:
            pytest.fail("Contaminated metadata reached iteration/decode")
        yield bad, object()  # require_good must fire before tile.to/model execution.

    dataset = SimpleNamespace(samples=[good if late else bad], iter_tiles=iterate)
    args = (SimpleNamespace(eval=lambda: None), dataset, "cpu")
    if statistic == "map_statistics":
        args += ((.9, .995),)
    with pytest.raises(DataLeakError):
        getattr(module, statistic)(*args)


def test_teacher_statistics_matches_population_oracle():
    import torch
    from pcb_lab.models.efficientad.train import teacher_statistics
    good = SimpleNamespace(sample_id="train_good", partition="train", is_defect=False, boxes=())
    values = [torch.tensor([[[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]]]),
              torch.tensor([[[[5., 6.], [7., 8.]], [[50., 60.], [70., 80.]]]])]

    class Source:
        samples = [good]
        num_tiles = 2

        def __len__(self):
            return 1

        def iter_tiles(self):
            return iter((good, value) for value in values)

    model = SimpleNamespace(teacher_features=lambda tile: tile,
        teacher_mean=torch.zeros(1, 2, 1, 1), teacher_std=torch.zeros(1, 2, 1, 1))
    audit = teacher_statistics(model, Source(), "cpu")
    np.testing.assert_allclose(model.teacher_mean.numpy().ravel(), [4.5, 45.], atol=1e-6, rtol=0)
    np.testing.assert_allclose(model.teacher_std.numpy().ravel(),
                               [math.sqrt(5.25), math.sqrt(525.)], atol=1e-6, rtol=0)
    assert audit["n_feature_pixels_per_channel"] == 8
    assert audit["sample_ids"] == [good.sample_id]


def test_map_statistics_matches_sorted_quantile_oracle():
    import torch
    from pcb_lab.models.efficientad.train import map_statistics
    good = SimpleNamespace(sample_id="cal_good", partition="calibration", is_defect=False, boxes=())
    values = np.array([[[[0, 1], [2, 3]], [[10, 20], [30, 40]]],
                       [[[4, 5], [6, 7]], [[50, 60], [70, 80]]]], dtype=np.float32)

    class Source:
        samples = [good]
        num_tiles = 2

        def __len__(self):
            return 1

        def iter_tiles(self):
            return iter((good, torch.from_numpy(value[None])) for value in values)

    model = SimpleNamespace(eval=lambda: None, raw_maps=lambda tile: tile,
                            map_quantiles=torch.zeros(2, 2))
    audit = map_statistics(model, Source(), "cpu", (.9, .995))
    expected = [[6.3, 6.965], [73., 79.65]]
    np.testing.assert_allclose(audit["branch_quantiles"], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.map_quantiles.numpy(), expected, atol=4e-6, rtol=0)
    combined = [(a + b) / 2 for a, b in zip(range(8), range(10, 81, 10))]
    assert audit["quantile_min"] == pytest.approx(linear_quantile(combined, .9), abs=1e-12)
    assert audit["quantile_max"] == pytest.approx(linear_quantile(combined, .995), abs=1e-12)
    assert audit["mean"] == pytest.approx(24.25, abs=1e-12)
    assert audit["std"] == pytest.approx(math.sqrt(158.8125), abs=1e-12)
    assert audit["n_map_pixels_per_branch"] == 8


def test_real_cpu_smoke_audits_sources_statistics_and_first_save(tmp_path, monkeypatch, real_dataset_root):
    teacher_path = os.environ.get("STEP4_TEACHER_PATH")
    if not teacher_path:
        pytest.skip("Set STEP4_TEACHER_PATH to cached pretrained weights for independent CPU smoke")
    assert Path(teacher_path).is_file(), teacher_path
    import torch
    import yaml
    import pcb_lab.models.efficientad.train as runner
    from pcb_lab.data.samples import Sample, EvalSet
    from pcb_lab.models.efficientad.adapter import EfficientAdDetector, EfficientAdEngine

    monkeypatch.setenv("DATASET_ROOT", str(real_dataset_root))
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs/models/efficientad_s.yaml").read_text(encoding="utf-8"))
    cfg["pretrained_teacher"] = str(Path(teacher_path).resolve())
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    records = [json.loads(line) for line in (real_dataset_root /
        "benchmarks/deeppcb/manifests/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["sample_id"]: r for r in records}
    counts = {split: {kind: sum(r["split"] == split and r["is_defect"] == defect for r in records)
        for kind, defect in (("good", False), ("defect", True))}
        for split in ("train", "calibration", "fusion")}
    assert counts == {"train": {"good": 895, "defect": 897},
                      "calibration": {"good": 230, "defect": 230},
                      "fusion": {"good": 153, "defect": 153}}
    state = {"stage": "setup", "decoded": {}, "counts": counts}
    original_load = Sample.load_image

    def checked_load(sample):
        assert sample.partition in ("train", "calibration", "fusion"), "No test pixels"
        stage = state["stage"]
        state["decoded"].setdefault(stage, []).append(sample.sample_id)
        if stage in ("teacher", "optimizer", "map_stats"):
            assert not sample.is_defect and not sample.boxes
            assert sample.partition == ("calibration" if stage == "map_stats" else "train")
        return original_load(sample)

    monkeypatch.setattr(Sample, "load_image", checked_load)
    original_teacher, original_maps, original_save = runner.teacher_statistics, runner.map_statistics, runner._save

    def teacher_audit(model, dataset, device):
        state["stage"] = "teacher"
        features = []
        hook = model.teacher.register_forward_hook(lambda _m, _args, result:
            features.append(result[:, [0, 191, 383]].detach().cpu().numpy().copy()))
        try:
            audit = original_teacher(model, dataset, device)
        finally:
            hook.remove()
        # Concatenate observations and use centered population variance, unlike
        # implementation's streaming E[X^2] - E[X]^2.
        channels = np.concatenate([a.transpose(1, 0, 2, 3).reshape(3, -1) for a in features], axis=1).astype(np.float64)
        expected_mean, expected_std = channels.mean(axis=1), channels.std(axis=1)
        actual_mean = model.teacher_mean[0, [0, 191, 383], 0, 0].cpu().numpy()
        actual_std = model.teacher_std[0, [0, 191, 383], 0, 0].cpu().numpy()
        np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-6, rtol=0)
        np.testing.assert_allclose(actual_std, expected_std, atol=1e-6, rtol=0)
        assert channels.shape[1] == 4 * 9 * 56 * 56 == audit["n_feature_pixels_per_channel"]
        state["teacher"] = {"mean_channel0": float(expected_mean[0]), "std_channel0": float(expected_std[0]),
            "mean_max_abs": float(np.max(np.abs(actual_mean - expected_mean))),
            "std_max_abs": float(np.max(np.abs(actual_std - expected_std))), "pixels": channels.shape[1]}
        state["stage"] = "optimizer"
        return audit

    def maps_audit(model, dataset, device, probabilities):
        state["stage"] = "map_stats"
        state["live_model"] = model
        maps = []
        original_raw = model.raw_maps

        def observed_raw(tile):
            result = original_raw(tile)
            maps.append(result.detach().cpu().numpy().copy())
            return result

        with monkeypatch.context() as patch:
            patch.setattr(model, "raw_maps", observed_raw)
            audit = original_maps(model, dataset, device, probabilities)
        branches = np.concatenate([a.transpose(1, 0, 2, 3).reshape(2, -1) for a in maps], axis=1).astype(np.float64)
        expected = [[linear_quantile(branch, q) for q in probabilities] for branch in branches]
        np.testing.assert_allclose(audit["branch_quantiles"], expected, atol=1e-12, rtol=0)
        combined = (branches[0] + branches[1]) / 2
        for key, expected_value in (("quantile_min", linear_quantile(combined, .9)),
                                    ("quantile_max", linear_quantile(combined, .995)),
                                    ("mean", float(combined.mean())), ("std", float(combined.std()))):
            assert audit[key] == pytest.approx(expected_value, abs=1e-12, rel=0)
        assert audit["n_map_pixels_per_branch"] == 112896
        state["normalization"] = {key: audit[key] for key in
            ("quantile_min", "quantile_max", "mean", "std", "branch_quantiles", "sample_ids")}
        state["stage"] = "postfit"
        return audit

    def save_audit(path, payload):
        if path.name == "model.pt":
            assert not path.exists(), "Probe must precede FIRST model.pt save"
            state["stage"] = "pre_save"
            samples = sorted(EvalSet(real_dataset_root, "calibration"), key=lambda s: s.sample_id)
            selected = [next(s for s in samples if s.is_defect), next(s for s in samples if not s.is_defect)]
            selected += [s for s in samples if s.sample_id not in {x.sample_id for x in selected}][:3]
            state["probe_ids"] = [s.sample_id for s in selected]
            state["probe_images"] = [s.load_image() for s in selected]
            detector = EfficientAdDetector(EfficientAdEngine(state["live_model"], "cpu"), payload)
            state["pre_save"] = [detector.predict(image) for image in state["probe_images"]]
            state["stage"] = "report_reload"
        return original_save(path, payload)

    monkeypatch.setattr(runner, "teacher_statistics", teacher_audit)
    monkeypatch.setattr(runner, "map_statistics", maps_audit)
    monkeypatch.setattr(runner, "_save", save_audit)
    result = runner.train_efficientad(config_path, out_root=tmp_path / "output", device="cpu",
                                     smoke=True, allow_download=False)
    meta = json.loads((result.artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["optimizer_steps"] == 36 and manifest["epochs_run"] == 1
    assert meta["smoke"] is True and meta["normalization_stats"]["n_good_images"] == 4
    assert not (tmp_path / "output/artifacts").exists()
    for stage, source, split in (("teacher", manifest["teacher_statistics"], "train"),
                                 ("map_stats", meta["normalization_stats"], "calibration")):
        assert len(source["sample_ids"]) == len(set(source["sample_ids"])) == 4
        assert set(state["decoded"][stage]) == set(source["sample_ids"])
        for sid in source["sample_ids"]:
            assert by_id[sid]["split"] == split and not by_id[sid]["is_defect"] and not by_id[sid]["boxes"]
    assert set(state["decoded"]["optimizer"]) == set(manifest["teacher_statistics"]["sample_ids"])
    assert hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest() == meta["checkpoint_sha256"]
    assert hashlib.sha256(result.manifest_path.read_bytes()).hexdigest() == meta["run_manifest"]["sha256"]
    loaded = EfficientAdDetector.from_artifact(result.artifact_dir, allow_smoke=True)
    deltas = []
    for image, before in zip(state["probe_images"], state["pre_save"]):
        after = loaded.predict(image)
        np.testing.assert_allclose(after.anomaly_map, before.anomaly_map, atol=1e-5, rtol=0)
        assert abs(after.anomaly_score - before.anomaly_score) <= 1e-5
        deltas.append({"map": float(np.max(np.abs(after.anomaly_map - before.anomaly_map))),
                       "score": abs(after.anomaly_score - before.anomaly_score)})
    evidence = {key: state[key] for key in ("counts", "teacher", "normalization", "probe_ids", "decoded")}
    evidence.update(reload_deltas=deltas, checkpoint_sha256=meta["checkpoint_sha256"],
                    artifact_dir=str(result.artifact_dir), optimizer_steps=manifest["optimizer_steps"],
                    torch=str(torch.__version__))
    (tmp_path / "numerical-audit.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("VERIFIER_AUDIT " + json.dumps(evidence, sort_keys=True))
