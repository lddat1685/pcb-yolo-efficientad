"""Step 4 config, normal-only/test gates, and artifact schema (spec-first)."""

from pathlib import Path

import pytest

from helpers.step4_spec import (assert_artifact, assert_calibration, assert_config,
                               assert_named_error, assert_run, read_json, resolve_manifest,
                               sha256, write_config)

REPO = Path(__file__).resolve().parents[1]


def test_checked_in_config_matches_locked_recipe():
    import yaml
    path = REPO / "configs/models/efficientad_s.yaml"
    assert path.is_file(), "Step 4 must provide configs/models/efficientad_s.yaml"
    assert_config(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("section", [None, "train", "normalization", "infer"])
def test_unknown_config_key_is_config_error(tmp_path, monkeypatch, section):
    from pcb_lab.models.efficientad.train import train_efficientad
    def mutate(config):
        (config if section is None else config[section])["verifier_unknown_key"] = True
    config = write_config(tmp_path / "unknown.yaml", mutate)
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "absent-dataset"))
    with pytest.raises(Exception) as caught:
        train_efficientad(config, out_root=tmp_path / "out", device="cpu", allow_download=False)
    assert_named_error(caught.value, "ConfigError")
    assert not list((tmp_path / "out").rglob("*.pt"))


@pytest.mark.parametrize("key", ["lr", "weight_decay", "device"])
def test_unapproved_null_prevents_training(tmp_path, monkeypatch, key):
    from pcb_lab.models.efficientad.train import train_efficientad
    config = write_config(tmp_path / "null.yaml", lambda c: c["train"].update({key: None}))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "absent-dataset"))
    with pytest.raises(Exception) as caught:
        # No CLI device override: config device=null remains unapproved.
        train_efficientad(config, out_root=tmp_path / "out", allow_download=False)
    assert_named_error(caught.value, "ConfigError")
    assert not list((tmp_path / "out").rglob("*.pt"))


@pytest.mark.parametrize("section,key,value", [
    ("train", "tile_size", 320), ("train", "stride", 256), ("train", "batch_size", 2),
    ("normalization", "teacher_stats_split", "calibration"),
    ("normalization", "teacher_stats_split", "test"),
    ("normalization", "map_norm_split", "fusion"),
    ("normalization", "map_norm_split", "test"),
    ("infer", "tile_size", 320), ("infer", "stitch_method", "average"),
    ("infer", "score_metric", "mean"), ("infer", "morph_k", 1),
    ("infer", "nms_iou", 0.5), ("infer", "box_class_name", "pin_hole"),
])
def test_locked_recipe_cannot_be_silently_changed(tmp_path, monkeypatch, section, key, value):
    from pcb_lab.models.efficientad.train import train_efficientad
    config = write_config(tmp_path / "invalid.yaml", lambda c: c[section].update({key: value}))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "absent-dataset"))
    with pytest.raises(Exception) as caught:
        train_efficientad(config, out_root=tmp_path / "out", device="cpu", allow_download=False)
    assert_named_error(caught.value, "ConfigError")


@pytest.mark.parametrize("limit", [None, 1])
def test_test_dataset_is_denied_even_before_files_exist(tmp_path, limit):
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    with pytest.raises(PermissionError):
        EfficientAdDataset(tmp_path / "does-not-exist", split="test", limit=limit)


def test_normal_only_guard_rejects_poisoned_ad_train_set(step4_dataset, monkeypatch):
    from pcb_lab.data.samples import AdTrainSet, ManifestDataset, Sample
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    candidates = list(ManifestDataset(step4_dataset.root, "train"))
    assert any(s.is_defect for s in candidates)
    # Poison the Step 2 normal-only provider, NOT the canonical train manifest:
    # the latter legitimately contains 897 defects and must be filtered normally.
    monkeypatch.setattr(AdTrainSet, "__iter__", lambda self: iter(candidates))
    monkeypatch.setattr(AdTrainSet, "__getitem__", lambda self, i: candidates[i])
    monkeypatch.setattr(AdTrainSet, "__len__", lambda self: len(candidates))
    def forbidden_decode(*args, **kwargs):
        pytest.fail("normal-only audit must reject contamination before image decoding")
    monkeypatch.setattr(Sample, "load_image", forbidden_decode)
    with pytest.raises(Exception) as caught:
        EfficientAdDataset(step4_dataset.root, split="train")
    assert isinstance(caught.value, ValueError) or any(
        c.__name__ == "DataLeakError" for c in type(caught.value).__mro__)


def test_normal_train_filters_canonical_defects_before_loading(step4_dataset, monkeypatch):
    from pcb_lab.data.samples import Sample
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    original = Sample.load_image
    loaded = []
    def checked(sample):
        assert sample.partition == "train" and sample.is_defect is False
        loaded.append(sample.sample_id)
        return original(sample)
    monkeypatch.setattr(Sample, "load_image", checked)
    dataset = EfficientAdDataset(step4_dataset.root, split="train")
    for _ in dataset:
        pass
    wanted = {r["sample_id"] for r in step4_dataset.records if r["split"] == "train" and not r["is_defect"]}
    assert set(loaded) == wanted


@pytest.mark.parametrize("split", ["calibration", "fusion"])
def test_eval_includes_good_and_defect_without_test(step4_dataset, monkeypatch, split):
    from pcb_lab.data.samples import Sample
    from pcb_lab.models.efficientad.dataset import EfficientAdDataset
    original = Sample.load_image
    loaded = []
    def checked(sample):
        assert sample.partition == split
        loaded.append(sample.sample_id)
        return original(sample)
    monkeypatch.setattr(Sample, "load_image", checked)
    for _ in EfficientAdDataset(step4_dataset.root, split=split):
        pass
    wanted = {r["sample_id"] for r in step4_dataset.records if r["split"] == split}
    assert set(loaded) == wanted


def test_real_artifact_layout_schema_and_hashes(step4_artifact_dir):
    root = step4_artifact_dir
    meta = read_json(root / "artifact.json")
    assert_artifact(meta)
    assert meta["smoke"] is False
    assert root.parts[-4:] == ("artifacts", "efficientad", meta["model_id"], f"seed{meta['seed']}")
    for name in ("model.pt", "artifact.json", "model_card.md", "calibration_stats.json",
                 "preds_calibration.jsonl", "preds_fusion.jsonl"):
        assert (root / name).is_file(), name
        assert (root / name).stat().st_size > 0
    manifest = resolve_manifest(root, meta)
    assert sha256(manifest) == meta["run_manifest"]["sha256"]
    run = read_json(manifest)
    assert_run(run)
    assert run["smoke"] is False
    assert manifest.parent.parts[-4:] == ("runs", "efficientad", meta["model_id"], f"seed{meta['seed']}")
    for name in ("env.json", "train.log"):
        assert (manifest.parent / name).is_file(), name
    assert sha256(root / "model.pt") == meta["checkpoint_sha256"] == run["checkpoint_sha256"]
    for key in ("model_id", "seed", "config_hash"):
        assert meta[key] == run[key]
    assert meta["created_by_git_commit"] == run["git_commit"]
    project = root.parents[3]
    assert (project / "reports" / f"efficientad_{meta['model_id']}_seed{meta['seed']}_validation.md").is_file()


def test_real_calibration_statistics_schema(step4_artifact_dir):
    assert_calibration(read_json(step4_artifact_dir / "calibration_stats.json"))
