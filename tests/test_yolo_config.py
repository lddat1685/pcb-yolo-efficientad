"""Reject unapproved/unknown config before touching training or weights."""

import pytest

from conftest import REPO_ROOT
from helpers.step3_spec import (approved_test_config, assert_error_name, assert_sha,
                               reverse_keys, write_config)


def _forbid_model_creation(monkeypatch):
    # B evidence: ignored nested typos reached real train. Reject before model creation.
    import importlib
    module = importlib.import_module("pcb_lab.models.yolo.train")
    def forbidden(*args, **kwargs):
        pytest.fail("Trainer reached model creation instead of raising ConfigError")
    monkeypatch.setattr(module, "YOLO", forbidden)


@pytest.mark.parametrize("key", ["lr0", "batch", "workers", "augment"])
@pytest.mark.parametrize("smoke", [False, True])
def test_trainer_rejects_each_null(synthetic_dataset_root, tmp_path, monkeypatch, key, smoke):
    # Protect contract config/train.py invariant 3: explicit null never falls back to YOLO defaults.
    from pcb_lab.models.yolo.train import train_yolo
    _forbid_model_creation(monkeypatch)
    config = approved_test_config()
    config["train"][key] = None
    path = write_config(tmp_path / "config", config)
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    assert_error_name("ConfigError", lambda: train_yolo(
        path, seed=42, out_root=tmp_path / "output", device="cpu", smoke=smoke, allow_download=False))


@pytest.mark.parametrize("section", [None, "train", "selection", "infer"])
def test_trainer_rejects_unknown_keys(synthetic_dataset_root, tmp_path, monkeypatch, section):
    # Protect contract config: typo/extra keys fail, including nested train/infer selection.
    from pcb_lab.models.yolo.train import train_yolo
    _forbid_model_creation(monkeypatch)
    config = approved_test_config()
    target = config if section is None else config[section]
    target["unapproved_typo_123"] = 1
    path = write_config(tmp_path / "config", config)
    monkeypatch.setenv("DATASET_ROOT", str(synthetic_dataset_root))
    assert_error_name("ConfigError", lambda: train_yolo(
        path, out_root=tmp_path / "output", device="cpu", smoke=True, allow_download=False))


def test_b01_b02_differ_only_in_model_identity():
    # Protect plan §5 Bước 3/§6 and contract config: fair, shared recipe for B01 versus B02.
    from pcb_lab.models.yolo.train import _merge_configs
    directory = REPO_ROOT / "configs/models"
    configs = [_merge_configs([directory / "yolo_common.yaml", directory / filename])
               for filename in ("yolo11n.yaml", "yolo11s.yaml")]
    expected = [("B01_yolo11n", "yolo11n"), ("B02_yolo11s", "yolo11s")]
    for config, (model_id, architecture) in zip(configs, expected):
        assert config["model_id"] == model_id and config["architecture"] == architecture
        assert config["pretrained"].replace("\\", "/") == f"artifacts/pretrained/{architecture}.pt"
    differing = {key for key in configs[0].keys() | configs[1].keys()
                 if configs[0].get(key) != configs[1].get(key)}
    assert differing == {"model_id", "architecture", "pretrained"}
    # A justified OOM batch exception needs coordinator review, not an automatic weakening here.


@pytest.mark.parametrize("key", ["lr0", "batch", "workers", "augment"])
def test_check_nulls_rejects_each_unapproved_value(key):
    # Protect contract config + supervisor API clarification: each null independently fails.
    from pcb_lab.models.yolo.train import _check_nulls, ConfigError
    config = approved_test_config()
    _check_nulls(config)  # positive control prevents an always-raise implementation passing.
    config["train"][key] = None
    with pytest.raises(ConfigError):
        _check_nulls(config)


def test_merge_rejects_unknown_top_level_key(tmp_path):
    # Protect contract unknown-key rule at the supervisor-approved loader boundary.
    from pcb_lab.models.yolo.train import _merge_configs, ConfigError
    config = approved_test_config()
    path = write_config(tmp_path / "config", config)
    assert _merge_configs([path.parent / "yolo_common.yaml", path]) == config
    config["unapproved_key"] = True
    write_config(path.parent, config)
    with pytest.raises(ConfigError):
        _merge_configs([path.parent / "yolo_common.yaml", path])


def test_config_hash_stable_key_order_and_seed_independent(tmp_path):
    # Protect contract config_hash + supervisor clarification: canonical JSON, excluding seed.
    from pcb_lab.models.yolo.train import _config_hash, _merge_configs
    config = approved_test_config()
    original = write_config(tmp_path / "one", config)
    reordered = write_config(tmp_path / "two", reverse_keys(config))
    merged = [_merge_configs([path.parent / "yolo_common.yaml", path]) for path in (original, reordered)]
    expected = _config_hash(merged[0])
    assert_sha(expected)
    assert _config_hash(merged[0]) == expected == _config_hash(merged[1])
    # seed is excluded from hash, but remains forbidden as a YAML config key (CLI-only).
    assert _config_hash({**merged[0], "seed": 42}) == expected
    assert _config_hash({**merged[0], "seed": 43}) == expected
    changed = approved_test_config()
    changed["infer"]["iou"] = 0.6
    assert _config_hash(changed) != expected, "Hash must reflect configuration changes"
