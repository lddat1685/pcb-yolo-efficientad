"""Dataset resolution must survive relocation and a different working directory."""
from pathlib import Path

from pcb_lab.data.manifest import resolve_dataset_root


def test_relative_yaml_uses_config_directory(tmp_path, monkeypatch):
    monkeypatch.delenv('DATASET_ROOT', raising=False)
    config = tmp_path / 'relocated repo' / 'configs' / 'dataset.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('DATASET_ROOT: ../data\n', encoding='utf-8')
    elsewhere = tmp_path / 'unrelated shell'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert resolve_dataset_root(config=config) == config.parent.parent / 'data'


def test_argument_and_environment_override_yaml(tmp_path, monkeypatch):
    env_root = tmp_path / 'environment dataset'
    explicit = tmp_path / 'explicit dataset'
    monkeypatch.setenv('DATASET_ROOT', str(env_root))
    missing_config = tmp_path / 'missing.yaml'
    assert resolve_dataset_root(config=missing_config) == env_root
    assert resolve_dataset_root(explicit, config=missing_config) == explicit


def test_auto_fallback_independent_of_cwd(tmp_path, monkeypatch):
    import pcb_lab.data.manifest as manifest
    relocated = tmp_path / 'relocated repo'
    config = relocated / 'configs' / 'dataset.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('DATASET_ROOT: auto\n', encoding='utf-8')
    monkeypatch.setattr(manifest, '__file__', str(relocated / 'src/pcb_lab/data/manifest.py'))
    monkeypatch.delenv('DATASET_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_dataset_root() == tmp_path / 'DatasetVer4_Public'
    (relocated / 'data').mkdir()
    assert resolve_dataset_root() == relocated / 'data'
