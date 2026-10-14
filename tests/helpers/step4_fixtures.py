"""Fixtures load optional EfficientAD modules only while executing a test."""

import os
from pathlib import Path

import numpy as np
import pytest

from helpers.step4_spec import FakeEngine, build_dataset, detector_meta, tile_maps_from_global


@pytest.fixture(autouse=True)
def step4_no_network(request, monkeypatch):
    if not request.node.path.name.startswith("test_efficientad_"):
        return
    import socket
    def deny(*args, **kwargs):
        pytest.fail("Step 4 verification must not download pretrained assets")
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture
def step4_dataset(tmp_path):
    return build_dataset(tmp_path / "dataset")


@pytest.fixture
def step4_artifact_dir():
    value = os.environ.get("STEP4_ARTIFACT_DIR")
    if not value:
        pytest.skip("Set STEP4_ARTIFACT_DIR for real B03 artifact acceptance; no fake PASS")
    path = Path(value).resolve()
    assert path.is_dir(), f"STEP4_ARTIFACT_DIR does not exist: {path}"
    assert (path / "artifact.json").is_file(), f"Missing artifact.json: {path}"
    return path


@pytest.fixture
def step4_predict_tiles():
    def predict(tile_maps):
        from pcb_lab.models.efficientad.adapter import EfficientAdDetector
        engine = FakeEngine(tile_maps)
        detector = EfficientAdDetector(engine, detector_meta())
        result = detector.predict(np.zeros((640, 640, 3), dtype=np.uint8))
        assert len(engine.inputs) == 1, "predict must call engine.infer_tiles with all nine tiles"
        return result
    return predict


@pytest.fixture
def step4_predict_map(step4_predict_tiles):
    return lambda heatmap: step4_predict_tiles(tile_maps_from_global(heatmap))
