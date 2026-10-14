"""Step 8 Spec-first Test Suite: Verification of UI & InferenceService Invariants."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from pcb_lab.inference.tiling import TileManager

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registered_models_in_service():
    """Verify that all mandatory models from shortlist and Step 7 are supported."""
    from pcb_lab.inference.registry import ModelRegistry
    reg = ModelRegistry(REPO_ROOT)
    models = set(reg.specs.keys())
    expected = {"B01", "B02", "B03", "H00", "H01", "H02"}
    assert expected.issubset(models), f"Expected models {expected} to be in {models}"


def test_base_image_source_contract(tmp_path):
    """Verify Image sources convert various inputs into standardized RGB uint8 (H, W, 3)."""
    from pcb_lab.inference.sources import UploadImageSource, FileImageSource

    # 1. UploadImageSource (PNG bytes)
    buf = io.BytesIO()
    Image.new("RGB", (640, 640), color="blue").save(buf, format="PNG")
    src_up = UploadImageSource(buf.getvalue(), name="test_upload.png")
    sid, frame = src_up.get_frame()
    assert sid.startswith("upload:")
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (640, 640, 3)
    assert frame.dtype == np.uint8

    # 2. FileImageSource (external file outside dataset root)
    img_path = tmp_path / "test_file.png"
    Image.new("RGB", (320, 320), color="red").save(img_path)
    src_file = FileImageSource(img_path)
    sid_f, frame_f = src_file.get_frame()
    assert sid_f == "test_file.png"
    assert frame_f.shape == (320, 320, 3)
    assert frame_f.dtype == np.uint8


def test_no_fake_accuracy_on_unlabeled_input():
    """Verify inference result on arbitrary image has no synthetic accuracy."""
    from pcb_lab.inference.sources import UploadImageSource
    buf = io.BytesIO()
    Image.new("RGB", (640, 640), color="white").save(buf, format="PNG")
    src = UploadImageSource(buf.getvalue(), name="user_upload.png")

    meta = src.describe()
    assert meta.get("ground_truth_available") is False
    assert "ground_truth" not in meta


def test_early_exit_trace_invariant_on_h01():
    """Verify that H01 records early-exit when YOLO is positive and skips AD."""
    from pcb_lab.inference.registry import ModelRegistry
    from pcb_lab.inference.service import InferenceService, LoadedModel
    from pcb_lab.inference.sources import UploadImageSource
    from pcb_lab.models.yolo.adapter import Detection

    calls = []

    class MockYolo:
        tile_manager = TileManager()

        def describe(self):
            return {"infer_params": {"tile_size": 256, "stride": 224}}

        def predict(self, pixels):
            calls.append("yolo")
            return [Detection(0, "short", 0.95, (10.0, 10.0, 50.0, 50.0))]

    class MockAD:
        tile_manager = TileManager()

        def describe(self):
            return {"infer_params": {}}

        def predict(self, pixels):
            calls.append("ad")
            return SimpleNamespace(anomaly_score=0.5, anomaly_map=np.zeros(pixels.shape[:2], np.float32), boxes=[])

    def mock_loader(asset, device):
        if "yolo" in asset.model_id.lower() or asset.model_id.startswith("B01") or asset.model_id.startswith("B02"):
            return LoadedModel(MockYolo())
        return LoadedModel(MockAD())

    reg = ModelRegistry(REPO_ROOT)
    svc = InferenceService(reg, model_loader=mock_loader)

    buf = io.BytesIO()
    Image.new("RGB", (640, 640), color="green").save(buf, format="PNG")
    src = UploadImageSource(buf.getvalue())
    res = svc.inspect(src, model_id="H01")

    assert res.decision == "DEFECT", f"Unexpected decision {res.decision}: {res.error_reason}"
    assert res.early_exit is True
    assert res.efficientad_executed is False
    assert res.reason == "yolo_hit"
    assert res.anomaly_score is None
    assert "ad" not in calls


def test_unconditional_or_trace_invariant_on_h00():
    """Verify that H00 evaluates both models even when YOLO is positive."""
    from pcb_lab.inference.registry import ModelRegistry
    from pcb_lab.inference.service import InferenceService, LoadedModel
    from pcb_lab.inference.sources import UploadImageSource
    from pcb_lab.models.yolo.adapter import Detection

    calls = []

    class MockYolo:
        tile_manager = TileManager()

        def describe(self):
            return {"infer_params": {"tile_size": 256, "stride": 224}}

        def predict(self, pixels):
            calls.append("yolo")
            return [Detection(0, "short", 0.95, (10.0, 10.0, 50.0, 50.0))]

    class MockAD:
        tile_manager = TileManager()

        def describe(self):
            return {"infer_params": {}}

        def predict(self, pixels):
            calls.append("ad")
            return SimpleNamespace(anomaly_score=1.5, anomaly_map=np.zeros(pixels.shape[:2], np.float32), boxes=[])

    def mock_loader(asset, device):
        if "yolo" in asset.model_id.lower() or asset.model_id.startswith("B01") or asset.model_id.startswith("B02"):
            return LoadedModel(MockYolo())
        return LoadedModel(MockAD())

    reg = ModelRegistry(REPO_ROOT)
    svc = InferenceService(reg, model_loader=mock_loader)

    buf = io.BytesIO()
    Image.new("RGB", (640, 640), color="green").save(buf, format="PNG")
    src = UploadImageSource(buf.getvalue())
    res = svc.inspect(src, model_id="H00")

    assert res.decision == "DEFECT", f"Unexpected decision {res.decision}: {res.error_reason}"
    assert res.early_exit is False
    assert res.efficientad_executed is True
    assert res.anomaly_score == 1.5
    assert "ad" in calls
