"""Implementer checks for routing, cache provenance, image I/O and UI state."""
from io import BytesIO
import json
import runpy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from pcb_lab.inference.registry import ModelRegistry, REPO_ROOT
from pcb_lab.inference.service import InferenceService, LoadedModel, load_model
from pcb_lab.inference.sources import UploadImageSource, URLImageSource, GalleryImageSource
from pcb_lab.inference.tiling import TileManager
from pcb_lab.models.yolo.adapter import Detection


def png(value=0, size=(640, 640)):
    buffer = BytesIO()
    Image.new("RGB", size, (value, value, value)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_service(tmp_path):
    frozen = json.loads((REPO_ROOT / "artifacts/test/frozen_evaluation.json").read_text())
    target = tmp_path / "artifacts/test/frozen_evaluation.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(frozen))
    for entry in frozen["checkpoints"].values():
        ckpt = tmp_path / entry["path"]
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"fixture")
        (ckpt.parent / "artifact.json").write_text("{}")
    state = {"score": .9, "ad": 2., "loads": [], "calls": []}

    class Detector:
        tile_manager = TileManager()

        def __init__(self, asset):
            self.asset = asset

        def describe(self):
            return {"infer_params": {"tile_size": 256, "stride": 224}}

        def predict(self, pixels):
            name = self.asset.model_id
            state["calls"].append(name)
            if state.get("fail") == name:
                raise RuntimeError("controlled model failure")
            if name.startswith("B03"):
                return SimpleNamespace(anomaly_score=state["ad"], anomaly_map=np.full(pixels.shape[:2], state["ad"], np.float32), boxes=[])
            return [Detection(0, "open_circuit", state["score"], (2., 3., 20., 30.))]

    def loader(asset, device):
        state["loads"].append(asset.model_id)
        return LoadedModel(Detector(asset))

    return InferenceService(ModelRegistry(tmp_path), model_loader=loader), state


@pytest.mark.parametrize("model,score,ad,decision,early,ad_called,reason", [
    ("B01", .14, 2., "DEFECT", False, False, "yolo_hit"),
    ("B02", .37, 2., "GOOD", False, False, "all_pass"),
    ("B03", .9, 1.4598, "DEFECT", False, True, "ad_hit"),
    ("H01", .14, 2., "DEFECT", True, False, "yolo_hit"),
    ("H01", .13, 1.4598, "DEFECT", False, True, "ad_hit"),
    ("H01", .13, 1.4, "GOOD", False, True, "all_pass"),
    ("H00", .9, 2., "DEFECT", False, True, "both_hit"),
    ("H00", .1, 1., "GOOD", False, True, "all_pass"),
    ("H02", .38, 2., "DEFECT", True, False, "yolo_hit"),
    ("H01_sec", .47, 1., "GOOD", False, True, "all_pass"),
])
def test_actual_routing_and_boundaries(fake_service, model, score, ad, decision, early, ad_called, reason):
    service, state = fake_service
    state.update(score=score, ad=ad)
    result = service.inspect(UploadImageSource(png(), "misleading_defect.png"), model)
    assert result.error_reason is None
    assert (result.decision, result.early_exit, result.efficientad_executed, result.reason) == (decision, early, ad_called, reason)
    assert ("B03_efficientad_s" in state["calls"]) == ad_called
    assert (result.anomaly_map is not None) == ad_called
    assert result.metadata["ground_truth_available"] is False
    assert "ground_truth" not in result.metadata
    json.dumps(result.to_dict(), allow_nan=False)


def test_cache_shares_models_and_invalidates_changed_artifact(fake_service):
    service, state = fake_service
    for model in ["B01", "H00", "H01", "H02", "B02", "B03"]:
        assert service.inspect(UploadImageSource(png()), model).decision != "ERROR"
    assert sorted(state["loads"]) == ["B01_yolo11n", "B02_yolo11s", "B03_efficientad_s"]
    path = service.registry.get("B01").yolo.checkpoint.parent / "artifact.json"
    path.write_text('{"changed":true}')
    service.inspect(UploadImageSource(png()), "B01")
    assert state["loads"].count("B01_yolo11n") == 2


def test_streamlit_resource_cache_shares_across_services(fake_service, monkeypatch):
    from pcb_lab.ui import resources
    service, state = fake_service
    resources.cached_model.clear()
    monkeypatch.setattr(resources, "load_model", service.model_loader)
    services = [InferenceService(service.registry, model_loader=resources.model_loader) for _ in range(2)]
    for instance in services:
        instance.inspect(UploadImageSource(png()), "H00")
    assert state["loads"] == ["B01_yolo11n", "B03_efficientad_s"]
    resources.cached_model.clear()


def test_errors_do_not_become_good_or_abort_batch(fake_service):
    service, state = fake_service
    results = service.inspect_batch([UploadImageSource(b"broken"), UploadImageSource(png())])
    assert [r.decision for r in results] == ["ERROR", "DEFECT"]
    state.update(score=.1, fail="B03_efficientad_s")
    assert service.inspect(UploadImageSource(png())).decision == "ERROR"
    state.update(score=.9)
    assert service.inspect(UploadImageSource(png())).decision == "DEFECT"
    assert service.inspect(UploadImageSource(png()), "H00").decision == "ERROR"
    state.update(fail=None, ad=float("nan"))
    assert service.inspect(UploadImageSource(png()), "B03").decision == "ERROR"
    assert service.inspect(UploadImageSource(png()), "unknown").decision == "ERROR"
    with pytest.raises(ValueError):
        service.inspect_batch([UploadImageSource(png())] * 21)


def test_checkpoint_rejected_before_loading(fake_service):
    service, _ = fake_service
    with pytest.raises(ValueError, match="Checkpoint"):
        load_model(service.registry.get("B01").yolo)


def test_cli_uses_service_and_preserves_error_exit(fake_service, tmp_path, monkeypatch, capsys):
    service, _ = fake_service
    cli = runpy.run_path(str(REPO_ROOT / "scripts/inspect_image.py"))
    monkeypatch.setitem(cli["main"].__globals__, "InferenceService", lambda registry: service)
    path = tmp_path / "input.png"
    path.write_bytes(png())
    args = ["--image", str(path), "--repo-root", str(service.registry.repo_root), "--model", "H00"]
    expected = service.inspect(UploadImageSource(path.read_bytes()), "H00").to_dict()
    assert cli["main"](args) == 0
    actual = json.loads(capsys.readouterr().out)
    for field in ("decision", "reason", "yolo_image_score", "anomaly_score", "detections", "heatmap"):
        assert actual[field] == json.loads(json.dumps(expected[field]))
    path.write_bytes(b"invalid")
    assert cli["main"](args) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "ERROR"


def test_sources_and_geometry(fake_service, monkeypatch):
    service, state = fake_service
    result = service.inspect(UploadImageSource(png(size=(300, 160))), "H00")
    assert result.image.shape == (160, 300, 3)
    assert result.anomaly_map.shape == (160, 300)
    assert result.detections[0]["xyxy_original"] == (2., 3., 20., 30.)
    from pcb_lab.ui.visuals import overlay
    before = result.image.copy()
    assert overlay(result).size == (300, 160)
    np.testing.assert_array_equal(before, result.image)
    with pytest.raises(ValueError):
        UploadImageSource(png(size=(4097, 1))).get_frame()
    with pytest.raises(ValueError):
        URLImageSource("file:///test.png").get_frame()
    class Response(BytesIO):
        def geturl(self):
            return "https://example.test/image"
    monkeypatch.setattr("pcb_lab.inference.sources.urlopen", lambda *a, **k: Response(png(size=(30, 20))))
    assert URLImageSource("https://example.test/image").get_frame()[1].shape == (20, 30, 3)


def test_test_gallery_requires_frozen_reports(synthetic_dataset_root):
    registry = ModelRegistry()
    with pytest.raises(PermissionError):
        registry.gallery("test", dataset_root=synthetic_dataset_root)
    with pytest.raises(ValueError, match="khớp"):
        registry.gallery("test", dataset_root=synthetic_dataset_root, report_mode=True)


def test_benchmark_arithmetic_and_pareto():
    from pcb_lab.inference.reports import load_benchmark, pareto_models
    report = load_benchmark()
    values = report.h00_h01()
    assert values["latency_reduction_pct"] == pytest.approx(52.730158, abs=.00001)
    assert values["recall_delta"] == values["frr_delta"] == 0
    assert "H01" not in pareto_models(report.summary)  # B01 has equal quality, lower latency.
    assert "B01" in pareto_models(report.summary)
    assert len({r["class_name"] for r in report.per_class}) == 6
    assert len({r["source_group"] for r in report.per_group}) == 2


def test_apptest_screens_and_rerun(fake_service, monkeypatch, synthetic_dataset_root):
    from streamlit.testing.v1 import AppTest
    from pcb_lab.ui import resources
    from pcb_lab.data.samples import ManifestDataset
    service, state = fake_service
    samples = tuple(ManifestDataset(synthetic_dataset_root, "calibration"))
    monkeypatch.setattr(resources, "get_service", lambda: service)
    monkeypatch.setattr(resources, "get_gallery", lambda *a, **k: samples)
    app = AppTest.from_file(str(REPO_ROOT / "src/pcb_lab/ui/app.py"), default_timeout=30).run()
    assert not app.exception
    app.radio(key="single_source").set_value("Gallery").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.single_result[1].decision == "DEFECT"
    calls = len(state["calls"])
    app.slider(key="single_opacity").set_value(.6).run()
    assert len(state["calls"]) == calls
    app.sidebar.selectbox[0].set_value("H00").run()
    assert not app.metric  # Never show the previous model's result.
    state["fail"] = "B03_efficientad_s"
    app.button[0].click().run()
    assert not app.exception
    assert "ERROR" in app.error[0].value
    state["fail"] = None
    app.sidebar.radio[0].set_value("Benchmark H00 vs H01").run()
    assert not app.exception
    assert app.metric[0].value == "52.7%"
    app.sidebar.radio[0].set_value("Batch Inspection").run()
    app.radio[0].set_value("Gallery").run()
    app.multiselect(key="batch_samples").set_value([samples[0].sample_id]).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.session_state.batch_results[1]) == 1
