"""Regression checks for Phase C metadata corrections; no training required."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

@pytest.fixture
def runner():
    # Collection must work without loading optional torch DLLs.
    from pcb_lab.models.yolo import train
    return train


def test_f1_confidence_is_curve_coordinate_and_rejects_nan(runner):
    box = SimpleNamespace(f1=np.array([0.91, 0.82]), px=np.linspace(0, 1, 101),
                          f1_curve=np.zeros((2, 101)))
    box.f1_curve[:, 55:66] = 1
    assert runner._f1_optimal_conf(box) == pytest.approx(0.6)
    box.f1_curve[0, 0] = np.nan
    assert runner._f1_optimal_conf(box) is None
    assert runner._f1_optimal_conf(SimpleNamespace()) is None


def test_validation_uses_square_inputs_and_actual_class_counts(tmp_path, monkeypatch, runner):
    import ultralytics
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    box = SimpleNamespace(map50=0.8, map=0.6, mp=0.7, mr=0.9,
                          ap_class_index=np.array([1, 5]), ap50=[0.8, 0.8], ap=[0.6, 0.6],
                          p=[0.7, 0.7], r=[0.9, 0.9],
                          f1_curve=np.array([[0, 1, 0]]), px=np.array([0, 0.5, 1]))
    called = {}
    def validate(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(box=box, nt_per_class=np.array([0, 160, 0, 0, 0, 265]))
    monkeypatch.setattr(ultralytics, "YOLO", lambda _: SimpleNamespace(
        names={0: "open_circuit", 1: "short", 2: "mouse_bite", 3: "spur", 4: "spurious_copper", 5: "pin_hole"},
        val=validate))
    view = SimpleNamespace(data_yaml=tmp_path / "data.yaml", box_counts={"calibration": {"short": 160}})
    params = dict(imgsz=640, conf_floor=0.001, iou=0.7, max_det=300, agnostic_nms=False, half=False)
    overall, classes = runner._validate("B01", checkpoint, view, params, "cpu")
    assert called["rect"] is False and called["imgsz"] == 640 and called["split"] == "val"
    assert called["half"] is False and called["workers"] == 0
    assert classes["short"]["n_gt_boxes"] == 160
    assert classes["pin_hole"]["n_gt_boxes"] == 265
    assert overall["pr_conf"] == 0.5


def test_training_summary_uses_csv_and_last_equal_best(tmp_path, runner):
    (tmp_path / "results.csv").write_text(
        " epoch, time, metrics/mAP50-95(B)\n1, 10, 0.5\n2, 20, 0.8\n3, 30.26, 0.8\n", encoding="utf-8")
    assert runner._read_training_summary(tmp_path, 100) == dict(
        epochs_run=3, best_epoch=3, train_time_s=30.3, early_stopped=True)
    assert runner._read_training_summary(tmp_path, 3)["early_stopped"] is False


def test_missing_training_evidence_does_not_invent_epochs(tmp_path, runner):
    with pytest.raises(runner.ConfigError, match="Cannot derive training summary"):
        runner._read_training_summary(tmp_path, 100)


def test_git_state_uses_bundle_provenance_without_git(tmp_path, monkeypatch, runner):
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "src/pcb_lab/models/yolo/train.py"))
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(runner.subprocess, "check_output", no_git)
    (tmp_path / "bundle_provenance.json").write_text(json.dumps({
        "git_commit": "a" * 40, "git_dirty": False}), encoding="utf-8")
    assert runner._git_state() == ("a" * 40, False)


def test_model_card_records_actual_training_versions(tmp_path, runner):
    card = tmp_path / "model_card.md"
    config = {"train": dict(optimizer="AdamW", lr0=0.001, batch=16, imgsz=640, epochs=100, patience=20),
              "infer": dict(conf_floor=0.001, iou=0.7, max_det=300)}
    view = SimpleNamespace(counts={"train": dict(images=1792, good=895, defect=897),
                                   "calibration": dict(images=460)}, view_signature="v")
    runner._write_model_card(card, "B01", "yolo11n", config, view, {}, {},
                             ultralytics_version="8.4.161", torch_version="2.14.0+cu130", config_hash="a" * 64)
    text = card.read_text(encoding="utf-8")
    assert "n/a" not in text
    assert "torch 2.14.0+cu130" in text and "config_hash=" + "a" * 64 in text
