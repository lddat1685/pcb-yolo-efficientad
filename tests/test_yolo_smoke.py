"""Opt-in cost category: real-data tiny-view, one-epoch CPU train and checkpoint reload."""

import pytest

from conftest import REPO_ROOT
from helpers.step3_spec import (CLASSES, approved_test_config, assert_run_schema, dataset_snapshot,
                               detection_record, read_json, sha256, write_config, write_json)

pytestmark = [pytest.mark.smoke, pytest.mark.dataset]


def test_one_epoch_cpu_train_reload_and_no_official_artifact(
    step3_cpu_dependencies, real_dataset_root, tmp_path, monkeypatch
):
    # Protect plan §5 Bước 3/§11 reload and contract train smoke: CPU, one epoch, tiny view, no artifacts/.
    import numpy as np
    import shutil
    from pcb_lab.data.samples import ManifestDataset
    from pcb_lab.inference.preprocessing import prepare_input
    from pcb_lab.models.yolo.adapter import YoloDetector
    from pcb_lab.models.yolo.train import train_yolo
    torch, ultralytics = step3_cpu_dependencies
    monkeypatch.setenv("DATASET_ROOT", str(real_dataset_root))
    config = approved_test_config()
    # Missing pretrained proves smoke initializes architecture, never fetches pretrained weights.
    config["pretrained"] = str(tmp_path / "intentionally-absent-weights.pt")
    config_path = write_config(tmp_path / "config", config)
    out = tmp_path / "output"
    def official_snapshot():
        root = REPO_ROOT / "artifacts"
        return {path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, sha256(path))
                for path in root.rglob("*") if path.is_file()} if root.exists() else None
    official_before = official_snapshot()
    before = dataset_snapshot(real_dataset_root)
    try:
        train_yolo(config_path, seed=42, out_root=out, device="cpu", smoke=True, allow_download=False)
        assert not (out / "artifacts").exists(), "Smoke must not create even an empty official artifact directory"
        manifests = list((out / "runs/smoke").rglob("run_manifest.json"))
        assert len(manifests) == 1
        run = read_json(manifests[0])
        assert run["smoke"] is True and run["epochs_run"] == 1
        assert str(run["device"]).lower() == "cpu"
        assert 0 < run["partitions_used"]["train"] < 1792
        assert run["args_used"]["epochs"] == 1 and run["args_used"]["workers"] == 0
        # Ultralytics setup_model initializes YAML without weights even when default pretrained=True.
        assert str(run["args_used"]["model"]).endswith(".yaml")
        assert run["pretrained"]["sha256"] is None
        for kind in ("best", "last"):
            matches = list(manifests[0].parent.rglob(f"{kind}.pt"))
            assert matches and all(sha256(path) == run["checkpoints"][f"{kind}_sha256"] for path in matches)
        best = next(manifests[0].parent.rglob("best.pt"))
        import torch
        resumable = torch.load(manifests[0].parent / "weights/last.pt", map_location="cpu", weights_only=False)
        assert resumable["epoch"] >= 0
        assert isinstance(resumable["optimizer"], dict) and resumable["optimizer"]["state"]
        assert resumable["ema"] is not None
        model = ultralytics.YOLO(str(best))
        names = model.names
        assert ([names[index] for index in range(6)] if isinstance(names, dict) else list(names)) == CLASSES
        sample = next(iter(ManifestDataset(real_dataset_root, "calibration")))
        infer = config["infer"]
        kwargs = {"device": "cpu", "imgsz": infer["imgsz"], "conf": infer["conf_floor"],
                  "iou": infer["iou"], "max_det": infer["max_det"], "agnostic_nms": False,
                  "half": False, "verbose": False, "save": False}
        from_path = model.predict(source=str(sample.image_path), **kwargs)[0].boxes.data.cpu().numpy()
        # Construct a verifier-only reload bundle INSIDE runs/smoke, using actual checkpoint bytes.
        bundle = out / "runs/smoke/verifier-reload"
        bundle.mkdir()
        shutil.copy2(best, bundle / "best.pt")
        shutil.copy2(next(manifests[0].parent.rglob("last.pt")), bundle / "last.pt")
        prep = prepare_input(sample.load_image(), "yolo").meta
        write_json(bundle / "artifact.json", {
            "schema_version": 1, "model_id": run["model_id"], "seed": 42, "smoke": True,
            "checkpoint_best_sha256": sha256(bundle / "best.pt"),
            "checkpoint_last_sha256": sha256(bundle / "last.pt"), "class_order": CLASSES,
            "config_hash": run["config_hash"], "preprocessing_version": prep["preprocessing_version"],
            "preprocessing_hash": prep["preprocessing_hash"], "inference_params": infer,
            "view_signature": run["view_signature"], "run_manifest": {
                "path": str(manifests[0].resolve()), "sha256": sha256(manifests[0])},
            "created_by_git_commit": run["git_commit"],
        })
        detector = YoloDetector.from_artifact(bundle, allow_smoke=True)
        rgb_dets = [detection_record(det) for det in detector.predict(sample.load_image())]
        # Compare adapter RGB input against pure Ultralytics path input, with the same infer recipe.
        adapter_rows = [[*det["xyxy_original"], det["confidence"], det["class_id"]] for det in rgb_dets]
        sort_key = lambda row: (-float(row[4]), int(row[5]), *map(float, row[:4]))
        expected = np.asarray(sorted(from_path.tolist(), key=sort_key)).reshape(-1, 6)
        actual = np.asarray(sorted(adapter_rows, key=sort_key)).reshape(-1, 6)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual[:, :4], expected[:, :4], atol=0.5, rtol=0)
        np.testing.assert_allclose(actual[:, 4], expected[:, 4], atol=1e-3, rtol=0)
        np.testing.assert_array_equal(actual[:, 5], expected[:, 5])
        assert not (out / "artifacts").exists()
        # Keep reload evidence available even when provenance/schema assertions fail.
        assert_run_schema(run)
    finally:
        # Includes source annotation caches; source test images remain stat-only throughout.
        assert dataset_snapshot(real_dataset_root) == before
        assert official_snapshot() == official_before
