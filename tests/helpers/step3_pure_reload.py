"""Standalone artifact probe: intentionally never imports pcb_lab.

Inputs/outputs are JSON paths supplied by the parent test. Only authorized
calibration image paths appear in the input file.
"""
import json
from pathlib import Path
import socket
import sys
import zipfile


def main(request_path, output_path):
    def offline(*args, **kwargs):
        raise AssertionError("Unexpected network request during local checkpoint reload")
    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    from ultralytics import YOLO
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    result = {"checkpoints": {}, "predictions": {}}
    for kind in ("best", "last"):
        path = Path(request["artifact_dir"]) / f"{kind}.pt"
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert any(name.endswith("/data.pkl") for name in archive.namelist())
        model = YOLO(str(path))
        ckpt = model.ckpt
        names = [model.names[i] for i in range(len(model.names))] if isinstance(model.names, dict) else list(model.names)
        result["checkpoints"][kind] = {
            "class_order": names, "version": ckpt.get("version"), "epoch": ckpt.get("epoch"),
            "date": ckpt.get("date"), "train_args": ckpt.get("train_args"),
            "optimizer_present": ckpt.get("optimizer") is not None,
            "train_metrics": ckpt.get("train_metrics"), "model_type": type(model.model).__name__,
        }
        if kind == "best":
            params = request["infer_params"]
            for sample in request["samples"]:
                prediction = model.predict(source=sample["path"], device="cpu", imgsz=params["imgsz"],
                                           conf=params["conf_floor"], iou=params["iou"], max_det=params["max_det"],
                                           agnostic_nms=params["agnostic_nms"], half=False, verbose=False, save=False)[0]
                result["predictions"][sample["sample_id"]] = prediction.boxes.data.cpu().numpy().tolist()
    assert not any(key == "pcb_lab" or key.startswith("pcb_lab.") for key in sys.modules)
    result["pcb_lab_imported"] = False
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
