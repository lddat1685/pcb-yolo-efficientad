"""Bounded camera snapshots with explicit sensor-to-inspection geometry."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image, ImageOps

from pcb_lab.cameras.imv import CameraFrame, MAX_SENSOR_PIXELS, MAX_SENSOR_SIDE
from .sources import BaseImageSource, MAX_PIXELS, validate_pixels


def saved_camera_frame(data: bytes, name: str) -> CameraFrame:
    if not data or len(data) > 64 * 1024 * 1024:
        raise ValueError("Ảnh camera phải dưới 64 MiB")
    with Image.open(BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG", "BMP", "TIFF"}:
            raise ValueError("Chỉ nhận PNG/JPEG/BMP/TIFF từ MV Viewer")
        if image.width * image.height > MAX_SENSOR_PIXELS or max(image.size) > MAX_SENSOR_SIDE:
            raise ValueError("Ảnh camera vượt giới hạn 16 triệu pixel / cạnh 8192")
        if image.mode not in {"L", "RGB"}:
            raise ValueError("Hãy lưu ảnh Mono8 (8-bit) hoặc RGB8 từ MV Viewer")
        pixels = np.array(ImageOps.exif_transpose(image), dtype=np.uint8)
    return CameraFrame(pixels, {"origin": "mv_viewer_file", "name": name,
                               "file_sha256": hashlib.sha256(data).hexdigest()})


class IndustrialCameraSource(BaseImageSource):
    def __init__(self, frame: CameraFrame, *, roi=None, max_side=1280):
        raw = frame.pixels
        if (raw.dtype != np.uint8 or raw.ndim not in (2, 3) or
                (raw.ndim == 3 and raw.shape[2] != 3)):
            raise ValueError("Frame phải là Mono8 hoặc RGB8")
        height, width = raw.shape[:2]
        if not width or not height or width * height > MAX_SENSOR_PIXELS or max(width, height) > MAX_SENSOR_SIDE:
            raise ValueError("Kích thước frame vượt giới hạn")
        if not isinstance(max_side, int) or not 256 <= max_side <= 4096:
            raise ValueError("Cạnh xử lý phải trong khoảng 256–4096")
        x, y, w, h = roi if roi is not None else (0, 0, width, height)
        if (not all(isinstance(v, int) for v in (x, y, w, h)) or x < 0 or y < 0 or
                min(w, h) < 1 or x + w > width or y + h > height):
            raise ValueError("ROI nằm ngoài ảnh camera")
        scale = min(1., max_side / max(w, h), math.sqrt(MAX_PIXELS / (w * h)))
        out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))
        image = Image.fromarray(raw[y:y + h, x:x + w])
        if image.size != (out_w, out_h):
            image = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
        self.pixels = validate_pixels(np.array(image.convert("RGB"))).copy()
        self.metadata = dict(super().describe(), camera_frame=deepcopy(frame.metadata),
            sensor_size=[width, height], roi_xywh=[x, y, w, h], inspection_size=[out_w, out_h],
            sensor_to_inspection_scale=[out_w / w, out_h / h],
            coordinate_space="inspection_image", sensor_image_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        digest = hashlib.sha256(self.pixels.tobytes()).hexdigest()
        self.metadata["image_sha256"] = digest
        self.identity = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def get_frame(self):
        return "industrial:" + self.identity[:16], self.pixels.copy()

    def describe(self):
        return deepcopy(self.metadata)
