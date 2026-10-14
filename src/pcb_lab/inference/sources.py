"""Bounded, read-only image sources shared by the CLI and UI."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps

from pcb_lab.data.manifest import sha256_file
from pcb_lab.data.samples import Sample
from .preprocessing import guard_image_path, to_canonical

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 4_000_000


class BaseImageSource(ABC):
    @abstractmethod
    def get_frame(self) -> tuple[str, np.ndarray]:
        """Return sample ID and RGB uint8 HWC pixels, or raise an input error."""

    def describe(self) -> dict:
        return {"source": type(self).__name__, "ground_truth_available": False}


def validate_pixels(pixels) -> np.ndarray:
    pixels = to_canonical(pixels)
    h, w = pixels.shape[:2]
    if h * w > MAX_PIXELS or max(h, w) > 4096:
        raise ValueError("Ảnh vượt giới hạn 4 triệu pixel hoặc cạnh 4096 px")
    return pixels


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Ảnh trống hoặc vượt giới hạn 20 MiB")
    with Image.open(BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Chỉ hỗ trợ ảnh PNG hoặc JPEG")
        if image.width * image.height > MAX_PIXELS or max(image.size) > 4096:
            raise ValueError("Ảnh vượt giới hạn 4 triệu pixel hoặc cạnh 4096 px")
        image.load()
        return validate_pixels(ImageOps.exif_transpose(image))


class UploadImageSource(BaseImageSource):
    def __init__(self, data: bytes, name: str = "upload"):
        self.data = bytes(data)
        self.name = Path(name).name

    def get_frame(self):
        digest = hashlib.sha256(self.data).hexdigest()
        return f"upload:{digest[:16]}", decode_image(self.data)

    def describe(self):
        return dict(super().describe(), name=self.name, size_bytes=len(self.data),
                    image_sha256=hashlib.sha256(self.data).hexdigest())


class WebcamImageSource(UploadImageSource):
    """A browser camera snapshot; never opens a server-side camera device."""


class FileImageSource(BaseImageSource):
    def __init__(self, path, *, dataset_root=None):
        self.path, self.dataset_root = Path(path), dataset_root

    def get_frame(self):
        path = guard_image_path(self.path, self.dataset_root)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Ảnh vượt giới hạn 20 MiB")
        return path.name, decode_image(path.read_bytes())


class GalleryImageSource(BaseImageSource):
    def __init__(self, sample: Sample):
        self.sample = sample

    def get_frame(self):
        sample = self.sample
        # Guard before any pixel/file-content access, including digest verification.
        path = guard_image_path(sample.image_path, sample._dataset_root, allow_test=sample._allow_test)
        if sha256_file(path) != sample.sha256:
            raise ValueError("Ảnh gallery không khớp SHA256 trong manifest")
        return sample.sample_id, validate_pixels(sample.load_image())

    def describe(self):
        s = self.sample
        return {"source": "gallery", "ground_truth_available": True,
                "ground_truth": "DEFECT" if s.is_defect else "GOOD",
                "ground_truth_boxes": [asdict(b) for b in s.boxes],
                "ground_truth_reference": s.sample_id, "split": s.partition,
                "source_group": s.source_group, "dataset_id": s.source,
                "image_sha256": s.sha256}


class URLImageSource(BaseImageSource):
    def __init__(self, url: str):
        self.url = url.strip()

    def get_frame(self):
        parts = urlsplit(self.url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("URL ảnh phải là HTTP(S), không chứa thông tin đăng nhập")
        request = Request(self.url, headers={"User-Agent": "PCB-Lab-Inspection/1.0"})
        with urlopen(request, timeout=10) as response:
            if urlsplit(response.geturl()).scheme not in {"http", "https"}:
                raise ValueError("URL chuyển hướng không hợp lệ")
            data = response.read(MAX_BYTES + 1)
        return "url:" + hashlib.sha256(data).hexdigest()[:16], decode_image(data)
