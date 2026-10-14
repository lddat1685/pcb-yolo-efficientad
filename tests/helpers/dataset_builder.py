# tests/helpers/dataset_builder.py
#
# Sở hữu: VERIFIER. Helper dựng dataset giả (tmp_path) theo ĐÚNG schema manifest
# thật (benchmarks/deeppcb/...) để test kiểm toán bắt được các lỗi có chủ đích.
#
# Không import bất kỳ module pcb_lab nào — chỉ tạo file/thư mục/text.
# Mọi con số/hash được tính từ dữ liệu, không hard-code kỳ vọng dataset thật.

import hashlib
import json
import shutil
import struct
from pathlib import Path

IMG_SIZE = 640
CLASSES = ["open_circuit", "short", "mouse_bite", "spur", "spurious_copper", "pin_hole"]

# holdout: 4 partition, mỗi partition 2 nhóm, không chồng lấp (clean fixture).
GROUPS = {
    "train": ["group00041", "group12100"],
    "calibration": ["group12000", "group13000"],
    "fusion": ["group50600", "group90100"],
    "test": ["group44000", "group92000"],
}

# 6 cặp partition cần kiểm rò rỉ.
PARTITION_PAIRS = [
    ("train", "calibration"),
    ("train", "fusion"),
    ("train", "test"),
    ("calibration", "fusion"),
    ("calibration", "test"),
    ("fusion", "test"),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _png_rgb(size: int, shade_fn) -> bytes:
    """Tạo PNG RGB `size x size` (color type 2) decode được bằng Pillow.

    shade_fn(x, y) -> (r, g, b). Dùng để impl decode thành RGB (transform convert RGB).
    """
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit truecolor RGB
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter byte
        for x in range(size):
            r, g, b = shade_fn(x, y)
            raw.append(r); raw.append(g); raw.append(b)
    idat_data = zlib.compress(bytes(raw))

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return sig + chunk(b"IHDR", ihdr_data) + chunk(b"IDAT", idat_data) + chunk(b"IEND", b"")


def make_png_gray(size=IMG_SIZE, shade: int = 255) -> bytes:
    """PNG RGB đơn sắc (decode được, mode RGB) — fixture good nền trắng."""
    return _png_rgb(size, lambda x, y: (shade, shade, shade))


def make_png_with_text(size=IMG_SIZE, text: str = "X") -> bytes:
    """PNG RGB có họa tiết để pixel_sha256 khác nhau giữa các ảnh (tránh trùng hash giả)."""
    import hashlib
    h = hashlib.md5(text.encode()).digest()
    base = h[0]

    def shade_fn(x, y):
        s = (base + ((x + y) % 7) * 8) % 256
        return (s, s, s)

    return _png_rgb(size, shade_fn)


def make_corrupt_png() -> bytes:
    """Dữ liệu không decode được thành ảnh (header PNG sai)."""
    return b"\x89PNG\r\n\x1a\nCORRUPT_NOT_A_PNG_FILE" + b"\x00" * 50


def pixel_sha_of_png(path: Path) -> str:
    """Tính RGB pixel SHA-256 theo schema thật: SHA-256 của byte RGB liên tiếp theo hàng.

    Dùng Pillow (có sẵn trong venv bước 1) để decode thành RGB raw.
    """
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        raw = im.tobytes()
    return sha256_bytes(raw)


def yolo_label(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    """Một dòng nhãn YOLO chuẩn hóa (class_id cx cy w h)."""
    return f"{class_id} {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}\n"


def _sample_record(
    sample_id: str,
    split: str,
    source_group: str,
    pair_id: str,
    is_defect: bool,
    image_rel: str,
    label_rel: str,
    image_sha: str,
    label_sha: str,
    pixel_sha: str,
    boxes: list,
    width=IMG_SIZE,
    height=IMG_SIZE,
) -> dict:
    return {
        "sample_id": sample_id,
        "source": "deeppcb",
        "source_group": source_group,
        "design_id": None,
        "physical_board_id": None,
        "capture_session_id": None,
        "parent_large_frame_id": None,
        "pair_ids": [pair_id],
        "split": split,
        "official_split": f"official_{split}",
        "is_defect": is_defect,
        "normal_evidence": None if is_defect else "author_checked_and_cleaned_template",
        "normal_verified_by": None if is_defect else "dataset_author",
        "source_image": image_rel,
        "source_annotation": None if is_defect else label_rel,
        "source_sha256": image_sha,
        "pixel_sha256": pixel_sha,
        "phash": "0000000000000000",
        "source_mode": "L",
        "width": width,
        "height": height,
        "image": image_rel,
        "label": label_rel,
        "boxes": boxes,
        "mask_status": "unavailable",
        "transform": {"convert": "RGB", "resize": None},
        "quality_flags": [],
        "sha256": image_sha,
        "label_sha256": label_sha,
    }


class DatasetFixture:
    """Dựng dataset giả trên một root tùy ý (thường là tmp_path).

    Cung cấp phương thức mutate để test cài các lỗi có chủ đích.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.images_dir = self.root / "benchmarks" / "deeppcb" / "images"
        self.ann_dir = self.root / "benchmarks" / "deeppcb" / "annotations"
        self.configs_dir = self.root / "benchmarks" / "deeppcb" / "configs"
        self.splits_dir = self.root / "benchmarks" / "deeppcb" / "splits"
        self.manifest_dir = self.root / "benchmarks" / "deeppcb" / "manifests"
        self.release_path = self.root / "benchmarks" / "deeppcb" / "release.json"
        for d in (self.images_dir, self.ann_dir, self.configs_dir, self.splits_dir, self.manifest_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.records: list[dict] = []
        self._written_files: dict[str, bytes] = {}  # rel_path -> bytes (để tính release)

    # ---- Ghi ảnh / nhãn và thu thập hash ---------------------------------
    def add_image(self, name: str, png_bytes: bytes, rel_dir="benchmarks/deeppcb/images") -> tuple[str, str]:
        rel = f"{rel_dir}/{name}"
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png_bytes)
        self._written_files[rel] = png_bytes
        return rel, sha256_bytes(png_bytes)

    def add_label(self, name: str, content: str) -> tuple[str, str]:
        rel = f"benchmarks/deeppcb/annotations/{name}"
        p = self.root / rel
        b = content.encode("utf-8")
        p.write_bytes(b)
        self._written_files[rel] = b
        return rel, sha256_bytes(b)

    def add_defect_sample(
        self,
        sample_id: str,
        split: str,
        source_group: str,
        pair_id: str,
        boxes: list,
        *,
        image_name=None,
        png_bytes=None,
    ):
        image_name = image_name or f"{sample_id}.png"
        png_bytes = png_bytes or make_png_with_text(text=sample_id)
        img_rel, img_sha = self.add_image(image_name, png_bytes)
        # nhãn: mỗi box 1 dòng YOLO
        label_lines = []
        for b in boxes:
            cid = b["class_id"]
            x1, y1, x2, y2 = b["xyxy"]
            cx = ((x1 + x2) / 2) / IMG_SIZE
            cy = ((y1 + y2) / 2) / IMG_SIZE
            w = (x2 - x1) / IMG_SIZE
            h = (y2 - y1) / IMG_SIZE
            label_lines.append(yolo_label(cid, cx, cy, w, h))
        label_rel, label_sha = self.add_label(f"{sample_id}.txt", "".join(label_lines))
        # pixel_sha: SHA của byte RGB liên tiếp theo hàng (schema thật)
        pixel_sha = pixel_sha_of_png(self.root / img_rel)
        rec = _sample_record(
            sample_id, split, source_group, pair_id, True,
            img_rel, label_rel, img_sha, label_sha, pixel_sha, boxes,
        )
        self.records.append(rec)
        return rec

    def add_good_sample(self, sample_id, split, source_group, pair_id, *, png_bytes=None):
        image_name = f"{sample_id}.png"
        # good: dùng ảnh có pixel khác nhau giữa các mẫu (tránh rò rỉ hash giả)
        png_bytes = png_bytes or make_png_with_text(text=sample_id)
        img_rel, img_sha = self.add_image(image_name, png_bytes)
        label_rel, label_sha = self.add_label(f"{sample_id}.txt", "")  # good = rỗng
        pixel_sha = pixel_sha_of_png(self.root / img_rel)
        rec = _sample_record(
            sample_id, split, source_group, pair_id, False,
            img_rel, label_rel, img_sha, label_sha, pixel_sha, [],
        )
        self.records.append(rec)
        return rec

    # ---- Ghi configs/manifest/release ------------------------------------
    def write_classes(self):
        data = {
            "names": CLASSES,
            "source_to_class_id": {str(i + 1): i for i in range(6)},
            "pin_hole_is_not_pku_missing_hole": True,
        }
        p = self.configs_dir / "classes.json"
        b = json.dumps(data, indent=2).encode()
        p.write_bytes(b)
        self._written_files["benchmarks/deeppcb/configs/classes.json"] = b

    def write_protocol(self):
        data = {
            "name": "deeppcb_source_group_holdout_v1",
            "training_seeds": [42, 43, 44],
            "pixel_metrics": None,
            "pixel_mask_reason": "No verified pixel masks",
            "source_group_is_not_verified_board_or_design": True,
        }
        p = self.configs_dir / "protocol.json"
        b = json.dumps(data, indent=2).encode()
        p.write_bytes(b)
        self._written_files["benchmarks/deeppcb/configs/protocol.json"] = b

    def write_holdout(self):
        groups = {k: list(v) for k, v in GROUPS.items()}
        component_groups = {g: [g] for glist in groups.values() for g in glist}
        # samples: partition -> list sample_id (theo schema thật)
        samples = {part: [] for part in groups}
        pair_ids = {part: [] for part in groups}
        for r in self.records:
            samples[r["split"]].append(r["sample_id"])
            for pid in r["pair_ids"]:
                if pid not in pair_ids[r["split"]]:
                    pair_ids[r["split"]].append(pid)
        actual_pair_counts = {part: len(pair_ids[part]) for part in groups}
        data = {
            "protocol": "source_group_holdout_v1",
            "partitions": ["train", "calibration", "fusion", "test"],
            "group_kind": "source_folder_proxy_not_verified_design_or_board",
            "groups": groups,
            "component_groups": component_groups,
            "target_pair_ratios": {"train": 0.6, "calibration": 0.15, "fusion": 0.1, "test": 0.15},
            "actual_pair_counts": actual_pair_counts,
            "selection": "exhaustive_minimum_squared_pair_ratio_error; lexical_tie_break; no_model_scores",
            "official_split_preserved_as_metadata_only": True,
            "seed_note": "Split allocation is deterministic; training seeds are separate.",
            "samples": samples,
            "pair_ids": pair_ids,
        }
        p = self.splits_dir / "holdout.json"
        b = json.dumps(data, indent=2).encode()
        p.write_bytes(b)
        self._written_files["benchmarks/deeppcb/splits/holdout.json"] = b

    def write_manifest(self):
        p = self.manifest_dir / "samples.jsonl"
        lines = [json.dumps(r, ensure_ascii=False) for r in self.records]
        b = ("\n".join(lines) + "\n").encode("utf-8")
        p.write_bytes(b)
        self._written_files["benchmarks/deeppcb/manifests/samples.jsonl"] = b

    def write_release(self, *, corrupt: bool = False, missing: list[str] | None = None):
        """Tính đúng SHA-256 của mọi file đã ghi vào release.json.

        corrupt=True: làm sai 1 hash khoá. missing: bỏ một file khỏi map.
        """
        files = dict(self._written_files)
        for m in missing or []:
            files.pop(m, None)
        release_files = {k: sha256_bytes(v) for k, v in files.items()}
        if corrupt:
            some_key = next(iter(release_files))
            release_files[some_key] = "0" * 64
        data = {
            "version": "deeppcb_source_group_holdout_v1",
            "frozen_at": "2026-09-23T00:00:00+07:00",
            "dataset_signature": "0" * 64,
            "files": release_files,
            "frozen": True,
        }
        b = json.dumps(data, indent=2).encode()
        self.release_path.write_bytes(b)

    # ---- Tiện ích mutate cho test ---------------------------------------
    def remove_manifest(self):
        m = self.manifest_dir / "samples.jsonl"
        if m.exists():
            m.unlink()

    def remove_release(self):
        if self.release_path.exists():
            self.release_path.unlink()

    def overwrite_record_image(self, sample_id: str, png_bytes: bytes):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        rel = rec["image"]
        p = self.root / rel
        p.write_bytes(png_bytes)
        rec["sha256"] = sha256_bytes(png_bytes)
        rec["source_sha256"] = rec["sha256"]
        # cập nhật lại _written_files để release tính đúng nếu gọi lại
        self._written_files[rel] = png_bytes

    def overwrite_record_label(self, sample_id: str, content: str):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        rel = rec["label"]
        p = self.root / rel
        b = content.encode("utf-8")
        p.write_bytes(b)
        rec["label_sha256"] = sha256_bytes(b)
        self._written_files[rel] = b

    def set_record_image_size(self, sample_id: str, w: int, h: int):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        rec["width"] = w
        rec["height"] = h

    def set_record_label_nonempty_good(self, sample_id: str, content: str = "0 0.5 0.5 0.1 0.1\n"):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        assert rec["is_defect"] is False, "chỉ áp dụng cho good"
        rel = rec["label"]
        p = self.root / rel
        b = content.encode("utf-8")
        p.write_bytes(b)
        rec["label_sha256"] = sha256_bytes(b)
        self._written_files[rel] = b

    def set_record_label_empty_defect(self, sample_id: str):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        assert rec["is_defect"] is True, "chỉ áp dụng cho defect"
        rel = rec["label"]
        p = self.root / rel
        b = b""
        p.write_bytes(b)
        rec["label_sha256"] = sha256_bytes(b)
        self._written_files[rel] = b

    def set_record_box_invalid(self, sample_id: str, xyxy):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        rec["boxes"][0]["xyxy"] = list(xyxy)

    def set_record_class_id(self, sample_id: str, class_id: int):
        rec = next(r for r in self.records if r["sample_id"] == sample_id)
        rec["boxes"][0]["class_id"] = class_id
        if class_id < len(CLASSES):
            rec["boxes"][0]["class_name"] = CLASSES[class_id]

    def inject_leakage(self, pair=("train", "calibration"), kind="source_group"):
        """Cài rò rỉ giữa 2 partition.

        kind: 'source_group' | 'pair' | 'image_hash' | 'pixel_hash'
        Bằng cách copy 1 sample từ partition A sang B với cùng giá trị tương ứng.
        """
        a, b = pair
        src = next(r for r in self.records if r["split"] == a)
        # tạo bản sao vào partition b
        new_id = f"{src['sample_id']}__leak_{kind}"
        rec = dict(src)
        rec["sample_id"] = new_id
        rec["split"] = b
        # dùng cùng source_group / pair / image hash / pixel hash tuỳ kind
        if kind == "source_group":
            pass  # giữ nguyên source_group
        elif kind == "pair":
            pass  # giữ nguyên pair_ids
        elif kind == "image_hash":
            pass  # giữ nguyên sha256
        elif kind == "pixel_hash":
            pass  # giữ nguyên pixel_sha256
        # copy ảnh/nhãn vật lý sang để decode được
        src_img = self.root / src["image"]
        dst_img = self.root / "benchmarks" / "deeppcb" / "images" / f"{new_id}.png"
        dst_img.write_bytes(src_img.read_bytes())
        src_lbl = self.root / src["label"]
        dst_lbl = self.root / "benchmarks" / "deeppcb" / "annotations" / f"{new_id}.txt"
        if src_lbl.exists():
            dst_lbl.write_bytes(src_lbl.read_bytes())
        rec["image"] = f"benchmarks/deeppcb/images/{new_id}.png"
        rec["label"] = f"benchmarks/deeppcb/annotations/{new_id}.txt"
        # cập nhật sha theo file mới copy (image + pixel)
        rec["sha256"] = sha256_bytes(dst_img.read_bytes())
        rec["source_sha256"] = rec["sha256"]
        rec["pixel_sha256"] = pixel_sha_of_png(dst_img)
        self._written_files[rec["image"]] = dst_img.read_bytes()
        self.records.append(rec)


def build_clean_fixture(root: Path) -> DatasetFixture:
    """Dựng dataset giả SẠCH: 4 partition, mỗi partition 1 good + 1 defect,
    6 lớp phủ đủ, 6 cặp partition KHÔNG rò rỉ. Dùng làm oracle PASS.

    Bảo vệ plan §1.2 (phân chia 4 partition) và contract (schema/leakage 6 cặp).
    """
    fx = DatasetFixture(root)
    # box mẫu hợp lệ cho mỗi lớp
    valid_boxes = [
        {"source_class_id": 1, "class_id": 0, "class_name": "open_circuit", "xyxy": [539.0, 259.0, 592.0, 316.0]},
        {"source_class_id": 2, "class_id": 1, "class_name": "short", "xyxy": [454.0, 300.0, 493.0, 396.0]},
        {"source_class_id": 3, "class_id": 2, "class_name": "mouse_bite", "xyxy": [466.0, 441.0, 493.0, 470.0]},
        {"source_class_id": 4, "class_id": 3, "class_name": "spur", "xyxy": [331.0, 248.0, 364.0, 283.0]},
        {"source_class_id": 5, "class_id": 4, "class_name": "spurious_copper", "xyxy": [151.0, 149.0, 182.0, 175.0]},
        {"source_class_id": 6, "class_id": 5, "class_name": "pin_hole", "xyxy": [492.0, 28.0, 525.0, 55.0]},
    ]
    for split, groups in GROUPS.items():
        g = groups[0]
        pair = g[5:]  # "group00041" -> "00041"
        # good (pixel khác nhau nhờ text=sample_id -> không rò rỉ hash giả)
        fx.add_good_sample(f"deeppcb_{pair}000_good", split, g, pair)
        # defect: 1 box lớp duy nhất theo thứ tự để phủ 6 lớp trên 6 sample đầu
    # thêm defect với các lớp khác nhau (đủ 6 lớp)
    idx = 0
    for split, groups in GROUPS.items():
        for g in groups:
            pair = g[5:]
            box = dict(valid_boxes[idx % 6])
            fx.add_defect_sample(
                f"deeppcb_{pair}001_defect", split, g, pair,
                [box], png_bytes=make_png_with_text(text=f"{split}{g}"),
            )
            idx += 1
    fx.write_classes()
    fx.write_protocol()
    fx.write_holdout()
    fx.write_manifest()
    fx.write_release()
    return fx
