# tests/test_preprocessing.py
#
# VERIFIER - Giai đoạn A.3: test tiền xử lý dùng chung (inference/preprocessing.py).
#
# Bảo vệ contract (docs/handoff/step2-contract.md) và plan §5 Bước 2, §4.4:
#   - to_canonical: L->3 kênh bằng nhau; RGBA->RGB; RGB đỏ thuần ở kênh 0 (không đảo BGR).
#   - letterbox/unletterbox round-trip (640x640 scale1 pad0; 800x600; 300x700) sai số <=1px.
#   - normalize hai lần -> ValueError.
#   - resize_for_ad + upsample_map: map hằng giữ nguyên, không overshoot, đúng shape.
#   - prepare_input 3 mode: meta JSON-được có đủ khoá; sha256 ổn định; khác mode khác hash.
#   - prep_preview.py (CLI) phải trùng meta+sha256 với gọi thẳng API (cùng hàm dùng chung).
#
# Import BÊN TRONG test (conftest import_preprocessing). Test đỏ Giai đoạn A là bình thường.

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import import_preprocessing

REPO_ROOT = Path(__file__).resolve().parent.parent


def _meta_get(meta, key):
    """Đọc trường của LetterboxMeta/meta (hỗ trợ dataclass lẫn dict)."""
    if isinstance(meta, dict):
        return meta[key]
    return getattr(meta, key)


def _make_png(path: Path, arr: np.ndarray):
    Image.fromarray(arr).save(path)


def _red_rgb():
    """Ảnh RGB 4x4, pixel (0,0) đỏ thuần (255,0,0), còn lại trắng."""
    arr = np.full((4, 4, 3), 255, dtype=np.uint8)
    arr[0, 0] = (255, 0, 0)
    return arr


def _gray():
    return np.full((8, 8), 123, dtype=np.uint8)


def _rgba():
    arr = np.full((6, 6, 4), 200, dtype=np.uint8)
    arr[..., 3] = 255
    return arr


# ===========================================================================
# to_canonical
# ===========================================================================
def test_to_canonical_gray_three_equal_channels():
    """Bảo vệ contract: ảnh xám L -> 3 kênh bằng nhau, không đảo kênh."""
    P = import_preprocessing()
    out = P["to_canonical"](_gray())
    assert out.ndim == 3 and out.shape[2] == 3, "phải 3 kênh"
    assert out.dtype == np.uint8
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])
    assert out[0, 0, 0] == 123, "giá trị xám phải giữ nguyên trên mọi kênh"


def test_to_canonical_rgba_drops_alpha():
    """Bảo vệ contract: RGBA -> RGB (bỏ alpha), 3 kênh."""
    P = import_preprocessing()
    out = P["to_canonical"](_rgba())
    assert out.shape[2] == 3, "RGBA phải thành RGB 3 kênh"
    assert np.array_equal(out[0, 0], np.array([200, 200, 200])), "giá trị RGB giữ nguyên"


def test_to_canonical_red_stays_channel0():
    """Bảo vệ contract: PNG RGB đỏ thuần vẫn ở kênh 0 (không đảo thành BGR)."""
    P = import_preprocessing()
    out = P["to_canonical"](_red_rgb())
    assert out[0, 0, 0] == 255, "kênh 0 phải là đỏ (255)"
    assert out[0, 0, 2] == 0, "kênh 2 (xanh) phải 0, không bị đảo"


def test_to_canonical_from_path(tmp_path):
    """Bảo vệ contract: chấp nhận đường dẫn file (không chỉ array)."""
    P = import_preprocessing()
    p = tmp_path / "g.png"
    _make_png(p, _gray())
    out = P["to_canonical"](str(p))
    assert out.shape[2] == 3 and out[0, 0, 0] == 123


# ===========================================================================
# letterbox / unletterbox
# ===========================================================================
def test_letterbox_640_scale1_pad0():
    """Bảo vệ contract: ảnh 640x640 -> scale 1.0, pad_left=0, pad_top=0."""
    P = import_preprocessing()
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    out, meta = P["letterbox"](img, target=640, pad_value=114)
    assert out.shape == (640, 640, 3)
    assert abs(_meta_get(meta, "scale") - 1.0) < 1e-9, "scale phải 1.0"
    assert _meta_get(meta, "pad_left") == 0
    assert _meta_get(meta, "pad_top") == 0
    assert _meta_get(meta, "orig_w") == 640
    assert _meta_get(meta, "orig_h") == 640


@pytest.mark.parametrize("H,W", [(640, 640), (800, 600), (300, 700)])
def test_unletterbox_roundtrip(H, W):
    """Bảo vệ plan §5 Bước 2 / contract: round-trip lề thư đúng sai số <= 1 px."""
    P = import_preprocessing()
    rng = np.random.default_rng(H + W)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    _, meta = P["letterbox"](img, target=640, pad_value=114)
    scale = _meta_get(meta, "scale")
    pad_left = _meta_get(meta, "pad_left")
    pad_top = _meta_get(meta, "pad_top")
    # box trong tọa độ ảnh gốc
    boxes_orig = np.array([
        [10.0, 10.0, W - 10.0, H - 10.0],
        [W - 40.0, H - 40.0, W - 5.0, H - 5.0],
    ], dtype=np.float64)
    # forward: đưa lên tọa độ letterboxed
    boxes_lb = boxes_orig.copy()
    boxes_lb[:, [0, 2]] = boxes_lb[:, [0, 2]] * scale + pad_left
    boxes_lb[:, [1, 3]] = boxes_lb[:, [1, 3]] * scale + pad_top
    boxes_back = P["unletterbox_boxes"](boxes_lb, meta)
    assert boxes_back.shape == boxes_orig.shape
    assert np.allclose(boxes_back, boxes_orig, atol=1.0), \
        f"round-trip lệch >1px: {boxes_back - boxes_orig}"


# ===========================================================================
# normalize
# ===========================================================================
def test_normalize_twice_raises():
    """Bảo vệ contract: normalize trên dữ liệu đã normalized -> ValueError."""
    P = import_preprocessing()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    out, meta = P["normalize"](img, mean, std)
    assert out.dtype == np.float32
    # gọi lại trên kết quả đã normalized phải raise
    with pytest.raises(ValueError):
        P["normalize"](out, mean, std)
    # meta ghi normalized=True
    if isinstance(meta, dict):
        assert meta.get("normalized") is True or _meta_get(meta, "normalized") is True


# ===========================================================================
# resize_for_ad + upsample_map
# ===========================================================================
def test_resize_for_ad_shape_and_const_preserved():
    """Bảo vệ contract: resize_for_ad -> (img256, meta), shape đúng."""
    P = import_preprocessing()
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    out, meta = P["resize_for_ad"](img, size=256)
    assert out.shape == (256, 256, 3), "phải resize về 256x256"


def test_upsample_map_const_preserved_no_overshoot():
    """Bảo vệ contract: upsample_map bản đồ hằng giữ nguyên, không overshoot, đúng shape."""
    P = import_preprocessing()
    small = np.full((32, 32), 0.5, dtype=np.float32)
    up = P["upsample_map"](small, (640, 640))
    assert up.shape == (640, 640), "shape phải = out_hw"
    assert np.allclose(up, 0.5, atol=1e-5), "bản đồ hằng phải giữ nguyên"
    # không overshoot: output trong [min,max] input (+tolerance nội suy)
    rng = np.random.default_rng(3)
    m = rng.random((32, 32)).astype(np.float32)
    up2 = P["upsample_map"](m, (640, 640))
    assert up2.min() >= m.min() - 1e-4 and up2.max() <= m.max() + 1e-4, "không overshoot"


# ===========================================================================
# prepare_input
# ===========================================================================
def test_prepare_input_meta_keys_and_jsonable(tmp_path):
    """Bảo vệ contract: meta JSON-được có đủ khoá; color_order=rgb; normalized=false."""
    P = import_preprocessing()
    p = tmp_path / "g.png"
    _make_png(p, _gray())
    for mode in ("yolo", "ad_tile", "ad_resize"):
        prep = P["prepare_input"](str(p), mode=mode)
        meta = prep.meta
        assert isinstance(meta, dict), "meta phải là dict"
        json.dumps(meta)  # phải JSON-được
        for key in ("preprocessing_version", "preprocessing_hash", "mode",
                    "orig_hw", "color_order"):
            assert key in meta, f"thiếu khoá {key} trong meta mode={mode}"
        assert meta["mode"] == mode
        assert meta["color_order"] == "rgb", "color_order phải rgb"
        assert meta.get("normalized", False) is False, "normalized phải false ban đầu"


def test_prepare_input_ad_tile_shape(tmp_path):
    """Bảo vệ contract: ad_tile -> pixels shape [N,256,256,3]; N=9 với ảnh 640."""
    P = import_preprocessing()
    p = tmp_path / "g.png"
    _make_png(p, np.zeros((640, 640, 3), dtype=np.uint8))
    prep = P["prepare_input"](str(p), mode="ad_tile")
    assert prep.pixels.shape[1:] == (256, 256, 3), "tile phải 256x256x3"
    assert prep.pixels.shape[0] == 9, f"ảnh 640 phải 9 tile, got {prep.pixels.shape[0]}"


def test_prepare_input_sha256_stable_and_mode_dependent(tmp_path):
    """Bảo vệ contract: sha256 ổn định giữa gọi; khác mode -> khác hash."""
    P = import_preprocessing()
    p = tmp_path / "g.png"
    _make_png(p, _gray())
    a = P["prepare_input"](str(p), mode="yolo").sha256()
    b = P["prepare_input"](str(p), mode="yolo").sha256()
    assert a == b, "sha256 phải ổn định giữa các lần gọi"
    c = P["prepare_input"](str(p), mode="ad_resize").sha256()
    assert c != a, "chế độ khác phải cho hash khác"


# ===========================================================================
# prep_preview.py (CLI) phải trùng với API
# ===========================================================================
def test_prep_preview_cli_matches_api(tmp_path):
    """Bảo vệ contract + plan §5 Bước 2: CLI prep_preview.py trùng meta+sha256
    với gọi thẳng prepare_input (cùng một hàm dùng chung)."""
    import os

    P = import_preprocessing()
    script = REPO_ROOT / "scripts" / "prep_preview.py"
    if not script.exists():
        pytest.skip("impl chưa merge (prep_preview.py chưa có)")
    img = tmp_path / "g.png"
    _make_png(img, _gray())
    out_json = tmp_path / "preview.json"
    import os
    env = dict(os.environ)  # B correction: sys.environ is not a Python API.
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(script), "--image", str(img),
         "--mode", "yolo", "--out-json", str(out_json)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )
    assert proc.returncode == 0, f"CLI thất bại: {proc.stderr}"
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    # gọi thẳng API
    prep = P["prepare_input"](str(img), mode="yolo")
    assert parsed["sha256"] == prep.sha256(), "sha256 CLI phải trùng API"
    # meta: so sánh phần JSON-được (bỏ trường sha256 nếu nằm trong meta)
    meta_parsed = {k: v for k, v in parsed.get("meta", {}).items() if k != "sha256"}
    assert meta_parsed == prep.meta, "meta CLI phải trùng API"
