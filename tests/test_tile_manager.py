# tests/test_tile_manager.py
#
# VERIFIER - Giai đoạn A.2: test tính chất TOÁN HỌC của TileManager.
#
# Nguyên tắc: KHÔNG dựa vào công thức trộn (feathering) cụ thể của implementer.
# Chỉ kiểm các tính chất bắt buộc từ plan §4.4 / §11.1 và contract
# (docs/handoff/step2-contract.md): plan 9 tile, overlap 32, row-major, phủ kín;
# split không đệm = pixel gốc; stitch hằng/phục hồi tuyến tính/sở hữu pixel/đệm
# feathering đơn điệu/float32 không NaN; boxes_to_global offset+clip+bỏ đệm+score;
# global_nms IoU>0.45 nghiêm ngặt (0.45 GIỮ), không phụ thuộc thứ tự, rỗng OK;
# tile_size != 256 -> ValueError.
#
# Import module BÊN TRONG test (từ conftest import_tile_manager). Test đỏ ở
# Giai đoạn A là bình thường (impl chưa merge).

import numpy as np
import pytest

from conftest import import_tile_manager


def _spec_fields(spec):
    """Trích (tile_id, row, col, x0, y0, size) từ TileSpec (dataclass hoặc dict)."""
    if isinstance(spec, dict):
        return spec["tile_id"], spec["row"], spec["col"], spec["x0"], spec["y0"], spec["size"]
    return (
        getattr(spec, "tile_id"), getattr(spec, "row"), getattr(spec, "col"),
        getattr(spec, "x0"), getattr(spec, "y0"), getattr(spec, "size"),
    )


def _find_spec(specs, x0, y0):
    for s in specs:
        _, _, _, sx, sy, _ = _spec_fields(s)
        if sx == x0 and sy == y0:
            return s
    raise KeyError(f"no spec at x0={x0}, y0={y0}")


# ===========================================================================
# plan
# ===========================================================================
def test_plan_640_gives_9_tiles():
    """Bảo vệ contract: plan(640,640) -> đúng 9 tile, x0,y0 ∈ {0,224,448}, overlap 32."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    assert len(specs) == 9, f"phải 9 tile, got {len(specs)}"
    x0s = sorted({_spec_fields(s)[3] for s in specs})
    y0s = sorted({_spec_fields(s)[4] for s in specs})
    assert x0s == [0, 224, 448], f"x0 phải {{0,224,448}}, got {x0s}"
    assert y0s == [0, 224, 448], f"y0 phải {{0,224,448}}, got {y0s}"
    # overlap = tile_size - stride = 256 - 224 = 32
    assert 256 - 224 == 32, "overlap phải 32"
    # mọi tile size = 256
    assert all(_spec_fields(s)[5] == 256 for s in specs), "mọi tile size phải 256"


def test_plan_640_row_major():
    """Bảo vệ contract: thứ tự row-major (col tăng nhanh nhất, rồi row)."""
    TileManager = import_tile_manager()
    specs = TileManager().plan(640, 640)
    seq = [(_spec_fields(s)[2], _spec_fields(s)[1]) for s in specs]  # (col, row)
    assert seq == [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
    ], f"thứ tự phải row-major, got {seq}"


@pytest.mark.parametrize("H,W", [(256, 256), (300, 300), (700, 500), (1, 1), (640, 300)])
def test_plan_count_formula(H, W):
    """Bảo vệ [đính chính] contract: số tile mỗi chiều = ceil((D-256)/224)+1 nếu D>256, ngược lại 1."""
    TileManager = import_tile_manager()
    specs = TileManager().plan(H, W)
    nrows = (H - 256 + 223) // 224 + 1 if H > 256 else 1
    ncols = (W - 256 + 223) // 224 + 1 if W > 256 else 1
    assert len(specs) == nrows * ncols, f"{H}x{W}: {len(specs)} != {nrows*ncols}"
    assert all(_spec_fields(s)[5] == 256 for s in specs), "mọi tile size 256"


@pytest.mark.parametrize("H,W", [(256, 256), (300, 300), (700, 500), (1, 1), (640, 300)])
def test_plan_tiles_cover_every_pixel(H, W):
    """Bảo vệ contract: hợp các tile phủ kín mọi pixel ảnh (không lõi trống)."""
    TileManager = import_tile_manager()
    specs = TileManager().plan(H, W)
    covered = np.zeros((H, W), dtype=bool)
    for s in specs:
        _, _, _, x0, y0, size = _spec_fields(s)
        # tile phủ [x0, x0+size) x [y0, y0+size) trên canvas; chỉ đếm phần trong ảnh
        xlo, xhi = max(0, x0), min(W, x0 + size)
        ylo, yhi = max(0, y0), min(H, y0 + size)
        if xhi > xlo and yhi > ylo:
            covered[ylo:yhi, xlo:xhi] = True
    assert covered.all(), f"{H}x{W}: có pixel không được tile nào phủ"


# ===========================================================================
# split
# ===========================================================================
def test_split_nonpadded_equals_original():
    """Bảo vệ contract: phần KHÔNG đệm của mỗi tile = đúng pixel ảnh gốc."""
    TileManager = import_tile_manager()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    tm = TileManager()
    specs = tm.plan(640, 640)
    tiles, specs2 = tm.split(img)
    assert len(tiles) == len(specs) == 9
    for t, s in zip(tiles, specs):
        _, _, _, x0, y0, size = _spec_fields(s)
        xlo, xhi = x0, min(640, x0 + size)
        ylo, yhi = y0, min(640, y0 + size)
        if xhi > xlo and yhi > ylo:
            assert np.array_equal(t[ylo - y0:yhi - y0, xlo - x0:xhi - x0],
                                  img[ylo:yhi, xlo:xhi]), \
                f"tile tại ({x0},{y0}) phần không đệm phải = pixel gốc"


@pytest.mark.parametrize("pad_mode", ["reflect", "replicate"])
def test_split_padding_reflect_replicate(pad_mode):
    """Bảo vệ contract: đệm reflect/replicate đúng bản chất từng chế độ.

    Ảnh gradient cột rút gọn (pixel = cột mod 16, nằm trong uint8) -> replicate:
    cột đệm = cột W-1; reflect: cột đệm j = cột (2W-2-j). Không dùng công thức impl.
    """
    TileManager = import_tile_manager()
    W = 640
    # gradient cột rút gọn: cột x có giá trị x mod 16 (giữ trong uint8)
    colvals = (np.arange(W) % 16).astype(np.uint8)
    img = np.broadcast_to(colvals[None, :, None], (640, W, 3)).copy()
    tm = TileManager(pad_mode=pad_mode)
    specs = tm.plan(640, 640)
    tiles, _ = tm.split(img)
    # tile cuối cùng hàng 0 (x0=448) có vùng đệm cột 640..703
    spec = _find_spec(specs, 448, 0)
    idx = specs.index(spec)
    tile = tiles[idx]
    pad_local_lo = 640 - 448  # = 192 (cột đầu đệm trong tile)
    for j in range(pad_local_lo, 256):
        img_col = 448 + j  # tọa độ canvas
        if pad_mode == "replicate":
            expected = (W - 1) % 16
        else:  # reflect (numpy/Pillow: không lặp điểm biên, cột đầu đệm = W-2)
            expected = ((2 * W - 2) - img_col) % 16
        # lấy kênh 0; mọi kênh bằng nhau trong ảnh gradient này
        assert tile[0, j, 0] == expected, \
            f"{pad_mode}: cột đệm {img_col} phải = {expected}, got {tile[0, j, 0]}"


def test_split_padding_constant_is_uniform():
    """Bảo vệ contract (mức yếu): đệm constant là hằng số trên dải đệm.

    Contract không pin giá trị constant; chỉ kiểm đệm là đồng nhất (không nhiễu).
    """
    TileManager = import_tile_manager()
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    tm = TileManager(pad_mode="constant")
    specs = tm.plan(640, 640)
    tiles, _ = tm.split(img)
    spec = _find_spec(specs, 448, 0)
    tile = tiles[specs.index(spec)]
    pad_local_lo = 640 - 448
    strip = tile[:, pad_local_lo:, 0]
    assert np.all(strip == strip.flat[0]), "đệm constant phải đồng nhất"


# ===========================================================================
# stitch
# ===========================================================================
def test_stitch_constant_map_recovers_constant():
    """Bảo vệ contract (a): map hằng c -> mọi pixel bằng c."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    c = 0.37
    tile_maps = np.full((len(specs), 256, 256), c, dtype=np.float32)
    out = tm.stitch(tile_maps, specs, (640, 640))
    assert out.shape == (640, 640)
    assert np.allclose(out, c, atol=1e-6), "map hằng phải phục hồi hằng"


def test_stitch_linear_function_recovered():
    """Bảo vệ contract (b): f(x,y)=ax+by cắt tile -> khôi phục f (sai số <= 1e-5).

    Feathering là trọng số tổng = 1 => tổng trọng số*f(p) = f(p) EXACT với f tuyến tính,
    độc lập với công thức trộn cụ thể. Mỗi tile luôn 256x256 (canvas 704x704 có đệm),
    nên tile map lấy từ canvas f mở rộng (ngoài ảnh = 0, không ảnh hưởng pixel trong ảnh).
    """
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    a, b = 0.001, 0.002
    ys, xs = np.mgrid[0:640, 0:640]
    f = (a * xs + b * ys).astype(np.float64)
    # canvas 704x704 (đúng contract: ảnh 640 -> canvas 704 với đệm phải/dưới)
    canvas = np.zeros((704, 704), dtype=np.float64)
    canvas[0:640, 0:640] = f
    tile_maps = np.stack([canvas[y0:y0 + 256, x0:x0 + 256].astype(np.float32)
                          for _, _, _, x0, y0, _ in (_spec_fields(s) for s in specs)])
    out = tm.stitch(tile_maps, specs, (640, 640))
    assert np.allclose(out, f, atol=1e-5), "f tuyến tính phải khôi phục sai số <= 1e-5"


def test_stitch_pixel_owned_by_single_tile():
    """Bảo vệ contract (c): pixel chỉ thuộc 1 tile = đúng giá trị tile đó."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    tile_maps = np.zeros((len(specs), 256, 256), dtype=np.float32)
    # tile (0,0) mang giá trị 0.9; pixel (112,112) chỉ thuộc tile (0,0)
    spec00 = _find_spec(specs, 0, 0)
    tile_maps[specs.index(spec00)] = 0.9
    out = tm.stitch(tile_maps, specs, (640, 640))
    assert abs(out[112, 112] - 0.9) < 1e-5, "pixel đơn sở hữu phải = giá trị tile"


def test_stitch_overlap_monotone_no_large_jump():
    """Bảo vệ contract (d): 2 tile 0 và 1 -> vùng chồng lấn nằm giữa 0 và 1,
    đơn điệu, không bước nhảy > 2/overlap giữa 2 pixel kề nhau."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    tile_maps = np.zeros((len(specs), 256, 256), dtype=np.float32)
    # tile cuối hàng 0 (x0=448) = 1; các tile khác = 0
    spec_last = _find_spec(specs, 448, 0)
    tile_maps[specs.index(spec_last)] = 1.0
    out = tm.stitch(tile_maps, specs, (640, 640))
    # dải chuyển tiếp nằm ở cột 448..479 (chồng lấn giữa x0=224 và x0=448)
    row = out[112, :]  # hàng thuộc hàng 0
    band = row[448:480]
    assert band.min() >= -1e-6 and band.max() <= 1 + 1e-6, "phải nằm trong [0,1]"
    assert np.all(np.diff(band) >= -1e-6), "phải đơn điệu không giảm dọc theo x"
    max_jump = np.max(np.abs(np.diff(band)))
    assert max_jump <= 2 / 32 + 1e-6, f"bước nhảy kề phải <= 2/overlap, got {max_jump}"


def test_stitch_dtype_float32_finite():
    """Bảo vệ contract (e): đầu ra float32, không NaN/inf, mép tile không chia 0."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    rng = np.random.default_rng(2)
    tile_maps = rng.random((len(specs), 256, 256)).astype(np.float32)
    out = tm.stitch(tile_maps, specs, (640, 640))
    assert out.dtype == np.float32, "đầu ra phải float32"
    assert np.all(np.isfinite(out)), "không được có NaN/inf (kể cả tại mép tile)"


# ===========================================================================
# boxes_to_global
# ===========================================================================
def test_boxes_to_global_offset():
    """Bảo vệ contract: box cục bộ (10,10,50,50) ở tile x0=224,y0=0 -> (234,10,274,50)."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    spec = _find_spec(specs, 224, 0)
    per_tile = [np.empty((0, 5), dtype=np.float64) for _ in specs]
    per_tile[specs.index(spec)] = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
    out = tm.boxes_to_global(per_tile, specs, (640, 640))
    assert out.shape == (1, 5)
    assert np.allclose(out[0], [234.0, 10.0, 274.0, 50.0, 0.9]), f"offset sai: {out[0]}"


def test_boxes_to_global_clip_and_score_kept():
    """Bảo vệ contract: box lố ảnh được clip vào ảnh; score giữ nguyên."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    spec = _find_spec(specs, 0, 0)
    # box (600,10,700,50) -> clip x2 từ 700 xuống 640
    per_tile = [np.empty((0, 5), dtype=np.float64) for _ in specs]
    per_tile[specs.index(spec)] = np.array([[600.0, 10.0, 700.0, 50.0, 0.42]])
    out = tm.boxes_to_global(per_tile, specs, (640, 640))
    assert np.allclose(out[0], [600.0, 10.0, 640.0, 50.0, 0.42]), f"clip/score sai: {out[0]}"


def test_boxes_to_global_drops_box_in_padding():
    """Bảo vệ contract: box nằm hoàn toàn trong vùng đệm (ngoài ảnh) bị bỏ."""
    TileManager = import_tile_manager()
    tm = TileManager()
    specs = tm.plan(640, 640)
    spec = _find_spec(specs, 448, 0)  # tile cuối, cột 448..703 (đệm 640..703)
    # box cục bộ (200,10,250,50) -> global x 648..698 > 640 => rỗng sau clip
    per_tile = [np.empty((0, 5), dtype=np.float64) for _ in specs]
    per_tile[specs.index(spec)] = np.array([[200.0, 10.0, 250.0, 50.0, 0.7]])
    out = tm.boxes_to_global(per_tile, specs, (640, 640))
    assert out.shape[0] == 0, f"box trong đệm phải bị bỏ, got {out}"


# ===========================================================================
# global_nms
# ===========================================================================
def test_global_nms_merge_high_iou():
    """Bảo vệ plan §11.1: 2 box trùng từ 2 tile liền kề IoU>0.45 gộp còn 1, giữ score cao."""
    TileManager = import_tile_manager()
    dets = np.array([
        [100.0, 100.0, 200.0, 200.0, 0.9],
        [110.0, 100.0, 210.0, 200.0, 0.5],  # IoU cao với box trên
    ], dtype=np.float64)
    out = TileManager().global_nms(dets, iou=0.45)
    assert out.shape[0] == 1, f"IoU cao phải gộp còn 1, got {out.shape[0]}"
    assert abs(out[0][4] - 0.9) < 1e-9, "phải giữ score cao nhất"


def test_global_nms_threshold_strict_045_kept():
    """Bảo vệ contract: IoU ĐÚNG BẰNG 0.45 -> GIỮ (nghiêm ngặt >).

    Ca dựng bằng số học float64 chính xác: A=[0,0,145,145], B=[55,0,200,145]
    => inter=13050, union=29000, IoU=13050/29000=0.45 (nguyên).
    """
    TileManager = import_tile_manager()
    dets = np.array([
        [0.0, 0.0, 145.0, 145.0, 0.9],
        [55.0, 0.0, 200.0, 145.0, 0.5],
    ], dtype=np.float64)
    iou = 13050.0 / 29000.0
    assert abs(iou - 0.45) < 1e-15, "ca must be exactly 0.45"
    out = TileManager().global_nms(dets, iou=0.45)
    assert out.shape[0] == 2, f"IoU=0.45 phải GIỮ (2 box), got {out.shape[0]}"


def test_global_nms_above_045_dropped():
    """Bảo vệ contract: IoU > 0.45 (0.4572) -> gộp còn 1."""
    TileManager = import_tile_manager()
    dets = np.array([
        [0.0, 0.0, 145.0, 145.0, 0.9],
        [54.0, 0.0, 199.0, 145.0, 0.5],  # inter=13195, union=28855, IoU=0.4572
    ], dtype=np.float64)
    out = TileManager().global_nms(dets, iou=0.45)
    assert out.shape[0] == 1, f"IoU>0.45 phải gộp, got {out.shape[0]}"


def test_global_nms_order_independent():
    """Bảo vệ contract: kết quả không phụ thuộc thứ tự đầu vào."""
    TileManager = import_tile_manager()
    a = np.array([
        [0.0, 0.0, 100.0, 100.0, 0.9],
        [10.0, 0.0, 110.0, 100.0, 0.8],
        [500.0, 500.0, 600.0, 600.0, 0.7],
    ], dtype=np.float64)
    b = a[::-1].copy()
    out_a = TileManager().global_nms(a, iou=0.45)
    out_b = TileManager().global_nms(b, iou=0.45)
    # sắp xếp theo (x1,y1,x2,y2) để so
    ka = np.sort(out_a[:, :4], axis=0).ravel()
    kb = np.sort(out_b[:, :4], axis=0).ravel()
    assert out_a.shape == out_b.shape, "cùng số box"
    assert np.allclose(ka, kb), "thứ tự đầu vào không được thay đổi tập kết quả"


def test_global_nms_empty_ok():
    """Bảo vệ contract: danh sách rỗng không lỗi."""
    TileManager = import_tile_manager()
    out = TileManager().global_nms(np.empty((0, 5), dtype=np.float64), iou=0.45)
    assert out.shape == (0, 5) or out.shape[0] == 0, f"rỗng phải OK, got {out.shape}"


# ===========================================================================
# tile_size constraint
# ===========================================================================
@pytest.mark.parametrize("bad_size", [320, 512, 128])
def test_tile_manager_rejects_non_256(bad_size):
    """Bảo vệ contract + plan §5 Bước 2 [R7/S2]: tile.size chỉ nhận 256 -> ValueError."""
    TileManager = import_tile_manager()
    with pytest.raises(ValueError):
        TileManager(tile_size=bad_size)
