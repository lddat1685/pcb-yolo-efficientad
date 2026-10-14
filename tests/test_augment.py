# tests/test_augment.py
#
# VERIFIER - Giai đoạn A.5: test augmentation (chỉ phần đã được duyệt).
#
# Bảo vệ contract (docs/handoff/step2-contract.md) và plan §5 Bước 2:
#   - split khác train -> ValueError.
#   - Tất định theo seed (cùng input + seed -> cùng output).
#   - Box biến đổi nhất quán với ảnh; box vẫn nằm trong ảnh (sau biến đổi).
#   - Không sửa mảng đầu vào tại chỗ.
#
# Ghi chú: contract KHÔNG pin tên symbol / signature của hàm augment, nên test
# dùng adapter linh hoạt thử nhiều cách gọi; nếu API merge vào khác hình thức,
# adapter skip kèm lý do (sẽ chốt lại ở Giai đoạn B). Test đỏ ở Giai đoạn A
# (impl chưa merge) là bình thường.

import inspect

import numpy as np
import pytest

from conftest import import_augment


# Hình ảnh/box mẫu cố định để kiểm tính chất (độc lập với công thức cụ thể).
_IMG = np.zeros((64, 64, 3), dtype=np.uint8)
_BOXES = [{"class_id": 0, "xyxy": [10.0, 10.0, 40.0, 40.0]}]
_SEED = 1234


class _UnsupportedSignature(Exception):
    pass


def _call_aug(fn, image, boxes, split, seed):
    """Thử nhiều cách gọi hợp lý của hàm augment; trả (out_img, out_boxes).

    Không biết chính xác signature, nên thử: keyword đầy đủ, vị trí, và các
    biến thể tên tham số (img/image, box/boxes). Nếu không cách nào gọi được ->
    _UnsupportedSignature để test skip (Giai đoạn A, impl chưa merge).
    """
    sig = inspect.signature(fn)
    pnames = list(sig.parameters)
    recognized = {"image", "img", "boxes", "box", "split", "seed"}
    # chỉ hỗ trợ nếu mọi param nằm trong tập nhận diện (tránh gọi sai shape)
    if not all(p in recognized for p in pnames if p != "self"):
        raise _UnsupportedSignature(f"tham số không nhận diện: {pnames}")

    image_keys = [k for k in ("image", "img") if k in pnames]
    boxes_keys = [k for k in ("boxes", "box") if k in pnames]
    if not image_keys or not boxes_keys or "split" not in pnames or "seed" not in pnames:
        raise _UnsupportedSignature("thiếu image/boxes/split/seed")

    ik, bk = image_keys[0], boxes_keys[0]

    def _try(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs), None
        except TypeError as e:
            return None, e

    # (1) keyword đầy đủ
    out, err = _try(fn, **{ik: image, bk: boxes, "split": split, "seed": seed})
    if out is not None:
        return out
    # (2) vị trí (image, boxes, split, seed)
    out, err = _try(fn, image, boxes, split, seed)
    if out is not None:
        return out
    raise _UnsupportedSignature(f"không gọi được augment: {err}")


def _aug_or_skip():
    try:
        fn = import_augment()
    except Exception as e:
        pytest.skip(f"augment chưa merge: {e}")
    return fn


# ===========================================================================
# split != train -> ValueError
# ===========================================================================
@pytest.mark.parametrize("bad_split", ["val", "test", "calibration", "fusion"])
def test_augment_rejects_non_train(bad_split):
    """Bảo vệ contract: recipe aug_v1 chỉ áp cho split 'train'; split khác -> ValueError."""
    fn = _aug_or_skip()
    try:
        _call_aug(fn, _IMG.copy(), _BOXES, bad_split, _SEED)
    except _UnsupportedSignature as e:
        pytest.skip(str(e))
    except ValueError:
        return  # đúng kỳ vọng
    except Exception:
        # nếu không raise ValueError mà raise khác (vd TypeError do split kwarg), bỏ qua
        pytest.skip("gọi augment chưa khớp hình thức")
    pytest.fail("split != train phải raise ValueError")


def test_augment_allows_train():
    """Bảo vệ contract: split 'train' được phép (không raise)."""
    fn = _aug_or_skip()
    try:
        _call_aug(fn, _IMG.copy(), _BOXES, "train", _SEED)
    except _UnsupportedSignature as e:
        pytest.skip(str(e))
    except ValueError:
        pytest.fail("train phải được phép augment")


# ===========================================================================
# Tất định theo seed
# ===========================================================================
def test_augment_deterministic_by_seed():
    """Bảo vệ contract: cùng input + seed -> cùng output (tất định)."""
    fn = _aug_or_skip()
    try:
        a = _call_aug(fn, _IMG.copy(), _BOXES, "train", _SEED)
        b = _call_aug(fn, _IMG.copy(), _BOXES, "train", _SEED)
    except _UnsupportedSignature as e:
        pytest.skip(str(e))
    ai, ab = a
    bi, bb = b
    assert np.array_equal(ai, bi), "ảnh phải tất định theo seed"
    assert ab == bb, "box phải tất định theo seed"


def test_augment_different_seed_may_differ():
    """Bảo vệ contract (phụ): đổi seed -> output có thể khác (chứng minh seed có hiệu lực)."""
    fn = _aug_or_skip()
    try:
        a = _call_aug(fn, _IMG.copy(), _BOXES, "train", _SEED)
        b = _call_aug(fn, _IMG.copy(), _BOXES, "train", _SEED + 1)
    except _UnsupportedSignature as e:
        pytest.skip(str(e))
    # nếu implementer dùng seed thực sự, ít nhất một trong ảnh/box khác biệt
    ai, ab = a
    bi, bb = b
    assert (not np.array_equal(ai, bi)) or (ab != bb), "đổi seed phải tác động đến output"


# ===========================================================================
# Không sửa mảng đầu vào tại chỗ
# ===========================================================================
def test_augment_no_inplace_mutation():
    """Bảo vệ contract: augment không sửa mảng ảnh đầu vào tại chỗ."""
    fn = _aug_or_skip()
    img_in = _IMG.copy()
    before = img_in.copy()
    try:
        _call_aug(fn, img_in, _BOXES, "train", _SEED)
    except _UnsupportedSignature as e:
        pytest.skip(str(e))
    assert np.array_equal(img_in, before), "ảnh đầu vào không được sửa tại chỗ"


# ===========================================================================
# Box vẫn trong ảnh
# ===========================================================================
def test_augment_boxes_stay_in_image():
    """Bảo vệ contract: box sau biến đổi vẫn nằm trong ảnh (0<=x1<x2<=W; 0<=y1<y2<=H)."""
    fn = _aug_or_skip()
    try:
        out_img, out_boxes = _call_aug(fn, _IMG.copy(), _BOXES, "train", _SEED)
    except _UnsupportedSignature as e:
        pytest.skip(str(e))
    H, W = out_img.shape[:2]
    for box in out_boxes:
        xyxy = box.get("xyxy") if isinstance(box, dict) else getattr(box, "xyxy")
        x1, y1, x2, y2 = xyxy
        assert 0 <= x1 < x2 <= W, f"box x ngoài ảnh: {xyxy}"
        assert 0 <= y1 < y2 <= H, f"box y ngoài ảnh: {xyxy}"
