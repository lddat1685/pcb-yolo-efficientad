"""Approved aug_v1: exact dihedral pixel permutations on train only."""
from __future__ import annotations

import numpy as np

from .samples import Box
from pcb_lab.inference.preprocessing import load_preprocessing_config, preprocessing_hash, to_canonical


def apply_augmentation(image, boxes, *, partition, seed, recipe_id="aug_v1"):
    """Return (uint8 HWC image, tuple[Box,...], JSON metadata).

    Choose k uniformly in 0..3, rotate k*90 degrees counterclockwise, then
    horizontally flip with probability 0.5. No crop/interpolation/photometrics.
    """
    if partition != "train":
        raise ValueError("aug_v1 is allowed only on train")
    if recipe_id != "aug_v1":
        raise ValueError("Unknown augmentation recipe")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    arr = to_canonical(image)
    h, w = arr.shape[:2]
    boxes = tuple(boxes)
    for box in boxes:
        if not isinstance(box, Box):
            raise ValueError("boxes must contain Box objects")
        x1, y1, x2, y2 = box.xyxy
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Box lies outside the input image")
    rng = np.random.default_rng(seed)
    k, flip = int(rng.integers(0, 4)), bool(rng.random() < 0.5)
    result = np.rot90(arr, k=k, axes=(0, 1))
    transformed = boxes
    for _ in range(k):
        transformed = tuple(Box(b.class_id, b.class_name,
                                (b.xyxy[1], w - b.xyxy[2], b.xyxy[3], w - b.xyxy[0])) for b in transformed)
        h, w = w, h
    if flip:
        result = result[:, ::-1, :]
        transformed = tuple(Box(b.class_id, b.class_name,
                                (w - b.xyxy[2], b.xyxy[1], w - b.xyxy[0], b.xyxy[3])) for b in transformed)
    config = load_preprocessing_config()
    meta = {"recipe_id": recipe_id, "seed": seed, "partition": partition, "rotation_ccw": 90 * k,
            "horizontal_flip": flip, "orig_hw": list(arr.shape[:2]), "out_hw": [h, w],
            "color_order": "rgb", "normalized": False,
            "preprocessing_version": config["preprocessing_version"], "preprocessing_hash": preprocessing_hash(config)}
    return np.ascontiguousarray(result).copy(), transformed, meta
