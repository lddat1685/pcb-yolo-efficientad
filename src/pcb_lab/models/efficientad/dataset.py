"""Normal-only PCB training, immutable manifest membership, no test pixels."""
import numpy as np
import torch

from pcb_lab.data.samples import AdTrainSet, EvalSet
from pcb_lab.inference.preprocessing import prepare_input
from pcb_lab.inference.tiling import TileManager


class DataLeakError(ValueError):
    """A sample violates the normal-only training/statistics boundary."""


def require_good(sample, split):
    if sample.partition != split or sample.is_defect or sample.boxes:
        raise DataLeakError(f"{sample.sample_id}: expected {split} good with no boxes")


class EfficientAdDataset:
    def __init__(self, dataset_root, split="train", limit=None):
        if split not in ("train", "calibration", "fusion"):
            raise PermissionError(f"EfficientAD cannot access partition {split!r}")
        self.split = split
        self.tile_manager = TileManager(256, 224, "reflect", 0.45)
        self.loader = AdTrainSet(dataset_root) if split == "train" else EvalSet(dataset_root, split)
        samples = tuple(self.loader)
        if split == "train":
            for sample in samples:
                require_good(sample, "train")
        if limit is not None:
            if type(limit) is int and limit > 0:
                samples = samples[:limit]
            elif isinstance(limit, dict) and set(limit) <= {"good", "defect"} and all(
                    type(v) is int and v >= 0 for v in limit.values()):
                used = {"good": 0, "defect": 0}
                selected = []
                for sample in samples:
                    kind = "defect" if sample.is_defect else "good"
                    if kind in limit and used[kind] < limit[kind]:
                        selected.append(sample)
                        used[kind] += 1
                samples = tuple(selected)
            else:
                raise ValueError("limit must be a positive integer or good/defect count mapping")
        self.samples = tuple(samples)
        self.num_tiles = sum(len(self.tile_manager.plan(s.height, s.width)) for s in self.samples)

    def __len__(self):
        return len(self.samples)

    def tiles_uint8(self, index):
        sample = self.samples[index]
        if self.split == "train":
            require_good(sample, "train")
        if sample.partition != self.split:
            raise DataLeakError("Sample partition differs from dataset")
        return prepare_input(sample, "ad_tile").pixels

    def __getitem__(self, index):
        pixels = self.tiles_uint8(index)
        return torch.from_numpy(np.ascontiguousarray(pixels.transpose(0, 3, 1, 2))).float().div_(255)

    def iter_tiles(self):
        for index, sample in enumerate(self.samples):
            for tile in self[index]:
                yield sample, tile.unsqueeze(0)
