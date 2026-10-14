"""EfficientAD-S: PDN teacher/student and global feature autoencoder.

Architecture reference: EfficientAD (arXiv:2303.14535), anomalib v2.0.0
torch_model.py (Apache-2.0). This standalone recipe uses PCB-only losses;
it deliberately does not claim reproduction of ImageNette-regularized results.
"""
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

import torch
from torch import nn
from torch.nn import functional as F

from pcb_lab.data.manifest import sha256_file

TEACHER_URL = ("https://github.com/open-edge-platform/anomalib/releases/download/"
               "efficientad_pretrained_weights/efficientad_pretrained_weights.zip")
TEACHER_ARCHIVE_SHA256 = "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a"
CHANNELS = 384


def deterministic_bilinear(x, size):
    """align_corners=False interpolation using deterministic index_select backward.

    PyTorch 2.6 native CUDA bilinear backward rejects strict deterministic mode.
    Separable interpolation avoids that kernel while preserving bilinear geometry.
    """
    height, width = (size, size) if isinstance(size, int) else size
    for axis, target in ((-1, width), (-2, height)):
        source = x.shape[axis]
        positions = ((torch.arange(target, device=x.device, dtype=torch.float64) + .5)
                     * (source / target) - .5).clamp_min(0)
        lower = positions.floor().long().clamp_max(source - 1)
        upper = (lower + 1).clamp_max(source - 1)
        shape = [1] * x.ndim
        shape[axis] = target
        weight = (positions - lower).to(x.dtype).view(shape)
        x = x.index_select(axis, lower) * (1 - weight) + x.index_select(axis, upper) * weight
    return x


def _resize(x, size):
    if x.is_cuda:
        return deterministic_bilinear(x, size)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


class SmallPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels=CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 4)
        self.conv2 = nn.Conv2d(128, 256, 4)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.conv4 = nn.Conv2d(256, out_channels, 4)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        return self.conv4(F.relu(self.conv3(x)))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        previous = 3
        for channels in (32, 32, 64, 64, 64):
            layers.extend((nn.Conv2d(previous, channels, 4, 2, 1), nn.ReLU()))
            previous = channels
        layers.append(nn.Conv2d(64, 64, 8))
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.ModuleList([nn.Conv2d(64, 64, 4, 1, 2) for _ in range(6)])
        self.final = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(64, CHANNELS, 3, 1, 1))

    def forward(self, x):
        x = self.encoder(x)
        for layer, size in zip(self.decoder, (3, 8, 15, 32, 63, 127)):
            x = _resize(x, size)
            x = F.dropout(F.relu(layer(x)), p=0.2, training=self.training)
        return self.final(_resize(x, 56))


class EfficientAdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = SmallPatchDescriptionNetwork()
        self.student = SmallPatchDescriptionNetwork(2 * CHANNELS)
        self.autoencoder = Autoencoder()
        self.teacher.requires_grad_(False).eval()
        self.register_buffer("teacher_mean", torch.zeros(1, CHANNELS, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, CHANNELS, 1, 1))
        self.register_buffer("map_quantiles", torch.tensor([[0., 1.], [0., 1.]]))
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def normalized_input(self, pixels):
        # Caller passes float RGB in [0,1]. Exactly one ImageNet normalization.
        return (pixels - self.rgb_mean) / self.rgb_std

    def teacher_features(self, pixels):
        return self.teacher(self.normalized_input(pixels))

    def _features(self, pixels):
        x = self.normalized_input(pixels)
        with torch.no_grad():
            teacher = (self.teacher(x) - self.teacher_mean) / self.teacher_std
        return teacher, self.student(x), self.autoencoder(x)

    def loss(self, pixels):
        teacher, student, ae = self._features(pixels)
        distance = (teacher - student[:, :CHANNELS]).square()
        hard = distance[distance >= torch.quantile(distance.detach(), .999)].mean()
        reconstruction = (teacher - ae).square().mean()
        student_ae = (ae - student[:, CHANNELS:]).square().mean()
        return hard + reconstruction + student_ae

    def raw_maps(self, pixels):
        teacher, student, ae = self._features(pixels)
        return torch.cat(((teacher - student[:, :CHANNELS]).square().mean(1, keepdim=True),
                          (ae - student[:, CHANNELS:]).square().mean(1, keepdim=True)), dim=1)

    def forward(self, pixels):
        maps = self.raw_maps(pixels)
        low = self.map_quantiles[:, 0].view(1, 2, 1, 1)
        scale = (self.map_quantiles[:, 1] - self.map_quantiles[:, 0]).clamp_min(1e-12).view(1, 2, 1, 1)
        maps = (0.1 * (maps - low) / scale).mean(1, keepdim=True)
        # PDN has a 33px receptive field; pad the 56px output before upsampling.
        maps = F.pad(maps, (4, 4, 4, 4))
        return F.interpolate(maps, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]


def load_teacher(model, path, *, allow_download=False):
    path = Path(path)
    downloaded = False
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(f"Pretrained teacher missing: {path}; use --allow-download explicitly")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            archive = Path(temporary) / "teacher.zip"
            with urllib.request.urlopen(TEACHER_URL, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256_file(archive) != TEACHER_ARCHIVE_SHA256:
                raise ValueError("Pretrained teacher archive SHA256 mismatch")
            with zipfile.ZipFile(archive) as bundle:
                members = [n for n in bundle.namelist() if Path(n).name == "pretrained_teacher_small.pth"]
                if len(members) != 1:
                    raise ValueError("Teacher archive must contain one small checkpoint")
                payload = bundle.read(members[0])
            temporary_weight = Path(temporary) / "small.pt"
            temporary_weight.write_bytes(payload)
            temporary_weight.replace(path)
        downloaded = True
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    # Accommodate the reference Sequential PDN as well as anomalib conv keys.
    if "0.weight" in weights:
        weights = {f"conv{j}.{part}": weights[f"{i}.{part}"]
                   for j, i in enumerate((0, 3, 6, 8), 1) for part in ("weight", "bias")}
    model.teacher.load_state_dict(weights, strict=True)
    model.teacher.requires_grad_(False).eval()
    return {"path": str(path), "sha256": sha256_file(path), "downloaded": downloaded,
            "source": TEACHER_URL if downloaded else "local_pretrained_checkpoint",
            "archive_sha256": TEACHER_ARCHIVE_SHA256 if downloaded else None}
