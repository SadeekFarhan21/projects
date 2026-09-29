"""Datasets: 2D toys and CIFAR-10 held entirely on the training device."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------- 2D toys

def gaussian_mixture(n: int, generator: torch.Generator | None = None, k: int = 8, radius: float = 2.0,
                     std: float = 0.15) -> torch.Tensor:
    """k Gaussians evenly spaced on a circle, rescaled to roughly unit variance."""
    idx = torch.randint(0, k, (n,), generator=generator)
    ang = 2 * math.pi * idx.float() / k
    centers = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1) * radius
    x = centers + std * torch.randn(n, 2, generator=generator)
    return x / math.sqrt(radius**2 / 2 + std**2)


def two_moons(n: int, generator: torch.Generator | None = None, noise: float = 0.08) -> torch.Tensor:
    """Two interleaving half circles (same construction as sklearn), standardised."""
    n1 = n // 2
    n2 = n - n1
    a1 = math.pi * torch.rand(n1, generator=generator)
    a2 = math.pi * torch.rand(n2, generator=generator)
    upper = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
    lower = torch.stack([1 - torch.cos(a2), 0.5 - torch.sin(a2)], dim=1)
    x = torch.cat([upper, lower])
    x = x + noise * torch.randn(x.shape, generator=generator)
    x = x[torch.randperm(n, generator=generator)]
    # fixed standardisation constants (population mean and std of the noiseless moons)
    return (x - torch.tensor([0.5, 0.25])) / torch.tensor([0.87, 0.5])


TOYS = {"gmm8": gaussian_mixture, "moons": two_moons}


# ---------------------------------------------------------------- CIFAR-10

def load_cifar10(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "cifar10.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: uv run python scripts/download_cifar10.py")
    d = np.load(path)
    return d[f"x_{split}"], d[f"y_{split}"]


class DeviceLoader:
    """Keeps all images on the device as uint8 and yields random batches in [-1, 1] with flips."""

    def __init__(self, x_uint8: np.ndarray, y: np.ndarray, batch: int, device, seed: int = 0, flip: bool = True):
        self.x = torch.from_numpy(x_uint8).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.batch = batch
        self.flip = flip
        self.g = torch.Generator().manual_seed(seed)
        self.device = device

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.x.shape[0], (self.batch,), generator=self.g).to(self.device)
        x = self.x[idx].float() / 127.5 - 1.0
        if self.flip:
            m = (torch.rand(self.batch, generator=self.g) < 0.5).to(self.device)
            x = torch.where(m[:, None, None, None], x.flip(3), x)
        return x, self.y[idx]


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
