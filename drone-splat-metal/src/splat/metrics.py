"""PSNR and SSIM (Gaussian window 11, sigma 1.5, the variant used by 3DGS and Mip-NeRF 360)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.clamp(0, 1) - b) ** 2).item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def _window(size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    x = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: [H, W, 3] in [0, 1]. Returns mean SSIM (differentiable scalar)."""
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    w = _window(device=x.device, dtype=x.dtype)
    pad = w.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    sxx = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x**2
    syy = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_x * mu_y
    C1, C2 = 0.01**2, 0.03**2
    m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sxx + syy + C2))
    return m.mean()
