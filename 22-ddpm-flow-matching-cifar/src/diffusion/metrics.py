"""Sample quality metrics: MMD and energy distance for toys, FID and KID for images."""
from __future__ import annotations

import numpy as np
import torch
from scipy import linalg


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidths=(0.1, 0.2, 0.5, 1.0, 2.0)) -> float:
    """Unbiased MMD^2 with a sum of RBF kernels exp(-|a-b|^2 / (2 h^2))."""
    x, y = x.double(), y.double()
    xx, yy, xy = torch.cdist(x, x) ** 2, torch.cdist(y, y) ** 2, torch.cdist(x, y) ** 2
    kxx = sum(torch.exp(-xx / (2 * h * h)) for h in bandwidths)
    kyy = sum(torch.exp(-yy / (2 * h * h)) for h in bandwidths)
    kxy = sum(torch.exp(-xy / (2 * h * h)) for h in bandwidths)
    n, m = x.shape[0], y.shape[0]
    sxx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    syy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    return float(sxx + syy - 2 * kxy.mean())


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """2 E|X-Y| - E|X-X'| - E|Y-Y'| (V-statistic)."""
    x, y = x.double(), y.double()
    return float(2 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean())


def fid_from_feats(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid_from_feats(a: np.ndarray, b: np.ndarray, subsets: int = 50, subset_size: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2 with the cubic polynomial kernel, averaged over random subsets."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(subset_size, a.shape[0], b.shape[0])
    vals = []
    for _ in range(subsets):
        x = a[rng.choice(a.shape[0], m, replace=False)].astype(np.float64)
        y = b[rng.choice(b.shape[0], m, replace=False)].astype(np.float64)
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


class InceptionFeatures:
    """2048-d pool features from torch-fidelity's FID-compatible InceptionV3."""

    def __init__(self, device: str = "cpu"):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        self.net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs_uint8: torch.Tensor, batch: int = 250) -> np.ndarray:
        out = []
        for i in range(0, imgs_uint8.shape[0], batch):
            f = self.net(imgs_uint8[i:i + batch].to(self.device))[0]
            out.append(f.double().cpu().numpy())
        return np.concatenate(out)
