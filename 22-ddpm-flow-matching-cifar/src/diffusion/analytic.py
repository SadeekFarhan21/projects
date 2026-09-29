"""Exact denoiser and exact probability flow map for Gaussian data N(mu, Sigma).

For x_tilde = x0 + sigma * eps with x0 ~ N(mu, Sigma):
  D(x, sigma) = E[x0 | x] = mu + Sigma (Sigma + sigma^2 I)^-1 (x - mu)
In the eigenbasis Sigma = U diag(lam) U^T the probability flow ODE decouples and
  (x(sigma) - mu)_i = (x(sigma_max) - mu)_i * sqrt(lam_i + sigma^2) / sqrt(lam_i + sigma_max^2)
which gives an exact reference endpoint for any starting point.
"""
from __future__ import annotations

import torch


class GaussianData:
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        self.mu = mu
        self.cov = cov
        lam, u = torch.linalg.eigh(cov)
        self.lam, self.u = lam.clamp(min=0), u

    @classmethod
    def random(cls, dim: int, seed: int = 0, lam_min: float = 0.01, lam_max: float = 4.0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype))
        lam = torch.logspace(torch.log10(torch.tensor(lam_min)).item(), torch.log10(torch.tensor(lam_max)).item(), dim, dtype=dtype)
        mu = torch.randn(dim, generator=g, dtype=dtype)
        return cls(mu, q @ torch.diag(lam) @ q.T)

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, y=None) -> torch.Tensor:
        s2 = sigma.to(x.dtype).view(-1, 1) ** 2
        z = (x - self.mu) @ self.u  # eigen coordinates
        z = z * (self.lam / (self.lam + s2))
        return self.mu + z @ self.u.T

    def marginal_sample(self, n: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
        """Exact sample of x_tilde at noise level sigma."""
        z = torch.randn(n, self.mu.numel(), generator=generator, dtype=self.mu.dtype)
        return self.mu + (z * torch.sqrt(self.lam + sigma**2)) @ self.u.T

    def flow_to_zero(self, x: torch.Tensor, sigma_max: float) -> torch.Tensor:
        z = (x - self.mu) @ self.u
        z = z * torch.sqrt(self.lam / (self.lam + sigma_max**2))
        return self.mu + z @ self.u.T

    def data_sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        return self.marginal_sample(n, 0.0, generator)


def gaussian_w2(x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> float:
    """2-Wasserstein distance between N(mean(x), cov(x)) and N(mu, cov) (Bures formula)."""
    m = x.mean(0)
    c = torch.cov(x.T)
    def sqrtm(a):
        lam, u = torch.linalg.eigh(a)
        return u @ torch.diag(lam.clamp(min=0).sqrt()) @ u.T
    rc = sqrtm(cov)
    cross = sqrtm(rc @ c @ rc)
    w2 = (m - mu).pow(2).sum() + torch.trace(c + cov - 2 * cross)
    return float(w2.clamp(min=0).sqrt())
