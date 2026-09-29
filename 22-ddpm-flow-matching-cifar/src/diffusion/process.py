"""Forward (noising) processes and the mapping into the shared EDM sigma space.

Every process here writes a noisy sample as

    x_t = alpha(t) * x0 + s(t) * eps,   eps ~ N(0, I),  t in (0, 1)

Dividing by alpha gives the variance exploding form used by all samplers:

    x_tilde = x_t / alpha = x0 + sigma * eps,   sigma = s(t) / alpha(t)

so a network trained on any of the processes can be wrapped as an EDM style
denoiser D(x_tilde, sigma) ~= E[x0 | x_tilde] and sampled with the same code.

Two processes:
  * VPCosine: alpha = cos(pi t / 2), s = sin(pi t / 2)  (continuous cosine
    schedule, alpha^2 + s^2 = 1). sigma = tan(pi t / 2).
  * Linear (rectified flow): alpha = 1 - t, s = t. sigma = t / (1 - t).
"""
from __future__ import annotations

import math

import torch

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0


class Process:
    name: str

    def alpha(self, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def s(self, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.s(t) / self.alpha(t)

    def t_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def t_range(self, sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX) -> tuple[float, float]:
        lo = self.t_of_sigma(torch.tensor(sigma_min, dtype=torch.float64)).item()
        hi = self.t_of_sigma(torch.tensor(sigma_max, dtype=torch.float64)).item()
        return lo, hi

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Closed form x_t given x0, per sample t of shape [B]."""
        shape = (-1,) + (1,) * (x0.dim() - 1)
        return self.alpha(t).view(shape) * x0 + self.s(t).view(shape) * eps


class VPCosine(Process):
    name = "vp_cosine"

    def alpha(self, t):
        return torch.cos(0.5 * math.pi * t)

    def s(self, t):
        return torch.sin(0.5 * math.pi * t)

    def t_of_sigma(self, sigma):
        return torch.atan(sigma) * (2.0 / math.pi)

    def discrete_betas(self, n: int) -> torch.Tensor:
        """DDPM style betas on the grid t_i = i / n so that prod(1 - beta) = alpha(t_i)^2."""
        t = torch.arange(n + 1, dtype=torch.float64) / n
        abar = self.alpha(t) ** 2
        abar = abar.clamp(min=1e-12)
        return (1.0 - abar[1:] / abar[:-1]).clamp(max=0.999)


class Linear(Process):
    """Rectified flow / conditional flow matching path with sigma_min = 0."""

    name = "linear"

    def alpha(self, t):
        return 1.0 - t

    def s(self, t):
        return t

    def t_of_sigma(self, sigma):
        return sigma / (1.0 + sigma)


def process_for(objective: str) -> Process:
    if objective in ("eps", "v"):
        return VPCosine()
    if objective == "rf":
        return Linear()
    raise ValueError(f"unknown objective {objective!r}")
