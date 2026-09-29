"""Samplers in the EDM sigma space.

All samplers take
  denoise : callable (x, sigma[B], y) -> x0_hat
  x       : initial sample at sigmas[0] (the caller draws it, which keeps
            seeding in one place)
  sigmas  : strictly decreasing grid ending in 0
and return the final sample. The number of denoiser calls (NFE) is

  ddpm, ddim, euler : len(sigmas) - 1
  heun              : 2 * (len(sigmas) - 1) - 1   (no correction on the last step to 0)

DDIM with eta = 0 is algebraically the same update as Euler on the probability
flow ODE dx/dsigma = (x - D(x, sigma)) / sigma. The two differ only in the grid
we pair them with: DDIM and DDPM use a grid uniform in the model's native time
t, Euler and Heun use the Karras rho = 7 grid, which is the usual EDM setup.
"""
from __future__ import annotations

from typing import Callable

import torch

from .process import SIGMA_MAX, SIGMA_MIN, Process

DenoiseFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]

SAMPLERS = ("ddpm", "ddim", "euler", "heun")


# ---------------------------------------------------------------- grids

def karras_sigmas(n: int, sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX, rho: float = 7.0) -> torch.Tensor:
    """n noise levels from sigma_max to sigma_min, then a trailing 0 (length n + 1)."""
    if n == 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float64)
    i = torch.arange(n, dtype=torch.float64)
    inv = 1.0 / rho
    s = (sigma_max**inv + i / (n - 1) * (sigma_min**inv - sigma_max**inv)) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64)])


def native_sigmas(n: int, process: Process, sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX) -> torch.Tensor:
    """n levels uniform in the process's native time t, then a trailing 0."""
    lo, hi = process.t_range(sigma_min, sigma_max)
    t = torch.linspace(hi, lo, n, dtype=torch.float64) if n > 1 else torch.tensor([hi], dtype=torch.float64)
    return torch.cat([process.sigma(t), torch.zeros(1, dtype=torch.float64)])


def steps_for_nfe(sampler: str, nfe: int) -> int:
    """Number of grid steps so the sampler uses at most `nfe` denoiser calls."""
    if sampler == "heun":
        return max(1, (nfe + 1) // 2)
    return nfe


def nfe_of(sampler: str, steps: int) -> int:
    return 2 * steps - 1 if sampler == "heun" else steps


def grid_for(sampler: str, steps: int, process: Process) -> torch.Tensor:
    if sampler in ("ddpm", "ddim"):
        return native_sigmas(steps, process)
    return karras_sigmas(steps)


# ---------------------------------------------------------------- samplers

def _sig(sigma: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    return torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)


@torch.no_grad()
def ddim(denoise: DenoiseFn, x: torch.Tensor, sigmas: torch.Tensor, y=None, eta: float = 0.0,
         generator: torch.Generator | None = None) -> torch.Tensor:
    """DDIM in variance exploding form. eta = 1 is DDPM ancestral sampling."""
    sig = [float(v) for v in sigmas]
    for i in range(len(sig) - 1):
        st, ss = sig[i], sig[i + 1]
        x0 = denoise(x, _sig(st, x), y)
        if ss == 0.0:
            return x0
        eps = (x - x0) / st
        c = eta * (ss * ss * (st * st - ss * ss) / (st * st)) ** 0.5
        x = x0 + (ss * ss - c * c) ** 0.5 * eps
        if c > 0:
            x = x + c * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return x


def ddpm(denoise: DenoiseFn, x, sigmas, y=None, generator=None):
    """DDPM ancestral: sample the Gaussian posterior q(x_s | x_t, x0 = D(x_t))."""
    return ddim(denoise, x, sigmas, y, eta=1.0, generator=generator)


@torch.no_grad()
def euler(denoise: DenoiseFn, x, sigmas, y=None, generator=None):
    sig = [float(v) for v in sigmas]
    for i in range(len(sig) - 1):
        st, ss = sig[i], sig[i + 1]
        x0 = denoise(x, _sig(st, x), y)
        d = (x - x0) / st
        x = x + (ss - st) * d
    return x


@torch.no_grad()
def heun(denoise: DenoiseFn, x, sigmas, y=None, generator=None):
    """EDM Algorithm 1 (deterministic, no churn)."""
    sig = [float(v) for v in sigmas]
    for i in range(len(sig) - 1):
        st, ss = sig[i], sig[i + 1]
        d = (x - denoise(x, _sig(st, x), y)) / st
        x_next = x + (ss - st) * d
        if ss > 0.0:
            d2 = (x_next - denoise(x_next, _sig(ss, x), y)) / ss
            x_next = x + (ss - st) * 0.5 * (d + d2)
        x = x_next
    return x


SAMPLER_FNS = {"ddpm": ddpm, "ddim": ddim, "euler": euler, "heun": heun}


def sample(sampler: str, denoise: DenoiseFn, x_init: torch.Tensor, nfe: int, process: Process, y=None,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Run `sampler` with a budget of `nfe` calls, starting from x_init at sigma_max."""
    steps = steps_for_nfe(sampler, nfe)
    sigmas = grid_for(sampler, steps, process)
    return SAMPLER_FNS[sampler](denoise, x_init, sigmas, y=y, generator=generator)
