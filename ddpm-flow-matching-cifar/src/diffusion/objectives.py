"""Training objectives and the network to denoiser conversion.

A network is always called in its native space as net(x_t, t, y) where t is in
(0, 1). What it predicts depends on the objective:

  eps : the noise eps                                  (VP cosine process)
  v   : v = alpha * eps - s * x0                        (VP cosine process)
  rf  : velocity u = dx_t/dt = eps - x0                 (linear process)

`Denoiser` turns any of these into x0_hat(x_tilde, sigma) in EDM sigma space.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .process import SIGMA_MAX, SIGMA_MIN, Process, process_for

OBJECTIVES = ("eps", "v", "rf")


def target(objective: str, x0: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if objective == "eps":
        return eps
    if objective == "v":
        return alpha * eps - s * x0
    if objective == "rf":
        return eps - x0
    raise ValueError(objective)


def x0_from_output(objective: str, out: torch.Tensor, x_t: torch.Tensor, alpha: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if objective == "eps":
        return (x_t - s * out) / alpha
    if objective == "v":
        # alpha^2 + s^2 = 1 for VP:  x0 = alpha * x_t - s * v
        return alpha * x_t - s * out
    if objective == "rf":
        # x_t = (1 - t) x0 + t eps,  u = eps - x0  =>  x0 = x_t - t u
        return x_t - t * out
    raise ValueError(objective)


def sample_t(n: int, process: Process, device, generator: torch.Generator | None = None) -> torch.Tensor:
    lo, hi = process.t_range(SIGMA_MIN, SIGMA_MAX)
    u = torch.rand(n, device=device, generator=generator)
    return lo + (hi - lo) * u


def diffusion_loss(
    net: nn.Module,
    objective: str,
    x0: torch.Tensor,
    y: torch.Tensor | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Plain MSE between the network output and the objective's target."""
    process = process_for(objective)
    b = x0.shape[0]
    t = sample_t(b, process, x0.device, generator)
    eps = torch.randn(x0.shape, device=x0.device, generator=generator)
    shape = (-1,) + (1,) * (x0.dim() - 1)
    alpha = process.alpha(t).view(shape)
    s = process.s(t).view(shape)
    x_t = alpha * x0 + s * eps
    out = net(x_t, t, y)
    return (out - target(objective, x0, eps, alpha, s)).pow(2).mean()


class Denoiser:
    """EDM style denoiser D(x_tilde, sigma, y) built from a native network.

    cfg_scale w applies classifier free guidance: D_u + w * (D_c - D_u), with the
    unconditional branch using label `null_label`. w = 1 is plain conditional.
    """

    def __init__(self, net: nn.Module, objective: str, null_label: int | None = None, cfg_scale: float = 1.0):
        self.net = net
        self.objective = objective
        self.process = process_for(objective)
        self.null_label = null_label
        self.cfg_scale = cfg_scale
        self.nfe = 0

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        self.nfe += 1
        b = x.shape[0]
        if sigma.dim() == 0:
            sigma = sigma.expand(b)
        sigma = sigma.to(x.dtype)
        t = self.process.t_of_sigma(sigma)
        shape = (-1,) + (1,) * (x.dim() - 1)
        alpha = self.process.alpha(t).view(shape)
        s = self.process.s(t).view(shape)
        x_t = alpha * x
        guided = y is not None and self.null_label is not None and self.cfg_scale != 1.0
        if guided:
            y_null = torch.full_like(y, self.null_label)
            out = self.net(torch.cat([x_t, x_t]), torch.cat([t, t]), torch.cat([y, y_null]))
            out_c, out_u = out.chunk(2)
            d_c = x0_from_output(self.objective, out_c, x_t, alpha, s, t.view(shape))
            d_u = x0_from_output(self.objective, out_u, x_t, alpha, s, t.view(shape))
            return d_u + self.cfg_scale * (d_c - d_u)
        out = self.net(x_t, t, y)
        return x0_from_output(self.objective, out, x_t, alpha, s, t.view(shape))
