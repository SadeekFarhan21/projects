"""Sampler error with the exact denoiser of a Gaussian data distribution.

Data is N(mu, Sigma) in 64 dimensions with eigenvalues log spaced from 1e-3 to
1. The denoiser is exact, so every error here is discretisation error of the
sampler plus its grid. Two numbers per (sampler, NFE):
  w2        : Bures 2-Wasserstein between the sample Gaussian fit and the target
  path_rmse : RMS distance to the exact probability flow endpoint from the same
              start point (deterministic samplers only)
Writes results/analytic_gaussian.csv and results/analytic_gaussian.png.

  uv run python scripts/analytic_gaussian.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion.analytic import GaussianData, gaussian_w2
from diffusion.process import SIGMA_MAX, Linear, VPCosine
from diffusion.samplers import SAMPLER_FNS, karras_sigmas, native_sigmas, nfe_of, steps_for_nfe

RES = Path(__file__).resolve().parents[1] / "results"
NFES = (4, 8, 16, 32, 64, 128, 256)
DIM, N = 64, 20_000

CONFIGS = [  # (label, sampler, grid)
    ("DDPM, cosine grid", "ddpm", "vp"),
    ("DDIM, cosine grid", "ddim", "vp"),
    ("DDIM, linear t grid", "ddim", "linear"),
    ("Euler, Karras grid", "euler", "karras"),
    ("Heun, Karras grid", "heun", "karras"),
]


def grid(kind: str, steps: int) -> torch.Tensor:
    if kind == "karras":
        return karras_sigmas(steps)
    return native_sigmas(steps, VPCosine() if kind == "vp" else Linear())


def main():
    torch.set_num_threads(4)
    RES.mkdir(exist_ok=True)
    data = GaussianData.random(DIM, seed=0, lam_min=1e-3, lam_max=1.0)
    g = torch.Generator().manual_seed(0)
    x = data.marginal_sample(N, SIGMA_MAX, g)
    exact = data.flow_to_zero(x, SIGMA_MAX)
    floor = gaussian_w2(data.data_sample(N, torch.Generator().manual_seed(1)), data.mu, data.cov)
    rows = [dict(config="exact samples", sampler="none", nfe=0, w2=floor, path_rmse=0.0)]
    for label, smp, kind in CONFIGS:
        for nfe in NFES:
            steps = steps_for_nfe(smp, nfe)
            out = SAMPLER_FNS[smp](data.denoise, x, grid(kind, steps), generator=torch.Generator().manual_seed(2))
            r = dict(config=label, sampler=smp, nfe=nfe_of(smp, steps), w2=gaussian_w2(out, data.mu, data.cov),
                     path_rmse=(out - exact).pow(2).sum(1).mean().sqrt().item() if smp != "ddpm" else float("nan"))
            rows.append(r)
            print(r, flush=True)
    with open(RES / "analytic_gaussian.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, _, _ in CONFIGS:
        pts = [(r["nfe"], r["w2"]) for r in rows if r["config"] == label]
        ax.plot(*zip(*pts), marker="o", label=label)
    ax.axhline(floor, color="gray", ls="--", label="exact samples (noise floor)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("W2 to the true Gaussian")
    ax.set_title("Sampler error with the exact Gaussian denoiser")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "analytic_gaussian.png", dpi=130)


if __name__ == "__main__":
    main()
