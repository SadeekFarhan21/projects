"""2D toys: DDPM (eps prediction, VP cosine) vs conditional flow matching (rectified flow).

Trains a small MLP per (dataset, objective) on CPU, then samples with every
sampler at several NFE budgets and scores against fresh data with MMD and
energy distance. Writes results/toy2d.csv, results/toy2d_mmd.png and
results/toy2d_samples.png.

  uv run python scripts/toy2d.py            # default: 6000 steps per model
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion.data import TOYS
from diffusion.ema import EMA
from diffusion.metrics import energy_distance, mmd_rbf
from diffusion.models import ToyMLP
from diffusion.objectives import Denoiser, diffusion_loss
from diffusion.process import SIGMA_MAX
from diffusion.samplers import SAMPLERS, nfe_of, sample, steps_for_nfe

RES = Path(__file__).resolve().parents[1] / "results"
MODELS = {"eps": "DDPM", "rf": "CFM"}
NFES = (4, 8, 16, 32, 64)


def train(dataset: str, objective: str, steps: int, seed: int = 0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = ToyMLP()
    ema = EMA(net, 0.999)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0 = TOYS[dataset](512, g)
        loss = diffusion_loss(net, objective, x0, None, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        ema.update(net)
    return ema.model, loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n", type=int, default=2000, help="samples per evaluation")
    args = ap.parse_args()
    torch.set_num_threads(4)
    RES.mkdir(exist_ok=True)
    rows, keep = [], {}
    for ds in TOYS:
        ref = TOYS[ds](args.n, torch.Generator().manual_seed(1000))
        floor_data = TOYS[ds](args.n, torch.Generator().manual_seed(2000))
        floor = {"mmd": mmd_rbf(floor_data, ref), "energy": energy_distance(floor_data, ref)}
        rows.append(dict(dataset=ds, model="data", sampler="none", nfe=0, **floor))
        for obj, mname in MODELS.items():
            t0 = time.time()
            net, last = train(ds, obj, args.steps)
            print(f"{ds} {mname}: trained in {time.time() - t0:.0f}s, last loss {last:.3f}", flush=True)
            d = Denoiser(net, obj)
            for smp in SAMPLERS:
                for nfe in NFES:
                    g = torch.Generator().manual_seed(7)
                    x = torch.randn(args.n, 2, generator=g) * SIGMA_MAX
                    out = sample(smp, d, x, nfe, d.process, generator=g)
                    r = dict(dataset=ds, model=mname, sampler=smp, nfe=nfe_of(smp, steps_for_nfe(smp, nfe)),
                             mmd=mmd_rbf(out, ref), energy=energy_distance(out, ref))
                    rows.append(r)
                    keep[(ds, mname, smp, nfe)] = out
                    print(r, flush=True)
    with open(RES / "toy2d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # MMD vs NFE, one panel per dataset, four curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    curves = [("DDPM", "ddpm"), ("DDPM", "heun"), ("CFM", "euler"), ("CFM", "heun")]
    for ax, ds in zip(axes, TOYS):
        for m, s in curves:
            pts = [(r["nfe"], max(r["mmd"], 1e-5)) for r in rows if r["dataset"] == ds and r["model"] == m and r["sampler"] == s]
            ax.plot(*zip(*pts), marker="o", label=f"{m} model, {s} sampler")
        fl = [r["mmd"] for r in rows if r["dataset"] == ds and r["model"] == "data"][0]
        ax.axhline(max(abs(fl), 1e-5), color="gray", ls="--", label="data vs data")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("MMD squared (lower is better)")
        ax.set_title({"gmm8": "Eight Gaussian mixture", "moons": "Two moons"}[ds])
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "toy2d_mmd.png", dpi=130)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ds in enumerate(TOYS):
        for j, (m, s, nfe) in enumerate([("DDPM", "ddpm", 8), ("DDPM", "ddpm", 64), ("CFM", "euler", 8), ("CFM", "euler", 64)]):
            x = keep[(ds, m, s, nfe)]
            ax = axes[i, j]
            ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.5)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{m} with {s}, {nfe} NFE", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "toy2d_samples.png", dpi=130)


if __name__ == "__main__":
    main()
