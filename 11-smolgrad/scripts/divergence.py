"""How fast do smolgrad and a PyTorch twin drift apart when trained from the same
weights on the same batches? Measures max |w_smolgrad - w_torch| per step for
{float32, float64} x {SGD+momentum, AdamW} on the MNIST MLP.

Usage: uv run python scripts/divergence.py [--steps 300]
Writes results/divergence.csv and results/divergence.png.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

import smolgrad as sg
import smolgrad.functional as F
from datasets import RESULTS, load_mnist
from train_mnist import build


def run(dtype: str, opt_name: str, steps: int, xtr, ytr):
    npdt = np.dtype(dtype)
    tdt = getattr(torch, dtype)
    sg.manual_seed(0)
    model = build(dtype=npdt)
    tlayers = []
    for layer in model.layers:
        if hasattr(layer, "weight"):
            tl = torch.nn.Linear(layer.weight.shape[1], layer.weight.shape[0], dtype=tdt)
            with torch.no_grad():
                tl.weight.copy_(torch.from_numpy(layer.weight.data))
                tl.bias.copy_(torch.from_numpy(layer.bias.data))
            tlayers.append(tl)
        else:
            tlayers.append(torch.nn.ReLU())
    tmodel = torch.nn.Sequential(*tlayers)
    if opt_name == "sgd_momentum":
        opt = sg.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        topt = torch.optim.SGD(tmodel.parameters(), lr=0.05, momentum=0.9)
    else:
        opt = sg.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        topt = torch.optim.AdamW(tmodel.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = np.random.default_rng(0)
    out = []
    for step in range(1, steps + 1):
        b = rng.integers(0, len(xtr), size=128)
        xb, yb = xtr[b].astype(npdt), ytr[b]
        opt.zero_grad()
        loss = F.cross_entropy(model(sg.Tensor(xb)), yb)
        loss.backward()
        opt.step()
        topt.zero_grad()
        tl = torch.nn.functional.cross_entropy(tmodel(torch.from_numpy(xb)), torch.from_numpy(yb))
        tl.backward()
        topt.step()
        diff = max(float(np.max(np.abs(p.data - tp.detach().numpy())))
                   for p, tp in zip(model.parameters(), tmodel.parameters()))
        out.append(dict(dtype=dtype, optimizer=opt_name, step=step, max_abs_weight_diff=diff,
                        loss_diff=abs(loss.item() - tl.item())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    xtr, ytr, _, _ = load_mnist()
    rows = []
    for dtype in ("float32", "float64"):
        for opt_name in ("sgd_momentum", "adamw"):
            r = run(dtype, opt_name, args.steps, xtr, ytr)
            rows += r
            marks = {s: r[s - 1]["max_abs_weight_diff"] for s in (1, 10, 100, args.steps)}
            print(dtype, opt_name, " ".join(f"step{s}={v:.3e}" for s, v in marks.items()), flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "divergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for dtype, ls in (("float32", "-"), ("float64", "--")):
        for opt_name in ("sgd_momentum", "adamw"):
            r = [x for x in rows if x["dtype"] == dtype and x["optimizer"] == opt_name]
            ax.semilogy([x["step"] for x in r], [max(x["max_abs_weight_diff"], 1e-17) for x in r], ls,
                        label=f"{opt_name}, {dtype}")
    ax.set_xlabel("training step")
    ax.set_ylabel("max |w_smolgrad - w_torch|")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Twin runs from identical init (source: results/divergence.csv)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "divergence.png", dpi=150)


if __name__ == "__main__":
    main()
