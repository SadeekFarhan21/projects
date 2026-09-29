"""Train an MLP on MNIST with smolgrad, optionally alongside a PyTorch twin that
starts from identical weights and sees identical batches.

Usage:
  uv run python scripts/train_mnist.py                # smolgrad only
  uv run python scripts/train_mnist.py --torch-twin   # also train the PyTorch twin

Writes results/mnist_train.csv, results/mnist_summary.json, results/mnist_curves.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

import smolgrad as sg
import smolgrad.functional as F
import smolgrad.nn as nn
from datasets import RESULTS, load_mnist


def build(hidden=(512, 256), dtype=np.float32):
    dims = (784, *hidden, 10)
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], dtype=dtype))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def evaluate(model, x, y, bs=2000):
    correct, loss_sum = 0, 0.0
    with sg.no_grad():
        for i in range(0, len(x), bs):
            logits = model(sg.Tensor(x[i:i + bs]))
            loss_sum += F.cross_entropy(logits, y[i:i + bs], reduction="sum").item()
            correct += int((logits.data.argmax(1) == y[i:i + bs]).sum())
    return loss_sum / len(x), correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-twin", action="store_true")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _f64")
    args = ap.parse_args()

    xtr, ytr, xte, yte = load_mnist()
    dtype = np.dtype(args.dtype)
    xtr, xte = xtr.astype(dtype), xte.astype(dtype)
    sg.manual_seed(args.seed)
    model = build(dtype=dtype)
    opt = sg.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"model: {model.num_parameters():,} parameters")

    twin = None
    if args.torch_twin:
        import torch
        torch.manual_seed(args.seed)
        tlayers = []
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                tl = torch.nn.Linear(*layer.weight.shape[::-1], dtype=getattr(torch, args.dtype))
                with torch.no_grad():
                    tl.weight.copy_(torch.from_numpy(layer.weight.data))
                    tl.bias.copy_(torch.from_numpy(layer.bias.data))
                tlayers.append(tl)
            else:
                tlayers.append(torch.nn.ReLU())
        tmodel = torch.nn.Sequential(*tlayers)
        topt = torch.optim.AdamW(tmodel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        twin = (torch, tmodel, topt)

    order_rng = np.random.default_rng(args.seed)
    rows = []
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        perm = order_rng.permutation(len(xtr))
        t0 = time.perf_counter()
        run_loss, run_tloss, nb = 0.0, 0.0, 0
        sg_time = 0.0
        for i in range(0, len(xtr), args.batch_size):
            b = perm[i:i + args.batch_size]
            xb, yb = xtr[b], ytr[b]
            ts = time.perf_counter()
            opt.zero_grad()
            loss = F.cross_entropy(model(sg.Tensor(xb)), yb)
            loss.backward()
            opt.step()
            sg_time += time.perf_counter() - ts
            run_loss += loss.item()
            if twin:
                torch, tmodel, topt = twin
                topt.zero_grad()
                tl = torch.nn.functional.cross_entropy(tmodel(torch.from_numpy(xb)), torch.from_numpy(yb))
                tl.backward()
                topt.step()
                run_tloss += tl.item()
            nb += 1
        train_loss = run_loss / nb
        test_loss, test_acc = evaluate(model, xte, yte)
        row = dict(epoch=epoch, train_loss=train_loss, test_loss=test_loss, test_acc=test_acc,
                   smolgrad_epoch_seconds=sg_time)
        if twin:
            torch, tmodel, _ = twin
            with torch.no_grad():
                tlog = tmodel(torch.from_numpy(xte))
                t_acc = float((tlog.argmax(1).numpy() == yte).mean())
                t_tl = float(torch.nn.functional.cross_entropy(tlog, torch.from_numpy(yte)))
            max_w_diff = max(float(np.max(np.abs(p.data - tp.detach().numpy())))
                             for p, tp in zip(model.parameters(), tmodel.parameters()))
            row.update(torch_train_loss=run_tloss / nb, torch_test_loss=t_tl, torch_test_acc=t_acc,
                       max_abs_weight_diff=max_w_diff)
        rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              f"wall={time.perf_counter() - t0:.1f}s", flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"mnist_train{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = dict(
        config=vars(args), params=model.num_parameters(), architecture="784-512-256-10 ReLU MLP",
        final_test_acc=rows[-1]["test_acc"], best_test_acc=max(r["test_acc"] for r in rows),
        final_test_loss=rows[-1]["test_loss"],
        mean_smolgrad_epoch_seconds=float(np.mean([r["smolgrad_epoch_seconds"] for r in rows])),
        total_wall_seconds=time.perf_counter() - t_start,
    )
    if twin:
        summary.update(torch_final_test_acc=rows[-1]["torch_test_acc"],
                       final_max_abs_weight_diff=rows[-1]["max_abs_weight_diff"])
    (RESULTS / f"mnist_summary{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [r["train_loss"] for r in rows], "o-", label="smolgrad train")
    ax[0].plot(ep, [r["test_loss"] for r in rows], "o-", label="smolgrad test")
    if twin:
        ax[0].plot(ep, [r["torch_test_loss"] for r in rows], "x--", label="PyTorch twin test")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy (nats)")
    ax[0].legend(frameon=False)
    ax[1].plot(ep, [100 * r["test_acc"] for r in rows], "o-", label="smolgrad")
    if twin:
        ax[1].plot(ep, [100 * r["torch_test_acc"] for r in rows], "x--", label="PyTorch twin")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("test accuracy (%)")
    ax[1].legend(frameon=False)
    fig.suptitle(f"MNIST, 784-512-256-10 MLP, AdamW, batch {args.batch_size}, {args.dtype} (source: results/mnist_train{args.tag}.csv)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / f"mnist_curves{args.tag}.png", dpi=150)


if __name__ == "__main__":
    main()
