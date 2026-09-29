"""Plot train/val loss and throughput from a run's log.csv.

Usage: uv run python scripts/plot_training.py --log results/train_v0_log.csv
Writes the PNG next to the log (same stem, _loss.png).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("results/train_v0_log.csv"))
    args = ap.parse_args()
    rows = list(csv.DictReader(args.log.open()))
    tokens = [int(r["tokens"]) / 1e6 for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tokens[1:], train[1:], "-", color="#1f77b4", label="train (mean over interval)")
    ax.plot(tokens, val, "-o", ms=3, color="#d62728", label="validation (fixed 40 batches)")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("cross-entropy loss (nats / token)")
    ax.set_ylim(min(val + train[1:]) - 0.1, min(6.0, max(val) + 0.1))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.annotate(f"final val {val[-1]:.3f}", (tokens[-1], val[-1]), textcoords="offset points",
                xytext=(-80, 20), fontsize=9)
    fig.text(0.01, 0.01, f"Source: {args.log}", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = args.log.with_name(args.log.stem.replace("_log", "") + "_loss.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
