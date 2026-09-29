"""Plot the memory system sweeps from results/bench.csv.

Usage: uv run scripts/plot_bench.py [results/bench.csv]
Writes results/latency_cycles.png, results/latency_stall_fraction.png and
results/interval_stall_fraction.png.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KERNELS = ["vecadd_8", "vecadd_64", "matmul_4x4", "matmul_8x8"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["sweep"]].append(r)
    return rows


def series(rows, kernel, x_key, y_key):
    pts = sorted(
        (int(r[x_key]), float(r[y_key])) for r in rows if r["kernel"] == kernel
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/bench.csv")
    out = src.parent
    rows = load(src)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, c in zip(KERNELS, COLORS):
        x, y = series(rows["latency"], k, "latency", "cycles")
        ax.plot(x, y, marker="o", color=c, label=k)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
    style(ax, "Kernel cycles vs DRAM latency", "DRAM latency (cycles)", "Total cycles")
    fig.tight_layout()
    fig.savefig(out / "latency_cycles.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, c in zip(KERNELS, COLORS):
        x, y = series(rows["latency"], k, "latency", "stall_frac")
        ax.plot(x, [100 * v for v in y], marker="o", color=c, label=k)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
    ax.set_ylim(0, 100)
    style(ax, "Share of cycles stalled on memory", "DRAM latency (cycles)", "Stalled cycles (%)")
    fig.tight_layout()
    fig.savefig(out / "latency_stall_fraction.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, c in zip(KERNELS, COLORS):
        x, y = series(rows["interval"], k, "interval", "stall_frac")
        ax.plot(x, [100 * v for v in y], marker="o", color=c, label=k)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(0, 100)
    style(
        ax,
        "Stalls vs DRAM issue interval at latency 8",
        "Cycles between accepted requests",
        "Stalled cycles (%)",
    )
    fig.tight_layout()
    fig.savefig(out / "interval_stall_fraction.png", dpi=150)
    print("wrote plots to", out)


if __name__ == "__main__":
    main()
