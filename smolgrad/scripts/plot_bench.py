"""Plot results/bench_threads_*.csv into results/bench_step_time.png and
results/bench_phases.png. Usage: uv run python scripts/plot_bench.py"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from datasets import RESULTS  # noqa: E402

COLORS = {"smolgrad": "#2a6fdb", "numpy_manual": "#8a8a8a", "torch": "#e0712c"}


def load(tag):
    p = RESULTS / f"bench_threads_{tag}.csv"
    if not p.exists():
        return None
    with open(p) as f:
        return [dict(r, batch=int(r["batch"]), min_ms=float(r["min_ms"]), median_ms=float(r["median_ms"]))
                for r in csv.DictReader(f)]


def main():
    tags = [t for t in ("1", "default") if load(t)]
    fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 3.8), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        rows = [r for r in load(tag) if r["workload"] == "mlp" and r["phase"] == "step"]
        for impl, c in COLORS.items():
            rr = sorted((r for r in rows if r["impl"] == impl), key=lambda r: r["batch"])
            ax.loglog([r["batch"] for r in rr], [r["min_ms"] for r in rr], "o-", color=c, label=impl)
        ax.set_xlabel("batch size")
        ax.set_ylabel("train step time, min over runs (ms)")
        ax.set_title(f"MLP 784-512-256-10, threads={tag}", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("source: results/bench_threads_*.csv", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "bench_step_time.png", dpi=150)

    tag = tags[0]
    rows = [r for r in load(tag) if r["workload"] == "mlp" and r["batch"] == 128 and r["phase"] != "step"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    impls = list(COLORS)
    bottom = [0.0] * len(impls)
    for phase, shade in (("forward", 1.0), ("backward", 0.65), ("optimizer", 0.35)):
        vals = [next(r["min_ms"] for r in rows if r["impl"] == i and r["phase"] == phase) for i in impls]
        ax.barh(impls, vals, left=bottom, color=[COLORS[i] for i in impls], alpha=shade, label=phase,
                edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    for y, total in enumerate(bottom):
        ax.text(total, y, f" {total:.2f} ms", va="center", fontsize=8)
    ax.set_xlabel("ms per step (forward, backward, optimizer; darker is earlier)")
    ax.set_xlim(0, max(bottom) * 1.25)
    ax.set_title(f"Phase split at batch 128, threads={tag} (source: results/bench_threads_{tag}.csv)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "bench_phases.png", dpi=150)


if __name__ == "__main__":
    main()
