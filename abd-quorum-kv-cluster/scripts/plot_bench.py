"""Plot results/bench_summary.csv (and the per-trial raw CSV for spread).

Usage: uv run python scripts/plot_bench.py [--tag bench]
Writes results/<tag>_nodes.png and results/<tag>_quorum.png.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path):
    with open(path) as f:
        return [{k: (float(v) if k not in ("sweep",) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench")
    a = ap.parse_args()
    summ = load(RESULTS / f"{a.tag}_summary.csv")
    raw = load(RESULTS / f"{a.tag}_raw.csv")

    # Sweep 1: cluster size.
    s = sorted([r for r in summ if r["sweep"] == "nodes"], key=lambda r: r["size"])
    rw = [r for r in raw if r["sweep"] == "nodes"]
    if s:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        x = [int(r["size"]) for r in s]
        ax[0].plot(x, [r["throughput_ops_s"] for r in s], "o-", color="#2a6f97", label="median")
        ax[0].scatter([r["size"] for r in rw], [r["throughput_ops_s"] for r in rw],
                      s=12, color="#2a6f97", alpha=0.35, label="each trial")
        ax[0].set(title="Throughput vs cluster size", xlabel="nodes", ylabel="ops/s")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].plot(x, [r["all_p50_ms"] for r in s], "o-", label="p50", color="#2a6f97")
        ax[1].plot(x, [r["all_p99_ms"] for r in s], "s--", label="p99", color="#c1666b")
        ax[1].set(title="Latency vs cluster size", xlabel="nodes", ylabel="ms")
        ax[1].legend(frameon=False, fontsize=8)
        ax[2].plot(x, [r["node_cpu_net_us_per_op"] for r in s], "o-", color="#4d908e")
        ax[2].set(title="Node CPU per op (idle subtracted)", xlabel="nodes", ylabel="CPU us / op")
        ax[2].set_ylim(bottom=0)
        for axis in ax:
            axis.set_xticks(x)
            axis.spines[["top", "right"]].set_visible(False)
        fig.text(0.01, 0.01, f"N=3 R=2 W=2, 50% reads. Source: results/{a.tag}_summary.csv", fontsize=7, color="#555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(RESULTS / f"{a.tag}_nodes.png", dpi=150)

    # Sweep 2: quorum settings at 5 nodes.
    q = [r for r in summ if r["sweep"] == "quorum"]
    if q:
        q.sort(key=lambda r: (r["r"] + r["w"], r["r"]))
        labels = [f"R{int(r['r'])}W{int(r['w'])}" for r in q]
        colors = ["#c1666b" if r["r"] + r["w"] <= 3 else "#2a6f97" for r in q]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        ax[0].bar(labels, [r["throughput_ops_s"] for r in q], color=colors)
        ax[0].set(title="Throughput by quorum (red: R+W<=N)", ylabel="ops/s")
        ax[1].bar(labels, [r["get_p50_ms"] for r in q], color="#2a6f97", width=0.4, align="edge", label="get p50")
        ax[1].bar(labels, [r["put_p50_ms"] for r in q], color="#f4a259", width=-0.4, align="edge", label="put p50")
        ax[1].set(title="Median latency by op", ylabel="ms")
        ax[1].legend(frameon=False, fontsize=8)
        ax[2].bar(labels, [r["node_cpu_net_us_per_op"] for r in q], color=colors)
        ax[2].set(title="Node CPU per op (idle subtracted)", ylabel="CPU us / op")
        for axis in ax:
            axis.tick_params(axis="x", rotation=45, labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)
        fig.text(0.01, 0.01, f"5 nodes, N=3, 50% reads. Source: results/{a.tag}_summary.csv", fontsize=7, color="#555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(RESULTS / f"{a.tag}_quorum.png", dpi=150)
    print("plots written")


if __name__ == "__main__":
    main()
