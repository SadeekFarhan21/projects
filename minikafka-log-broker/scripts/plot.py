"""Plots the CSVs written by scripts/run_benchmarks.sh into results/*.png.

Usage: .venv/bin/python scripts/plot.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
COLORS = ["#2f6db3", "#d9822b", "#3a9a5b", "#b8433f", "#7a5aa6"]


def rows(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def finish(ax, title, source, fname, fig):
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, which="major", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, f"Source: results/{source}", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / fname, dpi=150)
    plt.close(fig)


def plot_produce():
    by = defaultdict(list)
    for r in rows("produce_throughput.csv"):
        by[int(r["msg_size"])].append((int(r["batch"]), float(r["mb_per_s_median"]), float(r["msgs_per_s_median"])))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (size, pts) in enumerate(sorted(by.items())):
        pts.sort()
        a1.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[i], label=f"{size} B messages")
        a2.plot([p[0] for p in pts], [p[2] for p in pts], "o-", color=COLORS[i], label=f"{size} B messages")
    for a, yl in ((a1, "MB/s (payload)"), (a2, "messages/s")):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("records per produce request")
        a.set_ylabel(yl)
    a1.legend(frameon=False, fontsize=8)
    finish(a1, "Producer throughput vs batch size", "produce_throughput.csv", "produce_throughput.png", fig)


def plot_consume():
    by = defaultdict(list)
    for r in rows("consume_throughput.csv"):
        by[int(r["msg_size"])].append((int(r["fetch_bytes"]) / 1024, float(r["mb_per_s_median"])))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (size, pts) in enumerate(sorted(by.items())):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[i], label=f"{size} B messages")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fetch max_bytes (KiB)")
    ax.set_ylabel("MB/s (payload, CRC-checked)")
    ax.legend(frameon=False, fontsize=8)
    finish(ax, "Consumer throughput vs fetch size", "consume_throughput.csv", "consume_throughput.png", fig)


def plot_latency():
    by = defaultdict(list)
    for r in rows("latency_raw.csv"):
        by[(r["flush"], int(r["rate_msgs_per_s"]), int(r["batch"]))].append(float(r["latency_us"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (key, v) in enumerate(sorted(by.items(), key=lambda kv: (kv[0][0] != "none", kv[0][1]))):
        v.sort()
        n = len(v)
        ax.plot(v, [(k + 1) / n for k in range(n)], color=COLORS[i % len(COLORS)],
                label=f"flush={key[0]}, {key[1]} msg/s, batch {key[2]}")
    ax.set_xscale("log")
    ax.set_xlabel("end-to-end latency (microseconds)")
    ax.set_ylabel("fraction of messages")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    finish(ax, "End-to-end latency CDF (produce to consumer long poll)", "latency_raw.csv", "latency_cdf.png", fig)


def plot_flush():
    by = defaultdict(list)
    for r in rows("flush_policy.csv"):
        by[r["flush"]].append((int(r["batch"]), float(r["msgs_per_s_median"])))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (pol, pts) in enumerate(by.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[i], label=pol)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("records per produce request (1000 B each)")
    ax.set_ylabel("messages/s")
    ax.legend(frameon=False, fontsize=8, title="flush policy", title_fontsize=8)
    finish(ax, "Cost of durability: flush policy vs batch size", "flush_policy.csv", "flush_policy.png", fig)


def plot_scaling():
    r = rows("producer_scaling.csv")
    xs = [int(x["producers"]) for x in r]
    ys = [float(x["mb_per_s_median"]) for x in r]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", color=COLORS[0], label="measured")
    ax.plot(xs, [ys[0] * x for x in xs], "--", color="#999", label="linear from 1 producer")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("concurrent producers (one partition each)")
    ax.set_ylabel("aggregate MB/s")
    ax.legend(frameon=False, fontsize=8)
    finish(ax, "Producer scaling across partitions", "producer_scaling.csv", "producer_scaling.png", fig)


if __name__ == "__main__":
    plot_produce()
    plot_consume()
    plot_latency()
    plot_flush()
    plot_scaling()
    print("wrote plots to", RES)
