#!/usr/bin/env python3
"""Aggregate benchmark CSVs (median across repetitions) and draw plots.

Reads   results/bench/latency_<variant>_r<k>.csv, cdf_*.csv, throughput_*.csv
Writes  results/bench/summary_latency.csv, summary_throughput.csv and PNGs.

Usage: .venv/bin/python scripts/plot_bench.py [results/bench]
"""
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "results/bench"
PATHS = ["add", "cancel", "match"]
COLORS = {"add": "#4C72B0", "cancel": "#DD8452", "match": "#55A868"}


def base_variant(v):
    return re.sub(r"_r\d+$", "", v)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(D, pattern))):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


lat = load("latency_*_r*.csv")
if not lat:
    sys.exit(f"no latency files in {D}")

# (variant, scenario, size) -> metric -> [values across reps]
agg = defaultdict(lambda: defaultdict(list))
for r in lat:
    key = (base_variant(r["variant"]), r["scenario"], int(r["book_orders"]))
    for m in ["mean_ns", "p50_ns", "p90_ns", "p99_ns", "p999_ns", "max_ns"]:
        agg[key][m].append(float(r[m]))

metrics = ["mean_ns", "p50_ns", "p90_ns", "p99_ns", "p999_ns", "max_ns"]
with open(os.path.join(D, "summary_latency.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["variant", "scenario", "book_orders", "reps"] + [f"median_{m}" for m in metrics])
    for key in sorted(agg):
        reps = len(agg[key]["p50_ns"])
        w.writerow(list(key) + [reps] + [round(statistics.median(agg[key][m]), 1) for m in metrics])


def med(v, s, n, m):
    return statistics.median(agg[(v, s, n)][m])


sizes = sorted({k[2] for k in agg})
variants = sorted({k[0] for k in agg})
main = "idmap" if "idmap" in variants else variants[0]

# 1. Percentiles per path at a mid-sized book.
ref_size = 10_000 if 10_000 in sizes else sizes[0]
fig, ax = plt.subplots(figsize=(7, 4))
pcts = [("p50_ns", "p50"), ("p99_ns", "p99"), ("p999_ns", "p99.9")]
width = 0.25
for i, path in enumerate(PATHS):
    for j, (m, label) in enumerate(pcts):
        val = med(main, path, ref_size, m)
        x = i + (j - 1) * width
        ax.bar(x, val, width * 0.9, color=COLORS[path], alpha=[1.0, 0.65, 0.35][j])
        ax.text(x, val * 1.05, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(range(len(PATHS)))
ax.set_xticklabels([f"{p}\n(p50, p99, p99.9)" for p in PATHS])
ax.set_ylabel("latency per order, ns (log)")
ax.set_title(f"Engine latency by path, {ref_size:,} resting orders")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, f"source: {D}/latency_{main}_r*.csv, median of reps; timer step 41.7 ns",
         fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(D, "latency_percentiles.png"), dpi=150)
plt.close(fig)

# 2. p50 and p99 against book size.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (m, label) in zip(axes, [("p50_ns", "p50"), ("p99_ns", "p99")]):
    for path in PATHS:
        ys = [med(main, path, n, m) for n in sizes]
        ax.plot(sizes, ys, marker="o", color=COLORS[path], label=path)
        ax.annotate(f"{ys[-1]:.0f}", (sizes[-1], ys[-1]), textcoords="offset points", xytext=(4, 0),
                    fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("resting orders in book")
    ax.set_title(label)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("latency, ns (log)")
axes[0].legend(frameon=False, fontsize=8)
fig.text(0.01, 0.01, f"source: {D}/latency_{main}_r*.csv, median of reps", fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(D, "latency_vs_book_size.png"), dpi=150)
plt.close(fig)

# 3. Id index ablation: burst throughput per path (CPU time), median of reps.
burst = load("burst_*_r*.csv")
if burst and len(variants) > 1:
    bagg = defaultdict(list)
    for r in burst:
        bagg[(base_variant(r["variant"]), r["path"])].append(float(r["orders_per_cpu_sec"]))
    bpaths = ["add", "cancel", "match", "add_strided_ids"]
    vcolors = {"idmap": "#4C72B0", "stdmap": "#8C8C8C", "identity": "#C44E52"}
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(variants)
    for vi, v in enumerate(variants):
        for pi, path in enumerate(bpaths):
            if (v, path) not in bagg:
                continue
            val = statistics.median(bagg[(v, path)]) / 1e6
            x = pi + (vi - (len(variants) - 1) / 2) * width
            ax.bar(x, val, width * 0.9, color=vcolors.get(v, "#999999"), label=v if pi == 0 else None)
            ax.text(x, val * 1.08, f"{val:.2f}" if val < 1 else f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(bpaths)))
    ax.set_xticklabels(["add 1M", "cancel 1M", "match 1M", "add 20k\nstrided ids"])
    ax.set_ylabel("million orders per CPU second (log)")
    ax.set_title("Order id index variants, burst throughput")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, f"source: {D}/burst_<variant>_r*.csv, median of reps", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(D, "idmap_ablation.png"), dpi=150)
    plt.close(fig)

# 4. CDF per path at the reference size (first rep of the main variant).
cdf = [r for r in load(f"cdf_{main}_r1.csv") if int(r["book_orders"]) == ref_size]
if cdf:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for path in PATHS:
        pts = [(float(r["latency_ns"]), float(r["quantile"])) for r in cdf if r["scenario"] == path]
        ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=COLORS[path], label=path)
    ax.set_xscale("log")
    ax.set_xlabel("latency, ns (log)")
    ax.set_ylabel("fraction of orders")
    ax.set_title(f"Latency CDF, {ref_size:,} resting orders")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, f"source: {D}/cdf_{main}_r1.csv; steps are the 41.7 ns timer tick", fontsize=7,
             color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(D, "latency_cdf.png"), dpi=150)
    plt.close(fig)

# Throughput summaries (mixed flow and per-path bursts), median across runs.
def summarize(pattern, keyfn, cols, outname):
    groups = defaultdict(list)
    for r in load(pattern):
        groups[keyfn(r)].append(r)
    with open(os.path.join(D, outname), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "runs"] + [f"median_{c}" for c in cols] + [f"max_{c}" for c in cols])
        for k, rows in sorted(groups.items()):
            vals = {c: [float(r[c]) for r in rows] for c in cols}
            w.writerow([k, len(rows)] + [round(statistics.median(vals[c])) for c in cols]
                       + [round(max(vals[c])) for c in cols])


summarize("throughput_*_r*.csv", lambda r: base_variant(r["variant"]),
          ["events_per_wall_sec", "events_per_cpu_sec", "new_orders_per_cpu_sec"], "summary_throughput.csv")
summarize("burst_*_r*.csv", lambda r: base_variant(r["variant"]) + ":" + r["path"],
          ["orders_per_wall_sec", "orders_per_cpu_sec"], "summary_burst.csv")

print(open(os.path.join(D, "summary_latency.csv")).read())
print(open(os.path.join(D, "summary_throughput.csv")).read())
print(open(os.path.join(D, "summary_burst.csv")).read())
