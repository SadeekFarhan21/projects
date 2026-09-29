#!/usr/bin/env python3
"""Turn results/bench_*.jsonl into results/summary.csv and a few PNG charts.

Each configuration was run several times; charts and the summary use the
median across repetitions (the machine is shared, so single runs are noisy).

Usage: .venv/bin/python scripts/plot_results.py
"""
import csv
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

INK = "#1f2933"
MUTED = "#7b8794"
C1 = "#2f6fdf"
C2 = "#e0803a"
C3 = "#3a9e6f"
C4 = "#9b59b6"


def load(name):
    path = os.path.join(RES, "bench_%s.jsonl" % name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def style(ax, title, xlabel, ylabel, source):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.figure.text(0.01, 0.01, "Source " + source, fontsize=7, color=MUTED)


def median_by(rows, key, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for k, rs in groups.items():
        out[k] = {f: statistics.median([r[f] for r in rs if r.get(f) is not None] or [0]) for f in fields}
        out[k]["n"] = len(rs)
        out[k]["load1"] = statistics.median([r["load1_before"] for r in rs])
    return out


FIELDS = ["ops_per_sec", "p50_us", "p99_us", "p999_us", "ops_per_server_cpu_s"]


def main():
    summary = []

    # Pipeline depth sweep.
    rows = [r for r in load("pipeline") if r.get("label") == "pipeline"]
    if rows:
        m = median_by(rows, lambda r: r["pipeline"], FIELDS)
        xs = sorted(m)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        a1.plot(xs, [m[x]["ops_per_sec"] / 1e3 for x in xs], marker="o", color=C1, label="wall clock")
        a1.plot(xs, [m[x]["ops_per_server_cpu_s"] / 1e3 for x in xs], marker="o", color=C3,
                label="per server CPU second")
        a1.set_xscale("log", base=2)
        a1.set_xticks(xs, [str(x) for x in xs])
        a1.legend(frameon=False, fontsize=8)
        style(a1, "Throughput vs pipeline depth", "pipeline depth", "thousand SET per s", "results/bench_pipeline.jsonl")
        a2.plot(xs, [m[x]["p50_us"] for x in xs], marker="o", color=C1, label="p50")
        a2.plot(xs, [m[x]["p99_us"] for x in xs], marker="o", color=C2, label="p99")
        a2.set_xscale("log", base=2)
        a2.set_yscale("log")
        a2.set_xticks(xs, [str(x) for x in xs])
        a2.legend(frameon=False, fontsize=8)
        style(a2, "Request latency vs pipeline depth", "pipeline depth", "microseconds (log)", "results/bench_pipeline.jsonl")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(RES, "pipeline.png"), dpi=150)
        plt.close(fig)
        for x in xs:
            summary.append({"experiment": "pipeline", "config": "c=50 P=%d SET" % x, **m[x]})

    # Connection count sweep.
    rows = [r for r in load("clients") if r.get("label") == "clients"]
    if rows:
        m = median_by(rows, lambda r: r["clients"], FIELDS)
        xs = sorted(m)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        a1.plot(xs, [m[x]["ops_per_sec"] / 1e3 for x in xs], marker="o", color=C1, label="wall clock")
        a1.plot(xs, [m[x]["ops_per_server_cpu_s"] / 1e3 for x in xs], marker="o", color=C3,
                label="per server CPU second")
        a1.set_xscale("log", base=2)
        a1.set_xticks(xs, [str(x) for x in xs])
        a1.legend(frameon=False, fontsize=8)
        style(a1, "Throughput vs connections, no pipelining", "connections", "thousand SET per s", "results/bench_clients.jsonl")
        a2.plot(xs, [m[x]["p50_us"] for x in xs], marker="o", color=C1, label="p50")
        a2.plot(xs, [m[x]["p99_us"] for x in xs], marker="o", color=C2, label="p99")
        a2.set_xscale("log", base=2)
        a2.set_yscale("log")
        a2.set_xticks(xs, [str(x) for x in xs])
        a2.legend(frameon=False, fontsize=8)
        style(a2, "Latency vs connections", "connections", "microseconds (log)", "results/bench_clients.jsonl")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(RES, "clients.png"), dpi=150)
        plt.close(fig)
        for x in xs:
            summary.append({"experiment": "clients", "config": "c=%d P=1 SET" % x, **m[x]})

    # Workload mix.
    rows = [r for r in load("workload") if str(r.get("label", "")).startswith("workload-")]
    if rows:
        m = median_by(rows, lambda r: r["workload"], FIELDS)
        for w in ["set", "get", "mix"]:
            if w in m:
                summary.append({"experiment": "workload", "config": "c=50 P=16 %s" % w.upper(), **m[w]})

    # AOF fsync policy.
    rows = [r for r in load("aof") if str(r.get("label", "")).startswith("aof-") and "ops_per_sec" in r]
    if rows:
        m = median_by(rows, lambda r: (r["aof"], r["pipeline"]), FIELDS)
        modes = ["off", "no", "everysec", "always"]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        w = 0.38
        for i, (p, col) in enumerate([(1, C1), (16, C3)]):
            vals = [m[(md, p)]["ops_per_sec"] / 1e3 if (md, p) in m else 0 for md in modes]
            xs = [j + (i - 0.5) * w for j in range(len(modes))]
            bars = ax.bar(xs, vals, width=w, color=col, label="pipeline %d" % p)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, "%.0f" % v, ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(modes)), ["AOF off", "fsync no", "fsync everysec", "fsync always"])
        ax.legend(frameon=False, fontsize=8)
        style(ax, "Write throughput by persistence mode, 50 connections", "", "thousand SET per s",
              "results/bench_aof.jsonl")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(RES, "aof.png"), dpi=150)
        plt.close(fig)
        for md in modes:
            for p in (1, 16):
                if (md, p) in m:
                    summary.append({"experiment": "aof", "config": "c=50 P=%d SET aof=%s" % (p, md), **m[(md, p)]})

    # Single connection: fsync cost without group commit (summary only, no chart).
    rows = [r for r in load("fsync1") if "ops_per_sec" in r]
    if rows:
        m = median_by(rows, lambda r: r["aof"], FIELDS)
        for md in ("off", "no", "everysec", "always"):
            if md in m:
                summary.append({"experiment": "fsync1", "config": "c=1 P=1 SET aof=%s" % md, **m[md]})

    # Active expiry.
    rows = [r for r in load("expiry") if r.get("label") == "expiry"]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.plot([r["t_ms_after_deadline"] for r in rows], [r["dbsize"] / 1e3 for r in rows], color=C1, marker=".")
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.8)
        ax.text(10, ax.get_ylim()[1] * 0.97, "all TTL keys expire", fontsize=7, color=MUTED, va="top")
        style(ax, "Keys physically present after a mass expiry, no client reads",
              "ms after the shared deadline", "thousand keys", "results/bench_expiry.jsonl")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(RES, "expiry.png"), dpi=150)
        plt.close(fig)

    # AOF replay.
    rows = [r for r in load("replay") if r.get("label") == "replay"]
    if rows:
        by = defaultdict(list)
        for r in rows:
            by[r["commands"]].append(r)
        for n in sorted(by):
            rs = by[n]
            summary.append({"experiment": "replay", "config": "%d commands, %.1f MB" % (n, rs[0]["aof_bytes"] / 1e6),
                            "replay_wall_ms": statistics.median(r["replay_wall_ms"] for r in rs),
                            "replay_cpu_ms": statistics.median(r["replay_cpu_ms"] for r in rs),
                            "n": len(rs), "load1": statistics.median(r["load1"] for r in rs)})

    cols = ["experiment", "config", "n", "ops_per_sec", "ops_per_server_cpu_s", "p50_us", "p99_us", "p999_us",
            "replay_wall_ms", "replay_cpu_ms", "load1"]
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in summary:
            w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
    print("wrote results/summary.csv and charts")


if __name__ == "__main__":
    main()
