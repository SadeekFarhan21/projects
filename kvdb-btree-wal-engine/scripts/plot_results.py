"""Aggregate benchmark CSVs (median over repetitions) and draw the plots.

Usage: .venv/bin/python scripts/plot_results.py
Reads results/bench_main_rep*.csv and results/bench_sync.csv, writes
results/summary_main.csv and results/*.png.
"""
import csv
import glob
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
ENGINES = ["std::map", "kvdb", "kvdb_pool16MiB", "sqlite"]
COLORS = {"std::map": "#9aa0a6", "kvdb": "#1a73e8", "kvdb_pool16MiB": "#8ab4f8", "sqlite": "#e8710a"}
WORKLOADS = ["fillseq", "fillrandom", "readseq@seq", "readrandom@random"]


def load_main():
    runs = defaultdict(list)
    files = sorted(glob.glob(str(RES / "bench_main_rep*.csv")))
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                runs[(row["engine"], row["workload"])].append(row)
    summary = []
    for (eng, wl), rows in runs.items():
        med = lambda k: statistics.median(float(r[k]) for r in rows)  # noqa: E731
        summary.append({
            "engine": eng, "workload": wl, "n": rows[0]["n"], "reps": len(rows),
            "ops_per_sec_median": round(med("ops_per_sec")),
            "ops_per_sec_min": round(min(float(r["ops_per_sec"]) for r in rows)),
            "ops_per_sec_max": round(max(float(r["ops_per_sec"]) for r in rows)),
            "ops_per_cpu_sec_median": round(med("ops_per_cpu_sec")),
            "p50_ns_median": med("p50_ns"), "p99_ns_median": med("p99_ns"),
            "p999_ns_median": med("p999_ns"), "note_rep1": rows[0]["note"],
        })
    order = {e: i for i, e in enumerate(ENGINES)}
    summary.sort(key=lambda r: (order.get(r["engine"], 99), r["workload"]))
    with open(RES / "summary_main.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    return {(r["engine"], r["workload"]): r for r in summary}, len(files)


def grouped_bars(ax, data, key, ylabel, log=True):
    width = 0.8 / len(ENGINES)
    for i, eng in enumerate(ENGINES):
        vals = [data.get((eng, wl), {}).get(key, 0) for wl in WORKLOADS]
        xs = [j + (i - (len(ENGINES) - 1) / 2) * width for j in range(len(WORKLOADS))]
        ax.bar(xs, vals, width, label=eng, color=COLORS[eng])
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels(WORKLOADS)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_main(data, reps):
    n = next(iter(data.values()))["n"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    grouped_bars(axes[0], data, "ops_per_sec_median", "operations / second (log)")
    axes[0].set_title("Wall-clock throughput")
    grouped_bars(axes[1], data, "ops_per_cpu_sec_median", "")
    axes[1].set_title("Throughput per CPU-second (user + sys)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{int(n):,} keys (16 B key, 100 B value), median of {reps} runs")
    fig.text(0.01, 0.01, "Source: results/bench_main_rep*.csv via results/summary_main.csv",
             fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(RES / "throughput.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, key, name in [(axes[0], "p50_ns_median", "p50"), (axes[1], "p99_ns_median", "p99")]:
        grouped_bars(ax, data, key, "latency per op, ns (log)")
        ax.set_title(f"{name} latency")
        ax.tick_params(axis="x", labelrotation=15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Source: results/summary_main.csv (median of per-run percentiles)",
             fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(RES / "latency.png", dpi=150)
    plt.close(fig)


def plot_sync():
    path = RES / "bench_sync.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    modes = ["none", "fsync", "fullfsync"]
    for i, eng in enumerate(["kvdb", "sqlite"]):
        vals = [next((float(r["ops_per_sec"]) for r in rows
                      if r["engine"] == eng and r["workload"] == f"put_{m}"), 0) for m in modes]
        axes[0].bar([j + (i - 0.5) * 0.38 for j in range(3)], vals, 0.38, label=eng,
                    color=COLORS[eng])
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["no sync", "fsync", "F_FULLFSYNC"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("commits / second (log)")
    axes[0].set_title("One put per commit, by durability mode")
    axes[0].legend(frameon=False)
    batch = [(int(r["batch"]), float(r["ops_per_sec"])) for r in rows
             if r["workload"] == "batch_fullfsync"]
    axes[1].plot([b for b, _ in batch], [v for _, v in batch], "o-", color=COLORS["kvdb"])
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("puts per commit (WriteBatch size)")
    axes[1].set_ylabel("puts / second (log)")
    axes[1].set_title("Group commit with F_FULLFSYNC")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Source: results/bench_sync.csv", fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(RES / "sync.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data, reps = load_main()
    plot_main(data, reps)
    plot_sync()
    print("wrote", ", ".join(sorted(p.name for p in RES.glob("*.png"))), "and summary_main.csv")
