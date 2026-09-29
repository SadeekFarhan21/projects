"""Throughput / latency benchmark.

Starts a real local cluster per configuration, drives it with closed-loop
load from several load-generator processes, and records throughput and
latency percentiles.

Two sweeps:
  nodes   : cluster size in {3,5,7,9} at N=3, R=2, W=2
  quorum  : 5 nodes, N=3, every interesting (R, W)

Usage:
  uv run python scripts/bench.py                 # both sweeps, 3 trials each
  uv run python scripts/bench.py --sweep nodes --trials 1 --duration 3
Outputs results/bench_raw.csv (one row per trial) and results/bench_summary.csv
(median over trials); scripts/plot_bench.py turns them into PNGs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import multiprocessing as mp
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kvstore.client import KVClient  # noqa: E402
from kvstore.cluster import LocalCluster  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


async def _load(addrs, concurrency, duration, warmup, keyspace, read_frac, seed):
    rng = random.Random(seed)
    lat_get: list[float] = []
    lat_put: list[float] = []
    errors = 0
    total = {"get": 0, "put": 0}  # every op including warmup, for per-op cost ratios
    start = time.perf_counter()
    measure_from = start + warmup
    stop_at = measure_from + duration

    async def worker(i):
        nonlocal errors
        c = KVClient(addrs, seed=seed * 100 + i)
        try:
            while True:
                t0 = time.perf_counter()
                if t0 >= stop_at:
                    return
                key = f"key{rng.randrange(keyspace)}"
                is_read = rng.random() < read_frac
                r = await (c.get(key) if is_read else c.put(key, "x" * 64))
                t1 = time.perf_counter()
                total["get" if is_read else "put"] += 1
                if t0 >= measure_from and t1 <= stop_at:
                    if not r["ok"]:
                        errors += 1
                    (lat_get if is_read else lat_put).append(t1 - t0)
        finally:
            await c.close()

    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    return lat_get, lat_put, errors, total


def _load_proc(args):
    return asyncio.run(_load(*args))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def run_one(size, r, w, args, trial):
    with LocalCluster(size=size, n=3, r=r, w=w, eager_writeback=args.eager_writeback) as cluster:
        # Preload so reads mostly hit existing keys.
        async def preload():
            c = KVClient(cluster.addrs, seed=0)
            sem = asyncio.Semaphore(64)

            async def one(k):
                async with sem:
                    await c.put(f"key{k}", "x" * 64)

            await asyncio.gather(*(one(k) for k in range(args.keyspace)))
            await c.close()

        asyncio.run(preload())
        jobs = [
            (cluster.addrs, args.concurrency, args.duration, args.warmup, args.keyspace,
             args.read_frac, trial * 1000 + g)
            for g in range(args.generators)
        ]
        async def snapshot():
            c = KVClient(cluster.addrs)
            sts = [await c.status(j) for j in range(size)]
            await c.close()
            return (sum(s["cpu_s"] for s in sts),
                    sum(s["stats"]["replica_rpcs_sent"] for s in sts),
                    sum(s["stats"]["read_writebacks"] for s in sts))

        async def cpu_total():
            return (await snapshot())[0]

        # Idle CPU (heartbeats, hint loop) grows as O(nodes^2) pings; measure it
        # so we can report per-op CPU with the background cost subtracted.
        i0 = asyncio.run(cpu_total())
        time.sleep(1.0)
        idle_rate = asyncio.run(cpu_total()) - i0  # CPU-seconds per second

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.generators) as pool:
            # Warm the pool up first so process spawn does not eat into the window.
            pool.map(abs, range(args.generators))
            # Cost counters bracket the whole load run (warmup included) and are
            # divided by the total op count, so no window alignment is needed.
            cpu0, rpc0, wb0 = asyncio.run(snapshot())
            w0 = time.perf_counter()
            fut = pool.map_async(_load_proc, jobs)
            time.sleep(args.warmup + args.duration / 2)
            load0 = os.getloadavg()[0]
            outs = fut.get()
            cpu1, rpc1, wb1 = asyncio.run(snapshot())
            wall = time.perf_counter() - w0
    gets = [x for o in outs for x in o[0]]
    puts = [x for o in outs for x in o[1]]
    errors = sum(o[2] for o in outs)
    tot_get = sum(o[3]["get"] for o in outs)
    tot_ops = tot_get + sum(o[3]["put"] for o in outs)
    allops = gets + puts
    return {
        "size": size, "n": 3, "r": r, "w": w, "trial": trial,
        "ops": len(allops), "errors": errors,
        "throughput_ops_s": len(allops) / args.duration,
        "get_p50_ms": pct(gets, 50) * 1e3, "get_p99_ms": pct(gets, 99) * 1e3,
        "put_p50_ms": pct(puts, 50) * 1e3, "put_p99_ms": pct(puts, 99) * 1e3,
        "all_p50_ms": pct(allops, 50) * 1e3, "all_p99_ms": pct(allops, 99) * 1e3,
        "all_mean_ms": statistics.fmean(allops) * 1e3 if allops else float("nan"),
        # Cluster-wide node CPU per op: robust to other load on the machine,
        # unlike wall-clock throughput.
        "node_cpu_us_per_op": (cpu1 - cpu0) / max(1, tot_ops) * 1e6,
        "node_cpu_net_us_per_op": (cpu1 - cpu0 - idle_rate * wall) / max(1, tot_ops) * 1e6,
        "idle_cpu_frac": idle_rate,
        # Protocol cost, independent of machine load: remote replica RPCs per
        # client op and fraction of gets that needed a write-back.
        "replica_rpcs_per_op": (rpc1 - rpc0) / max(1, tot_ops),
        "writebacks_per_get": (wb1 - wb0) / max(1, tot_get),
        "loadavg_1m": load0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["nodes", "quorum", "both"], default="both")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--generators", type=int, default=4, help="load generator processes")
    ap.add_argument("--concurrency", type=int, default=32, help="in-flight ops per generator")
    ap.add_argument("--keyspace", type=int, default=2000)
    ap.add_argument("--read-frac", type=float, default=0.5)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--eager-writeback", action="store_true",
                    help="disable the straggler-read optimization (for comparison)")
    args = ap.parse_args()

    configs = []
    if args.sweep in ("nodes", "both"):
        configs += [("nodes", s, 2, 2) for s in (3, 5, 7, 9)]
    if args.sweep in ("quorum", "both"):
        configs += [("quorum", 5, r, w) for r, w in
                    [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]]

    RESULTS.mkdir(exist_ok=True)
    raw_path = RESULTS / f"{args.tag}_raw.csv"
    rows = []
    print(f"# {platform.platform()} python {platform.python_version()} cpus={os.cpu_count()}")
    print(f"# generators={args.generators} concurrency={args.concurrency} duration={args.duration}s "
          f"read_frac={args.read_frac} keyspace={args.keyspace}")
    for trial in range(args.trials):
        for sweep, size, r, w in configs:
            row = {"sweep": sweep, **run_one(size, r, w, args, trial)}
            rows.append(row)
            print(f"{sweep:6s} size={size} R={r} W={w} trial={trial} "
                  f"thr={row['throughput_ops_s']:8.0f} ops/s  p50={row['all_p50_ms']:.2f}ms "
                  f"p99={row['all_p99_ms']:.2f}ms  cpu/op={row['node_cpu_us_per_op']:.0f}us "
                  f"net={row['node_cpu_net_us_per_op']:.0f}us idle={row['idle_cpu_frac']:.3f} "
                  f"rpcs/op={row['replica_rpcs_per_op']:.2f} wb/get={row['writebacks_per_get']:.3f} "
                  f"load={row['loadavg_1m']:.0f} errors={row['errors']}", flush=True)
            with open(raw_path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)

    # Median over trials per config.
    summary = []
    keys = sorted({(r["sweep"], r["size"], r["r"], r["w"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2], k[3]))
    metrics = [k for k in rows[0] if k.endswith("_ms") or k in
               ("throughput_ops_s", "errors", "ops", "node_cpu_us_per_op",
                "node_cpu_net_us_per_op", "idle_cpu_frac", "loadavg_1m",
                "replica_rpcs_per_op", "writebacks_per_get")]
    for k in keys:
        group = [r for r in rows if (r["sweep"], r["size"], r["r"], r["w"]) == k]
        s = {"sweep": k[0], "size": k[1], "n": 3, "r": k[2], "w": k[3], "trials": len(group)}
        for m in metrics:
            s[m] = statistics.median(r[m] for r in group)
        if len(group) > 1:
            s["throughput_stdev"] = statistics.stdev(r["throughput_ops_s"] for r in group)
        summary.append(s)
    with open(RESULTS / f"{args.tag}_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    print(f"wrote {raw_path} and {RESULTS / f'{args.tag}_summary.csv'}")


if __name__ == "__main__":
    main()
