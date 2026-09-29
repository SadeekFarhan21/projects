"""Linearizability experiment: does the checker find anomalies, and when?

For each protocol configuration we run a Jepsen-style workload (concurrent
clients on a few hot keys, one node killed and restarted mid-run, random
per-replica delay to widen race windows) against a real 5-node cluster, then
run the WGL checker over the recorded history.

Configurations:
  safe        R=2 W=2, read write-back on   (R+W>N, the default protocol)
  no-wb       R=2 W=2, read write-back off  (Dynamo-style read: return max, no repair wait)
  weak        R=1 W=1, write-back on        (R+W<=N: quorums need not intersect)
  weak-no-wb  R=1 W=1, write-back off

Usage: uv run python scripts/linearizability.py [--seeds 5] [--duration 4]
Writes results/<tag>.csv (tag defaults to "linearizability") and one
counterexample history per failing configuration under results/histories/.
Redirect stdout to results/<tag>_log.txt to keep the log.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kvstore.checker import check_history  # noqa: E402
from kvstore.cluster import LocalCluster  # noqa: E402
from kvstore.jepsen import WorkloadConfig, kill_one_nemesis, run_workload, save_history  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONFIGS = [
    ("safe", 2, 2, True),
    ("no-wb", 2, 2, False),
    ("weak", 1, 1, True),
    ("weak-no-wb", 1, 1, False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--jitter-ms", type=float, default=5.0)
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    ap.add_argument("--tag", default="linearizability")
    args = ap.parse_args()
    configs = [c for c in CONFIGS if c[0] in args.configs.split(",")]
    (RESULTS / "histories").mkdir(parents=True, exist_ok=True)
    rows = []
    saved = set()
    for seed in range(args.seeds):
        for name, r, w, wb in configs:
            with LocalCluster(size=5, n=3, r=r, w=w, writeback=wb, jitter_ms=args.jitter_ms) as cl:
                cfg = WorkloadConfig(clients=args.clients, keys=args.keys,
                                     duration=args.duration, seed=seed)
                history, events = asyncio.run(run_workload(cl, cfg, kill_one_nemesis))
            t0 = time.perf_counter()
            res = check_history(history)
            check_s = time.perf_counter() - t0
            row = {
                "config": name, "r": r, "w": w, "writeback": wb, "seed": seed,
                "ops": len(history),
                "ok": sum(o.status == "ok" for o in history),
                "info": sum(o.status == "info" for o in history),
                "fail": sum(o.status == "fail" for o in history),
                "linearizable": res.ok, "states_explored": res.states_explored,
                "check_seconds": round(check_s, 3), "bad_key": res.key if not res.ok else "",
            }
            rows.append(row)
            print(row, flush=True)
            if not res.ok:
                print("   ", res.detail[:400], flush=True)
                if name not in saved:
                    saved.add(name)
                    stem = f"{args.tag}_{name}_seed{seed}"
                    save_history(str(RESULTS / "histories" / f"{stem}.jsonl"), history)
                    (RESULTS / "histories" / f"{stem}.txt").write_text(
                        f"events: {events}\nkey: {res.key}\n{res.detail}\n")
    with open(RESULTS / f"{args.tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("\nsummary (violations / runs):")
    for name, *_ in configs:
        rs = [x for x in rows if x["config"] == name]
        print(f"  {name:11s} {sum(not x['linearizable'] for x in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
