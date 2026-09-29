"""Constrained random regression across array sizes, in parallel shards.

  uv run python scripts/run_regression.py --gemms 10000 --workers 8

Every shard runs tests/tb_sa.py::test_random_gemms with its own seed and
random handshake profiles (random valid gaps and random o_ready backpressure
on most GEMMs). A failing shard is replayed on its failing GEMM with a VCD
dump in build/waves/. Also merges functional coverage from this run and the
directed tests (if they ran) into results/coverage.json.

Writes results/regression_summary.json and results/coverage.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from sysarray.coverage import closure, merge  # noqa: E402

SPLIT = {4: 0.3, 8: 0.5, 16: 0.2}   # share of GEMMs per array size


def run_shard(args):
    from test_sa import BUILD, cov_path, run_with_replay
    n, shard, count = args
    stats = BUILD / "regression" / f"stats_n{n}_s{shard}.json"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.unlink(missing_ok=True)
    env = {"SA_SEED": str(100_000 * n + shard), "SA_NUM_GEMMS": str(count),
           "SA_COV_OUT": cov_path(f"reg_n{n}_s{shard}"), "SA_STATS_OUT": str(stats)}
    ok = True
    try:
        run_with_replay(n, 64, "test_random_gemms", env, tag=f"_reg{shard}")
    except SystemExit:
        ok = False
    st = json.loads(stats.read_text()) if stats.exists() else {"n": n, "gemms": count}
    st["passed"] = ok
    st["shard"] = shard
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemms", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-shard", type=int, default=250)
    args = ap.parse_args()

    from test_sa import BUILD, build
    for n in SPLIT:
        build(n, 64)
    shards = []
    for n, frac in SPLIT.items():
        total = round(args.gemms * frac)
        k = max(1, -(-total // args.per_shard))
        for s in range(k):
            cnt = total // k + (1 if s < total % k else 0)
            shards.append((n, s, cnt))
    # longest shards first
    shards.sort(key=lambda t: -t[0] * t[2])
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        stats = list(ex.map(run_shard, shards))
    wall = time.time() - t0

    per_n = {}
    for n in SPLIT:
        mine = [s for s in stats if s["n"] == n]
        per_n[str(n)] = {
            "gemms": sum(s["gemms"] for s in mine),
            "shards": len(mine),
            "failed_shards": sum(1 for s in mine if not s["passed"]),
            "sim_cycles": sum(s.get("cycles", 0) for s in mine),
            "sva_errors": sum(s.get("sva_errors", 0) for s in mine),
            "stability_errors": sum(s.get("stability_errors", 0) for s in mine),
        }
    summary = {
        "gemms_total": sum(v["gemms"] for v in per_n.values()),
        "gemms_passed": sum(s["gemms"] for s in stats if s["passed"]),
        "failed_shards": sum(1 for s in stats if not s["passed"]),
        "sva_errors": sum(v["sva_errors"] for v in per_n.values()),
        "stability_errors": sum(v["stability_errors"] for v in per_n.values()),
        "sim_cycles": sum(v["sim_cycles"] for v in per_n.values()),
        "wall_seconds": round(wall, 1),
        "workers": args.workers,
        "per_array_size": per_n,
    }
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "regression_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    files = sorted((BUILD / "coverage").glob("*.json"))
    merged = merge(files)
    hit, total, holes = closure(merged)
    (res / "coverage.json").write_text(json.dumps(
        {"bins_hit": hit, "bins_total": total, "holes": holes,
         "sources": [f.name for f in files], "bins": merged}, indent=1))
    print(f"coverage {hit}/{total} bins, holes: {holes}")


if __name__ == "__main__":
    main()
