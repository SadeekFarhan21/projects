"""Validate the analytical cycle model against Verilator cycle counts.

Generates random GEMM jobs (shape, array size, buffer rows, bandwidth), runs
them on the RTL through tests/tb_sa.py::test_cycle_jobs in parallel shards,
and compares measured cycles with sysarray.cycle_model.predict.

  uv run python scripts/validate_model.py --jobs 200 --workers 6

Writes results/cycle_model_validation.csv, results/cycle_model_summary.json
and results/cycle_model_validation.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from sysarray.cycle_model import predict  # noqa: E402

BWS = [None, 32, 16, 8, 6, 4, 3, 2]
ADEPTH = 64


def make_jobs(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    for i in range(count):
        n = rng.choice([4, 8, 16])
        buf = rng.choice([16, 32, 64])
        jobs.append({
            "id": i, "n": n, "buf": buf,
            "M": rng.randint(1, 160), "K": rng.randint(1, 96), "N": rng.randint(1, 96),
            "bw": rng.choice(BWS), "seed": rng.getrandbits(31),
        })
    return jobs


def run_shard(args) -> list[dict]:
    from test_sa import run  # imported in the worker process
    n, shard, jobs = args
    work = ROOT / "build" / "model_jobs"
    work.mkdir(parents=True, exist_ok=True)
    jin = work / f"jobs_n{n}_s{shard}.json"
    jout = work / f"cycles_n{n}_s{shard}.json"
    jin.write_text(json.dumps(jobs))
    run(n, ADEPTH, "test_cycle_jobs",
        {"SA_JOBS_IN": str(jin), "SA_CYCLES_OUT": str(jout)}, tag=f"_s{shard}")
    return json.loads(jout.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    jobs = make_jobs(args.jobs, args.seed)
    shards = []
    for n in (4, 8, 16):
        mine = [j for j in jobs if j["n"] == n]
        k = max(1, args.workers // 3 + (1 if n == 16 else 0))
        for s in range(k):
            part = mine[s::k]
            if part:
                shards.append((n, s, part))
    # build once per array size before fanning out
    from test_sa import build
    for n in (4, 8, 16):
        build(n, ADEPTH)
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for part in ex.map(run_shard, shards):
            rows.extend(part)
    wall = time.time() - t0

    out = []
    for r in sorted(rows, key=lambda r: r["id"]):
        bw = float("inf") if r["bw"] is None else float(r["bw"])
        p = predict(r["M"], r["K"], r["N"], r["n"], r["buf"], bw)
        err = (p.cycles - r["measured"]) / r["measured"]
        out.append({**{k: r[k] for k in ("id", "n", "buf", "M", "K", "N")},
                    "bw": "inf" if r["bw"] is None else r["bw"],
                    "measured": r["measured"], "predicted": p.cycles,
                    "rel_err_pct": round(100 * err, 4),
                    "mac_util": round(p.macs / (r["n"] ** 2 * r["measured"]), 5)})
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    with open(res / "cycle_model_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    errs = [abs(o["rel_err_pct"]) for o in out]
    summary = {
        "jobs": len(out),
        "exact_matches": sum(1 for o in out if o["measured"] == o["predicted"]),
        "max_abs_err_pct": max(errs),
        "mean_abs_err_pct": sum(errs) / len(errs),
        "within_2pct": sum(1 for e in errs if e < 2.0),
        "total_measured_cycles": sum(o["measured"] for o in out),
        "wall_seconds": round(wall, 1),
    }
    (res / "cycle_model_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    plot(out, res / "cycle_model_validation.png")


def plot(out, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {4: "#4C72B0", 8: "#DD8452", 16: "#55A868"}
    for n in (4, 8, 16):
        xs = [o["measured"] for o in out if o["n"] == n]
        ys = [o["predicted"] for o in out if o["n"] == n]
        a1.scatter(xs, ys, s=14, alpha=0.8, color=colors[n], label=f"{n}x{n} array")
    lo = min(o["measured"] for o in out)
    hi = max(o["measured"] for o in out)
    a1.plot([lo, hi], [lo, hi], color="0.4", lw=1, ls="--", label="y = x")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("Verilator measured cycles")
    a1.set_ylabel("Model predicted cycles")
    a1.set_title("Predicted vs measured cycles, 200 random GEMMs")
    a1.legend(frameon=False, fontsize=9)

    errs = [o["rel_err_pct"] for o in out]
    a2.hist(errs, bins=30, color="#4C72B0")
    a2.set_xlabel("Model error (percent of measured cycles)")
    a2.set_ylabel("GEMMs")
    a2.set_title("Cycle model error distribution")
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Source: results/cycle_model_validation.csv", fontsize=8, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
