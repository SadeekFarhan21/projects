"""Verifier soundness check over every task.

For each task: the reference agent runs R times and the null agent runs R
times, through the full episode path (workspace, tools, sandbox, verifier).
A task is sound when reference passes R/R and null passes 0/R. A verifier is
flaky when its outcome differs across the R repeats of the same agent.

uv run python scripts/validate_verifiers.py --repeats 5 --workers 10
"""
import argparse
import json
import time

import pandas as pd

from agentenv import PROJECT_ROOT
from agentenv.metrics import load_traces
from agentenv.runner import run_eval
from agentenv.task import load_tasks



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--family", default=None)
    args = ap.parse_args()

    tasks = load_tasks(PROJECT_ROOT / "tasks", family=args.family)
    runs = PROJECT_ROOT / "results" / "runs"
    metas = {}
    t0 = time.time()
    for agent in ("reference", "null"):
        metas[agent] = run_eval(tasks, agent, runs / f"validation_{agent}.jsonl", n_samples=args.repeats,
                                workers=args.workers)
    rows = []
    for agent in ("reference", "null"):
        for tr in load_traces([runs / f"validation_{agent}.jsonl"]):
            rows.append({"agent": agent, "task_id": tr["task_id"], "family": tr["family"],
                         "difficulty": tr["difficulty"], "sample": tr["sample"], "passed": tr["passed"],
                         "stop_reason": tr["stop_reason"], "verify_wall_s": tr["verifier"].get("wall_s", 0.0),
                         "episode_wall_s": tr["wall_s"]})
    df = pd.DataFrame(rows)
    per = df.pivot_table(index=["task_id", "family", "difficulty"], columns="agent", values="passed",
                         aggfunc="sum").reset_index()
    per.columns.name = None
    per = per.rename(columns={"reference": "reference_passes", "null": "null_passes"})
    per["repeats"] = args.repeats
    per["reference_flaky"] = ~per["reference_passes"].isin([0, args.repeats])
    per["null_flaky"] = ~per["null_passes"].isin([0, args.repeats])
    per["sound"] = (per["reference_passes"] == args.repeats) & (per["null_passes"] == 0)
    per.to_csv(PROJECT_ROOT / "results" / "verifier_validation.csv", index=False)

    summary = {"repeats": args.repeats, "n_tasks": len(per), "wall_s": round(time.time() - t0, 1)}
    for fam, g in [("all", per), *per.groupby("family")]:
        d = df if fam == "all" else df[df.family == fam]
        summary[fam] = {
            "tasks": int(len(g)),
            "reference_pass_rate": float(d[d.agent == "reference"].passed.mean()),
            "null_pass_rate": float(d[d.agent == "null"].passed.mean()),
            "sound_tasks": int(g["sound"].sum()),
            "flaky_verifiers": int((g["reference_flaky"] | g["null_flaky"]).sum()),
            "median_verify_wall_s": float(d.verify_wall_s.median()),
            "p95_verify_wall_s": float(d.verify_wall_s.quantile(0.95)),
        }
    summary["runner"] = metas
    (PROJECT_ROOT / "results" / "verifier_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runner"}, indent=2))
    bad = per[~per["sound"]]
    if len(bad):
        print("UNSOUND TASKS\n", bad.to_string(index=False))


if __name__ == "__main__":
    main()
