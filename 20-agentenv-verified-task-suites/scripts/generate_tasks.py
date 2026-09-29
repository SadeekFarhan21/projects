"""Generate both task families into tasks/ as YAML.

uv run python scripts/generate_tasks.py [--seed 0] [--workers 8]
"""
import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from agentenv import PROJECT_ROOT
from agentenv.families import bugfix_family, sql_family

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--only", choices=["sql", "bugfix"], default=None)
args = ap.parse_args()

out = PROJECT_ROOT / "tasks"
stats = {}
t0 = time.time()
if args.only in (None, "sql"):
    sql = sql_family.generate(args.seed)
    shutil.rmtree(out / "sql", ignore_errors=True)
    for t in sql:
        t.save(out / "sql" / f"{t.id}.yaml")
    stats["sql"] = {"n": len(sql), "difficulty": Counter(t.difficulty for t in sql),
                    "db": Counter(t.metadata["db"] for t in sql)}
    print("sql", stats["sql"])
if args.only in (None, "bugfix"):
    bug, bstats = bugfix_family.generate(args.seed, workers=args.workers)
    shutil.rmtree(out / "bugfix", ignore_errors=True)
    for t in bug:
        t.save(out / "bugfix" / f"{t.id}.yaml")
    stats["bugfix"] = {"n": len(bug), "difficulty": Counter(t.difficulty for t in bug),
                       "operator": Counter(t.metadata["operator"] for t in bug),
                       "package": Counter(t.metadata["package"] for t in bug), "mutation_stats": bstats}
    print("bugfix", json.dumps(stats["bugfix"], indent=1))
stats["wall_s"] = round(time.time() - t0, 1)
stats["seed"] = args.seed
name = "task_generation.json" if args.only is None else f"task_generation_{args.only}.json"
(PROJECT_ROOT / "results" / name).write_text(json.dumps(stats, indent=2))
