"""Run an adapter over tasks and write JSONL traces plus a summary.

Examples
  uv run python scripts/run_eval.py --adapter reference --out results/runs/reference.jsonl
  uv run python scripts/run_eval.py --adapter mlx:mlx-community/Qwen2.5-1.5B-Instruct-4bit \
      --n-samples 4 --temperature 0.7 --workers 3 --subset 20 --out results/runs/qwen1.5b.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from agentenv import PROJECT_ROOT
from agentenv.agent import EpisodeConfig
from agentenv.metrics import episodes_frame, failure_table, load_traces, summarize
from agentenv.runner import run_eval
from agentenv.task import load_tasks



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=["sql", "bugfix"], default=None)
    ap.add_argument("--subset", type=int, default=0,
                    help="per family, pick this many tasks stratified by difficulty (0 = all)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens-per-turn", type=int, default=384)
    args = ap.parse_args()

    tasks = load_tasks(PROJECT_ROOT / "tasks", family=args.family)
    if args.subset:
        rng = random.Random(args.subset_seed)
        picked = []
        for fam in sorted({t.family for t in tasks}):
            by_diff = {}
            for t in tasks:
                if t.family == fam:
                    by_diff.setdefault(t.difficulty, []).append(t)
            per = {d: args.subset // 3 + (1 if i < args.subset % 3 else 0)
                   for i, d in enumerate(("easy", "medium", "hard"))}
            for d, lst in sorted(by_diff.items()):
                picked += rng.sample(lst, min(per[d], len(lst)))
        tasks = sorted(picked, key=lambda t: t.id)

    cfg = EpisodeConfig(temperature=args.temperature, max_tokens_per_turn=args.max_tokens_per_turn)
    meta = run_eval(tasks, args.adapter, args.out, n_samples=args.n_samples, base_seed=args.seed,
                    workers=args.workers, cfg=cfg)
    df = episodes_frame(load_traces([args.out]))
    ks = [k for k in (1, 2, 4, 8) if k <= args.n_samples]
    summ = summarize(df, ks=ks)
    out = Path(args.out)
    summ.to_csv(out.with_suffix(".summary.csv"), index=False)
    failure_table(df).to_csv(out.with_suffix(".failures.csv"), index=False)
    print(json.dumps(meta, indent=1))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
