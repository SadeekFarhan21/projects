#!/usr/bin/env bash
# The model runs reported in DEVLOG.md, sized to fit the ~45 minute per-job
# budget on a shared M4 Pro. Full-size versions are listed in README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
M=mlx:mlx-community/Qwen2.5-1.5B-Instruct-4bit
R=results/runs
# pass@k run, sql family: 30 tasks (10 per difficulty) x 4 samples at T=0.7
uv run python scripts/run_eval.py --adapter $M --family sql --subset 30 --n-samples 4 \
  --temperature 0.7 --workers 3 --seed 0 --out $R/qwen1.5b_sql_t07_n4_seed0.jsonl
# pass@k run, bugfix family: 15 tasks (5 per difficulty) x 4 samples at T=0.7
uv run python scripts/run_eval.py --adapter $M --family bugfix --subset 15 --n-samples 4 \
  --temperature 0.7 --workers 3 --seed 0 --out $R/qwen1.5b_bugfix_t07_n4_seed0.jsonl
# reproducibility: identical config and seed as the sql run
uv run python scripts/run_eval.py --adapter $M --family sql --subset 30 --n-samples 4 \
  --temperature 0.7 --workers 3 --seed 0 --out $R/qwen1.5b_sql_t07_n4_seed0_repeat.jsonl
