#!/usr/bin/env bash
# Reproduce every results/ file. Takes about 20 minutes on an idle M4 Pro.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/inversion_demo.py
uv run python scripts/linearizability.py --seeds 5 > results/linearizability_log.txt 2>&1
uv run python scripts/linearizability.py --configs safe,no-wb --seeds 6 --keys 1 --clients 10 \
    --jitter-ms 20 --tag linearizability_hotkey > results/linearizability_hotkey_log.txt 2>&1
uv run python scripts/bench.py --trials 3 --duration 5 --tag bench > results/bench_log.txt 2>&1
uv run python scripts/bench.py --sweep quorum --trials 3 --duration 5 --eager-writeback \
    --tag bench_eager_wb > results/bench_eager_wb_log.txt 2>&1
uv run python scripts/plot_bench.py --tag bench
