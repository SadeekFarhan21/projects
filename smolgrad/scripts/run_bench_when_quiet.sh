#!/usr/bin/env bash
# Wait (up to MAX_WAIT seconds) for the 1-minute load average to drop below
# LOAD_MAX, then run both benchmark configurations and the plots. The machine
# this was developed on was shared with other heavy jobs, so this matters.
set -euo pipefail
cd "$(dirname "$0")/.."
LOAD_MAX=${LOAD_MAX:-20}
MAX_WAIT=${MAX_WAIT:-2400}
waited=0
while :; do
  load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
  if [ "$load" -lt "$LOAD_MAX" ] || [ "$waited" -ge "$MAX_WAIT" ]; then break; fi
  sleep 30; waited=$((waited + 30))
done
echo "starting after ${waited}s wait, loadavg: $(sysctl -n vm.loadavg)"
uv run python scripts/bench.py --threads 1 --rounds 8 | tee results/bench_threads_1.log
uv run python scripts/bench.py --rounds 8 | tee results/bench_threads_default.log
uv run python scripts/plot_bench.py
echo "done, loadavg: $(sysctl -n vm.loadavg)"
