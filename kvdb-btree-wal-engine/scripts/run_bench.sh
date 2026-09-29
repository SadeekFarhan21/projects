#!/usr/bin/env bash
# Builds the release config and runs every benchmark, writing raw outputs to
# results/. Usage: scripts/run_bench.sh [N] [REPS]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-1000000}"
REPS="${2:-3}"
cmake --preset release >/dev/null
# -j2 keeps the build polite on a shared machine.
cmake --build --preset release --target kvdb_bench -j2 >/dev/null
mkdir -p results
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uname -srm)"
  sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize 2>/dev/null | paste -sd' ' -
  echo "# uptime: $(uptime)"
} > results/machine.txt
for r in $(seq 1 "$REPS"); do
  ./build/release/kvdb_bench --exp=main --n="$N" --out="results/bench_main_rep${r}.csv" \
    | tee "results/bench_main_rep${r}.txt"
done
./build/release/kvdb_bench --exp=sync --n=2000 --out=results/bench_sync.csv | tee results/bench_sync.txt
./build/release/kvdb_bench --exp=walgrow --n=3000 --out=results/bench_walgrow.csv | tee results/bench_walgrow.txt
echo "# uptime after: $(uptime)" >> results/machine.txt
