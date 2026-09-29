#!/usr/bin/env bash
# Runs every benchmark mode and writes raw CSVs to results/.
# Usage: scripts/run_benchmarks.sh [reps]   (default 3 repetitions per point)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-build/release}
REPS=${1:-3}
OUT=results
mkdir -p "$OUT"

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion)"
  echo "compiler: $(c++ --version | head -1)"
  echo "load average before run: $(sysctl -n vm.loadavg)"
  echo "reps per point: $REPS"
} > "$OUT/bench_env.txt"

echo "== produce";  "$BIN/mk-bench" produce --reps "$REPS" | tee "$OUT/produce_throughput.csv"
echo "== consume";  "$BIN/mk-bench" consume --reps "$REPS" | tee "$OUT/consume_throughput.csv"
echo "== latency";  "$BIN/mk-bench" latency --raw "$OUT/latency_raw.csv" | tee "$OUT/latency.csv"
echo "== flush";    "$BIN/mk-bench" flush --reps "$REPS" | tee "$OUT/flush_policy.csv"
echo "== scaling";  "$BIN/mk-bench" scaling --reps "$REPS" | tee "$OUT/producer_scaling.csv"
echo "load average after run: $(sysctl -n vm.loadavg)" >> "$OUT/bench_env.txt"
