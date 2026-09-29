#!/usr/bin/env bash
# Runs the latency/throughput benchmark for the three id-index variants
# (idmap = default open addressing with splitmix64, stdmap = std::unordered_map,
# identity = open addressing with an identity hash), alternating
# them across repetitions so background load hits both about equally.
# Results land in results/bench/. Machine state (load average) is recorded
# before and after each run because this machine is often shared.
#
# Usage: scripts/run_bench.sh [reps] [ops_per_scenario] [flow_events]
set -eu
cd "$(dirname "$0")/.."
REPS=${1:-3}
N=${2:-300000}
FLOW=${3:-3000000}
OUT=results/bench
mkdir -p "$OUT"
BIN=build/release
cmake --build "$BIN" --target bench_engine bench_engine_stdmap bench_engine_identity > /dev/null

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cpus, $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion)"
  echo "compiler: $(c++ --version | head -1)"
  echo "flags: -O2 -DNDEBUG (CMAKE_BUILD_TYPE=Release)"
  echo "reps=$REPS ops_per_scenario=$N flow_events=$FLOW"
} > "$OUT/machine.txt"

for r in $(seq 1 "$REPS"); do
  for v in ${VARIANTS:-idmap stdmap identity}; do
    exe=$BIN/bench_engine
    [ "$v" = stdmap ] && exe=$BIN/bench_engine_stdmap
    [ "$v" = identity ] && exe=$BIN/bench_engine_identity
    echo "rep $r $v load_before: $(sysctl -n vm.loadavg)" | tee -a "$OUT/machine.txt"
    "$exe" --variant "${v}_r${r}" --n "$N" --flow "$FLOW" --out "$OUT" | tee "$OUT/stdout_${v}_r${r}.txt"
    echo "rep $r $v load_after: $(sysctl -n vm.loadavg)" | tee -a "$OUT/machine.txt"
  done
done
