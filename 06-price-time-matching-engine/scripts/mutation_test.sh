#!/usr/bin/env bash
# Mutation check for the differential fuzzer: plant a known bug in a copy of
# the engine, rebuild, and confirm `exch fuzz` catches it. A fuzzer that
# passes on buggy code is not evidence of anything.
#
# Usage: scripts/mutation_test.sh [events_per_mutant]   (default 200000)
set -u
cd "$(dirname "$0")/.."
N=${1:-200000}
WORK=build/mutant
OUT=results/mutation.txt
rm -rf "$WORK" && mkdir -p "$WORK"
: > "$OUT"

# name | sed expression applied to src/engine.cpp
MUTANTS=(
  "modify_up_keeps_priority|s/if (in.price == o.price \&\& in.qty <= o.qty) {/if (in.price == o.price) {/"
  "fok_off_by_one|s/fillable(in.side, true, price, in.qty) < in.qty/fillable(in.side, true, price, in.qty) <= in.qty/"
  "lifo_within_level|s/pool_\[oi\] = Order{id, price, qty, L.tail, kNil, side};/pool_[oi] = Order{id, price, qty, kNil, L.head, side}; if (L.head != kNil) pool_[L.head].prev = oi; else L.tail = oi; L.head = oi; L.total += qty; ++L.count; ids_.insert(id, oi); return;/"
  "no_l2_on_amend|s/touch(side, L.price, L.total);$/(void)0;/"
  "trade_at_taker_price|s/t.price = L.price;/t.price = has_limit ? limit : L.price;/"
  "post_only_allows_lock|s/crosses(in.side, in.price, ol.back().price))/crosses(in.side, in.price, ol.back().price) \&\& in.price != ol.back().price)/"
)

for m in "${MUTANTS[@]}"; do
  name=${m%%|*}
  expr=${m#*|}
  rm -rf "$WORK/tree" && mkdir -p "$WORK/tree"
  cp -R CMakeLists.txt include src tools bench tests "$WORK/tree/"
  sed -i '' -e "$expr" "$WORK/tree/src/engine.cpp"
  if cmp -s src/engine.cpp "$WORK/tree/src/engine.cpp"; then
    echo "$name: MUTATION DID NOT APPLY" | tee -a "$OUT"
    continue
  fi
  cmake -S "$WORK/tree" -B "$WORK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DEXCH_BUILD_TESTS=OFF -DEXCH_BUILD_BENCH=OFF > /dev/null
  if ! cmake --build "$WORK/build" --target exch > "$WORK/build.log" 2>&1; then
    echo "$name: BUILD FAILED" | tee -a "$OUT"
    continue
  fi
  res=$("$WORK/build/exch" fuzz --n "$N" --seed 1 --seeds 2 --check-every 100 2>&1 | grep -E "DIVERGENCE|fuzz OK|invariant" | head -1)
  if echo "$res" | grep -q "fuzz OK"; then
    echo "$name: SURVIVED ($res)" | tee -a "$OUT"
  else
    echo "$name: CAUGHT -> $res" | tee -a "$OUT"
  fi
done
