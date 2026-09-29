#!/usr/bin/env bash
# Invariant fuzzing: every rule invariant is checked after every single action.
# Run 1: 2,000,000 games with chaos seats and scripted bots (no Bayesian seats).
# Run 2: 20,000 games that also seat the Bayesian bot (its exact posterior
#        makes games about 100x slower, so the sample is smaller).
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS="${THREADS:-2}"
./build/figgie_fuzz "${GAMES:-2000000}" 1 "$THREADS" results/fuzz_scripted.json 0
./build/figgie_fuzz "${BAYES_GAMES:-20000}" 2 "$THREADS" results/fuzz_with_bayes.json 1
