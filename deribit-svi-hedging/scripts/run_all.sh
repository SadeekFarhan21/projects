#!/usr/bin/env bash
# Run the test suite and every experiment in order, one at a time, with thread
# counts capped (the machine is shared). Outputs land in results/.
# usage: scripts/run_all.sh [day]   (default 2026-09-01; data must be downloaded)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2}
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-2}
DAY=${1:-2026-09-01}
uv run pytest -q
uv run python experiments/01_iv_benchmark.py | tee results/01_iv_benchmark.log
uv run python -W ignore::RuntimeWarning experiments/02_surface_fits.py "$DAY" | tee results/02_surface_fits.log
uv run python -W ignore::RuntimeWarning experiments/03_mark_iv_compare.py "$DAY" | tee results/03_mark_iv_compare.log
uv run python -W ignore::RuntimeWarning experiments/04_hedging.py "$DAY" | tee results/04_hedging.log
