#!/usr/bin/env bash
# Budget matched comparison: every run gets the same wall clock budget and runs
# concurrently, so all runs see the same (shared, noisy) machine load.
#   v0 in this session:   MINUTES=20 bash scripts/run_budget_matched.sh
#   full run (milestone): MINUTES=720 bash scripts/run_budget_matched.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MINUTES=${MINUTES:-20}
TAG=${TAG:-bm${MINUTES}}
mkdir -p runs results/train
for spec in unet:eps unet:v unet:rf dit:rf; do
  arch=${spec%%:*}; obj=${spec##*:}
  uv run python -m diffusion.train --arch "$arch" --objective "$obj" --minutes "$MINUTES" \
    --out "runs/${TAG}_${arch}_${obj}" > "runs/${TAG}_${arch}_${obj}.log" 2>&1 &
done
wait
for spec in unet_eps unet_v unet_rf dit_rf; do
  cp "runs/${TAG}_${spec}/log.json" "results/train/${TAG}_${spec}.json"
done
echo done
