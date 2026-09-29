#!/usr/bin/env bash
# Decode one day of TOPS and DEEP in parallel (one process per feed, one core each)
# and record wall time. Usage: scripts/decode_day.sh 20260918 [extra mdp_decode args]
set -euo pipefail
DATE=${1:?date YYYYMMDD}
shift || true
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RAW=$ROOT/data/raw
WORK=$ROOT/data/work/$DATE
mkdir -p "$WORK"
start=$(date +%s)
for FEED in TOPS1.6 DEEP1.0; do
  ( /usr/bin/time -p "$ROOT/build/mdp_decode" "$RAW/${DATE}_IEXTP1_${FEED}.pcap.gz" --out "$WORK/$FEED" "$@" \
      > "$WORK/$FEED.log" 2>&1 ) &
done
wait
end=$(date +%s)
echo "{\"date\": \"$DATE\", \"wall_seconds\": $((end - start))}" > "$WORK/decode_wall.json"
cat "$WORK/decode_wall.json"
