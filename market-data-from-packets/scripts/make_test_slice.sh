#!/usr/bin/env bash
# Cut the checked-in end-to-end fixture out of the 20260918 HIST captures:
# every packet captured before 2026-09-18 11:25:00 UTC (07:25 ET), both feeds,
# rewritten as classic nanosecond pcap and gzipped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNTIL=1789730700000000000
mkdir -p "$ROOT/testdata"
for pair in TOPS1.6:tops DEEP1.0:deep; do
  FEED=${pair%%:*}; NAME=${pair##*:}
  "$ROOT/build/mdp_slice" "$ROOT/data/raw/20260918_IEXTP1_${FEED}.pcap.gz" "$ROOT/testdata/slice_${NAME}.pcap" --until-ns $UNTIL
  gzip -9 -n -f "$ROOT/testdata/slice_${NAME}.pcap"
done
ls -la "$ROOT/testdata"
