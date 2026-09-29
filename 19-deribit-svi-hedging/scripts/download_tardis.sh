#!/usr/bin/env bash
# Stream one tardis.dev free sample day (first day of a month) of the Deribit
# options_chain dataset, keep BTC and ETH coin-settled options, and reduce it
# to one row per symbol per minute. The raw file (8 to 12 GB gzip for 2025 and
# 2026 days) is never written to disk.
#
# usage: scripts/download_tardis.sh 2026-09-01 [2026-08-01 ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/minute_snap/minute_snap"
if [[ ! -x "$BIN" ]]; then
  cmake -S "$ROOT/tools/minute_snap" -B "$ROOT/build/minute_snap" -G Ninja
  cmake --build "$ROOT/build/minute_snap"
fi
mkdir -p "$ROOT/data/raw"
for day in "$@"; do
  y=${day:0:4}; m=${day:5:2}; d=${day:8:2}
  out="$ROOT/data/raw/deribit_options_chain_${y}${m}${d}_1m.csv.gz"
  if [[ -s "$out" ]]; then echo "exists: $out"; continue; fi
  url="https://datasets.tardis.dev/v1/deribit/options_chain/${y}/${m}/${d}/OPTIONS.csv.gz"
  echo "streaming $url"
  curl -sfL --retry 3 "$url" | gunzip -c | "$BIN" BTC- ETH- | gzip -1 > "$out.tmp"
  mv "$out.tmp" "$out"
  ls -lh "$out"
done
