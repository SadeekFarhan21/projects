#!/usr/bin/env bash
# Download ~13 MB of daily Binance spot bars (300 USDT pairs, 2022-01 to 2026-08)
# from the public archive at data.binance.vision into ./data (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run qrp ingest --root data --start "${START:-2022-01}" --end "${END:-2026-08}" \
  --max-symbols "${MAX_SYMBOLS:-300}"
