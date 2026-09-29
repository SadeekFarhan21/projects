#!/usr/bin/env bash
# Fetch everything the task generators need into data/ (gitignored).
# Chinook SQLite, a TPC-H SF0.01 sample built with DuckDB's dbgen, and three
# small pure-Python packages pinned to exact release tags.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/packages data/dl

if [ ! -f data/chinook.sqlite ]; then
  curl -sSL -o data/chinook.sqlite \
    https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite
fi

if [ ! -f data/tpch.sqlite ]; then
  uv run python scripts/build_tpch.py data/tpch.sqlite
fi

fetch_pkg () {  # name url strip_dir
  local name=$1 url=$2
  if [ ! -d "data/packages/$name" ]; then
    curl -sSL -o "data/dl/$name.tar.gz" "$url"
    mkdir -p "data/packages/$name"
    tar -xzf "data/dl/$name.tar.gz" -C "data/packages/$name" --strip-components=1
  fi
}
fetch_pkg inflection https://github.com/jpvanhal/inflection/archive/refs/tags/0.5.1.tar.gz
fetch_pkg toolz      https://github.com/pytoolz/toolz/archive/refs/tags/1.0.0.tar.gz
fetch_pkg semver     https://github.com/python-semver/python-semver/archive/refs/tags/3.0.2.tar.gz
echo "data ready"; ls -la data data/packages
