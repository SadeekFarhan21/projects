#!/usr/bin/env bash
# Build the Objective-C++ / pybind11 Metal runtime into src/tcm/_metal*.so
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$(uv run python -c 'import sys; print(sys.executable)')"
PB="$(uv run python -m pybind11 --cmakedir)"
cmake -S runtime -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY" -Dpybind11_DIR="$PB"
cmake --build build
ls -1 src/tcm/_metal*.so
