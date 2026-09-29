#!/usr/bin/env bash
# Configure and build the C++ library, benchmark, tests and the c4core module.
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIND_DIR="$(uv run python -m pybind11 --cmakedir)"
PY_EXE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$PYBIND_DIR" -DPython_EXECUTABLE="$PY_EXE"
cmake --build build -j "${JOBS:-4}"
