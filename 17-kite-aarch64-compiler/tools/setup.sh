#!/usr/bin/env bash
# Idempotent toolchain setup for the Kite compiler.
# Installs opam (Homebrew), initialises opam without a default switch,
# creates a local switch in this directory on OCaml 5.x, and installs deps.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OCAML_VERSION="${OCAML_VERSION:-5.2.1}"

if ! command -v opam >/dev/null 2>&1; then
  echo "[setup] installing opam via Homebrew"
  /opt/homebrew/bin/brew install opam
fi

if [ ! -d "${OPAMROOT:-$HOME/.opam}" ]; then
  echo "[setup] opam init"
  opam init -y --disable-sandboxing --bare
fi

if [ ! -d "$ROOT/_opam" ]; then
  echo "[setup] creating local switch on OCaml $OCAML_VERSION"
  opam switch create "$ROOT" "ocaml-base-compiler.$OCAML_VERSION" -y --no-install
fi

eval "$(opam env --switch="$ROOT" --set-switch)"
echo "[setup] installing dune, alcotest"
opam install -y dune alcotest

echo "[setup] done. Run: eval \$(opam env --switch=$ROOT --set-switch)"
