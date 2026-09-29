#!/bin/sh
# Re-render every diagram with the site-font renderer; fails if any diagram
# uses a font other than Quicksand or Fragment Mono.
#   PY=<excalidraw skill venv>/bin/python sh tools/excalidraw/render-all.sh
set -e
cd "$(dirname "$0")/../.."
: "${PY:=$HOME/.claude/skills/excalidraw-diagram/references/.venv/bin/python}"
for f in diagrams/*.excalidraw; do
  n=$(basename "$f" .excalidraw)
  "$PY" tools/excalidraw/render_excalidraw.py "$f" --output "diagrams/png/$n.png" --scale 2 >/dev/null
  echo "rendered $n"
done
