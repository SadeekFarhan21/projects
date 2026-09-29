#!/bin/bash
set -e

# Ensure we're in the project root
cd "$(dirname "$0")"

# Use Quarto from PATH if available; otherwise install via build.sh
if ! command -v quarto &>/dev/null; then
  echo "Quarto not found in PATH. Running build.sh to install..."
  bash build.sh
  export PATH="$HOME/quarto/bin:$PATH"
else
  echo "Using Quarto from PATH: $(which quarto)"
  quarto --version
  echo "Rendering Quarto site..."
  quarto render --to html
fi

# Deploy to Netlify if token is set (e.g. export NETLIFY_AUTH_TOKEN=...)
if [ -n "${NETLIFY_AUTH_TOKEN}" ]; then
  if [ ! -f _publish.yml ] || grep -q "PASTE_YOUR_NETLIFY_SITE_ID" _publish.yml 2>/dev/null; then
    echo "First-time Netlify setup: run once interactively to create/link your site (then _publish.yml will be filled):"
    echo "  quarto publish netlify"
    echo "After that, re-run this script to publish with --no-prompt."
    exit 1
  fi
  echo "Publishing to Netlify..."
  quarto publish netlify --no-prompt --no-browser
  echo "Published. Open your site URL in Netlify."
else
  echo "Done. Commit and push for Netlify to build, or set NETLIFY_AUTH_TOKEN and re-run to publish from here."
fi
