#!/usr/bin/env bash
# Download BEIR datasets (public, hosted by UKP TU Darmstadt) into data/.
# scifact (2.8 MB) and nfcorpus (2.4 MB) are used for ranking quality,
# quora (15.9 MB zip, 523k short docs) is used for scale and latency.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
BASE=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets
for ds in "${@:-scifact nfcorpus quora}"; do
  for d in $ds; do
    if [ -f "data/$d/corpus.jsonl" ]; then echo "data/$d already present"; continue; fi
    curl -fL --retry 3 -o "data/$d.zip" "$BASE/$d.zip"
    unzip -q -o "data/$d.zip" -d data
    rm "data/$d.zip"
    echo "downloaded data/$d"
  done
done
