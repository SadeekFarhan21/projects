#!/usr/bin/env bash
# Download the Mip-NeRF 360 "garden" scene (quarter resolution images + COLMAP sparse model)
# from a public Hugging Face mirror. About 260 MB. Data lands in data/garden (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/garden"
BASE="https://huggingface.co/datasets/mileleap/mipnerf360/resolve/main/garden"
API="https://huggingface.co/api/datasets/mileleap/mipnerf360/tree/main/garden"
RES="${RES:-4}"   # 4 = quarter resolution (1297x840), 8 = eighth
mkdir -p "$DEST/sparse/0" "$DEST/images_$RES"
for f in cameras.bin images.bin points3D.bin; do
  [ -s "$DEST/sparse/0/$f" ] || curl -sSfL "$BASE/sparse/0/$f" -o "$DEST/sparse/0/$f"
done
curl -sSf "$API/images_$RES" | python3 -c "import json,sys; [print(x['path'].split('/')[-1]) for x in json.load(sys.stdin)]" > "$DEST/.filelist_$RES"
# 8 parallel downloads, skip files already present
< "$DEST/.filelist_$RES" xargs -P 8 -I{} sh -c '[ -s "$0/{}" ] || curl -sSfL "$1/{}" -o "$0/{}"' "$DEST/images_$RES" "$BASE/images_$RES"
echo "images: $(ls "$DEST/images_$RES" | wc -l)  in $DEST/images_$RES"
