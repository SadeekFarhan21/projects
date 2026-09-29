"""Download a bounded slice of TinyStories V2 (GPT-4 generated) into data/.

The full training file is about 2.2 GB. v0 only needs a few tens of millions of
tokens, so we fetch the first --train-mb megabytes with an HTTP Range request
and trim the slice back to the last complete story. The official validation
file (about 22 MB) is downloaded whole and used as the held-out split.

Usage: uv run python scripts/download_tinystories.py --train-mb 200
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"
EOT = "<|endoftext|>"


def fetch(url: str, max_bytes: int | None = None) -> bytes:
    headers = {"User-Agent": "tfs-download/0.1"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        return response.read()


def trim_to_last_story(raw: bytes) -> str:
    # A byte range can cut a UTF-8 character or a story in half. Cut back to the
    # last end-of-text marker so every story in the slice is complete.
    text = raw.decode("utf-8", errors="ignore")
    cut = text.rfind(EOT)
    if cut == -1:
        raise RuntimeError("slice contains no complete story")
    return text[: cut + len(EOT)] + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--train-mb", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    valid_path = args.out / "valid.txt"
    if not valid_path.exists():
        print(f"downloading {VALID_FILE}")
        valid_path.write_bytes(fetch(BASE + VALID_FILE))
    train_path = args.out / "train.txt"
    if not train_path.exists():
        print(f"downloading first {args.train_mb} MB of {TRAIN_FILE}")
        raw = fetch(BASE + TRAIN_FILE, max_bytes=args.train_mb * 1024 * 1024)
        train_path.write_text(trim_to_last_story(raw), encoding="utf-8")
    for path in (train_path, valid_path):
        text = path.read_text(encoding="utf-8")
        print(f"{path}: {len(text):,} chars, {text.count(EOT):,} stories")


if __name__ == "__main__":
    main()
