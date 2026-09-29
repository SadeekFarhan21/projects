"""Train the BPE tokenizer on a prefix of the training text, then encode both splits.

Usage: uv run python scripts/prepare_data.py --vocab-size 4096 --tokenizer-mb 20
Writes data/tokenizer.json, data/train.bin, data/valid.bin and
results/tokenizer_stats.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tfs.bpe import EOT, BPETokenizer
from tfs.data import encode_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--tokenizer-mb", type=float, default=20.0)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    train_txt = args.data / "train.txt"
    with train_txt.open(encoding="utf-8") as f:
        sample = f.read(int(args.tokenizer_mb * 1024 * 1024))
    t0 = time.perf_counter()
    tok = BPETokenizer.train(sample.split(EOT), args.vocab_size, [EOT], verbose=True)
    train_s = time.perf_counter() - t0
    tok_path = args.data / "tokenizer.json"
    tok.save(tok_path)
    print(f"tokenizer: {len(tok.merges)} merges, vocab {tok.vocab_size}, {train_s:.1f}s")

    stats = {
        "vocab_size": tok.vocab_size,
        "merges": len(tok.merges),
        "tokenizer_train_chars": len(sample),
        "tokenizer_train_seconds": round(train_s, 2),
        "longest_tokens": sorted(
            (tok.vocab[i].decode("utf-8", "replace") for i in range(256, 256 + len(tok.merges))),
            key=len,
        )[-15:],
        "first_merges": [tok.vocab[256 + i].decode("utf-8", "replace") for i in range(20)],
    }
    for split in ("train", "valid"):
        src = args.data / f"{split}.txt"
        t0 = time.perf_counter()
        n = encode_file(src, args.data / f"{split}.bin", tok_path)
        dt = time.perf_counter() - t0
        n_bytes = src.stat().st_size
        stats[split] = {
            "tokens": n,
            "bytes": n_bytes,
            "bytes_per_token": round(n_bytes / n, 3),
            "encode_seconds": round(dt, 1),
            "encode_mb_per_s": round(n_bytes / dt / 1e6, 2),
        }
        print(f"{split}: {n:,} tokens, {n_bytes / n:.2f} bytes/token, {dt:.1f}s")
    args.results.mkdir(exist_ok=True)
    (args.results / "tokenizer_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
