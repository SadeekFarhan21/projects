"""Single-process encode timing on the validation split (cold vs warm chunk cache).

Usage: uv run python scripts/profile_encode.py > results/encode_profile.txt
"""

import os
import time

from tfs.bpe import EOT, BPETokenizer

tok = BPETokenizer.load("data/tokenizer.json")
stories = open("data/valid.txt", encoding="utf-8").read().split(EOT)
print(f"load average at start: {os.getloadavg()[0]:.1f}")
for label, part in [("cold cache, stories 0-3000", stories[:3000]), ("warm cache, stories 3000-6000", stories[3000:6000])]:
    t = time.perf_counter()
    n = sum(len(tok.encode(s.strip(), allow_special=False)) for s in part)
    dt = time.perf_counter() - t
    print(f"{label}: {dt:.2f} s, {n:,} tokens, {n / dt:,.0f} tokens/s, cache size {len(tok._cache):,}")
