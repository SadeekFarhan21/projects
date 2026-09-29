"""Deterministic data so every rank (and the single-process reference) sees the same
global batch at step s. Each parallel mode then takes its own slice.

Two sources:
  synthetic : sequences from a fixed sparse Markov chain (learnable, no download).
  shakespeare : char-level tiny-shakespeare, fetched by scripts/download_data.sh.
"""

from __future__ import annotations

import os

import torch


class SyntheticMarkov:
    """Each token has 3 likely successors; the best achievable loss is well below
    log(vocab), so training curves actually move."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        logits = torch.full((vocab_size, vocab_size), -4.0)
        succ = torch.randint(0, vocab_size, (vocab_size, 3), generator=g)
        logits.scatter_(1, succ, torch.tensor([3.0, 2.0, 1.0]).expand(vocab_size, 3))
        self.probs = logits.softmax(-1)
        self.vocab_size, self.seq_len, self.seed = vocab_size, seq_len, seed

    def batch(self, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        x = torch.empty(batch_size, self.seq_len + 1, dtype=torch.long)
        x[:, 0] = torch.randint(0, self.vocab_size, (batch_size,), generator=g)
        for t in range(1, self.seq_len + 1):
            x[:, t] = torch.multinomial(self.probs[x[:, t - 1]], 1, generator=g).squeeze(1)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()


class CharText:
    def __init__(self, path: str, seq_len: int, seed: int = 1234):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len, self.seed = seq_len, seed

    def batch(self, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,), generator=g)
        x = torch.stack([self.data[i:i + self.seq_len] for i in ix.tolist()])
        y = torch.stack([self.data[i + 1:i + 1 + self.seq_len] for i in ix.tolist()])
        return x, y


def shakespeare_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "data", "tinyshakespeare.txt"))


def shard_batch(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Contiguous slice of the global batch for data-parallel rank `rank`."""
    assert x.shape[0] % world_size == 0, "global batch must divide by DP world size"
    per = x.shape[0] // world_size
    return x[rank * per:(rank + 1) * per]
