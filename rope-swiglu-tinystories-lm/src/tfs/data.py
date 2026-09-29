"""Corpus encoding to flat uint16 token files, and random-window batching."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .bpe import EOT, BPETokenizer

_worker_tok: BPETokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _worker_tok
    _worker_tok = BPETokenizer.load(tok_path)


def _encode_stories(stories: list[str]) -> np.ndarray:
    assert _worker_tok is not None
    eot = _worker_tok.eot_id
    ids: list[int] = []
    for s in stories:
        s = s.strip()
        if s:
            ids.extend(_worker_tok.encode(s, allow_special=False))
            ids.append(eot)  # every story ends with <|endoftext|>
    return np.asarray(ids, dtype=np.uint16)


def encode_file(
    text_path: Path, out_path: Path, tok_path: Path, workers: int | None = None
) -> int:
    """Encode a TinyStories-style file (stories separated by EOT) to uint16 ids."""
    tok = BPETokenizer.load(tok_path)
    if tok.vocab_size > 65536:
        raise ValueError("uint16 token files need vocab_size <= 65536")
    stories = text_path.read_text(encoding="utf-8").split(EOT)
    shard = 2000
    shards = [stories[i : i + shard] for i in range(0, len(stories), shard)]
    workers = workers or os.cpu_count() or 1
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(tok_path),)) as ex:
        parts = list(ex.map(_encode_stories, shards))  # map preserves order
    arr = np.concatenate(parts)
    arr.tofile(out_path)
    return int(arr.size)


class TokenDataset:
    """A flat token stream on disk; batches are random contiguous windows."""

    def __init__(self, path: str | Path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return int(self.tokens.size)

    def get_batch(
        self, batch_size: int, seq_len: int, generator: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Window of seq_len + 1 tokens: inputs are [:-1], targets are [1:].
        starts = generator.integers(0, len(self) - seq_len - 1, size=batch_size)
        idx = starts[:, None] + np.arange(seq_len + 1)[None, :]
        window = torch.from_numpy(self.tokens[idx].astype(np.int64))
        return window[:, :-1], window[:, 1:]
