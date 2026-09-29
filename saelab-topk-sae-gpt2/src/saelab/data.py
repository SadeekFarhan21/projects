"""Token streams from one OpenWebText parquet shard.

The shard has ~100 row groups of ~1000 documents. Row groups are split once,
by index, so the three uses never see the same documents:

  train      row groups 0..89    SAE training stream
  interp     row groups 90..97   auto-interp harvesting and attribution
  eval       row groups 98..100  fixed held-out eval tokens

Documents are tokenized with the GPT-2 tokenizer, joined with <|endoftext|>,
and cut into windows of ctx-1 tokens with a BOS (also <|endoftext|>, id 50256)
prepended, which is what TransformerLens does with prepend_bos=True.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
SHARD = ROOT / "data" / "owt" / "plain_text" / "train" / "0000.parquet"
BOS = 50256

SPLITS = {"train": range(0, 90), "interp": range(90, 98), "eval": range(98, 101)}


def _tokenizer():
    from transformers import AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()  # silence the "longer than 1024 tokens" warning
    return AutoTokenizer.from_pretrained("gpt2")


def iter_docs(split: str, shard: Path = SHARD) -> Iterator[str]:
    f = pq.ParquetFile(shard)
    for rg in SPLITS[split]:
        if rg >= f.num_row_groups:
            break
        for t in f.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            yield t


def chunk_tokens(token_lists: Iterator[list[int]], ctx: int) -> Iterator[np.ndarray]:
    """Concatenate docs (EOS separated) and yield [BOS] + (ctx-1) token windows."""
    buf: list[int] = []
    body = ctx - 1
    for toks in token_lists:
        buf.extend(toks)
        buf.append(BOS)
        while len(buf) >= body:
            yield np.array([BOS] + buf[:body], dtype=np.int64)
            buf = buf[body:]


def iter_batches(split: str, ctx: int = 128, batch: int = 64, tok_batch_docs: int = 256,
                 shard: Path = SHARD) -> Iterator[torch.Tensor]:
    """Yield LongTensor batches [batch, ctx] of token windows, one pass over the split."""
    tok = _tokenizer()

    def token_lists():
        docs: list[str] = []
        for d in iter_docs(split, shard):
            docs.append(d)
            if len(docs) == tok_batch_docs:
                yield from tok(docs)["input_ids"]
                docs = []
        if docs:
            yield from tok(docs)["input_ids"]

    rows: list[np.ndarray] = []
    for w in chunk_tokens(token_lists(), ctx):
        rows.append(w)
        if len(rows) == batch:
            yield torch.from_numpy(np.stack(rows))
            rows = []


def fixed_tokens(split: str, n_seqs: int, ctx: int = 128) -> torch.Tensor:
    """The first n_seqs windows of a split, as one tensor. Deterministic."""
    out = []
    for b in iter_batches(split, ctx=ctx, batch=64):
        out.append(b)
        if sum(x.shape[0] for x in out) >= n_seqs:
            break
    t = torch.cat(out)[:n_seqs]
    if t.shape[0] < n_seqs:
        raise ValueError(f"split {split} has only {t.shape[0]} windows")
    return t
