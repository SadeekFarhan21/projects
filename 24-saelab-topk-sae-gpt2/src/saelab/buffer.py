"""Shuffle buffer that streams residual activations from GPT-2 into SAE batches.

Nothing is written to disk. The buffer holds `capacity` token activations per
layer (fp16, on the model device). When the unread part falls to half, the
buffer runs GPT-2 on fresh token windows to refill the consumed half and then
permutes the whole buffer, so each training batch mixes tokens from many
documents and many refill rounds. Position 0 (BOS) is dropped: its residual
norm is tens of times larger than any other token and it carries no
information about the text.

Invariants (checked in tests/test_buffer.py):
  * every activation handed out is read exactly once
  * a batch never contains a BOS-position activation
  * all layers are sliced with the same permutation, so row i of every layer
    comes from the same token
"""
from __future__ import annotations

from typing import Callable, Iterator

import torch


class ActivationBuffer:
    def __init__(self, token_batches: Iterator[torch.Tensor],
                 harvest_fn: Callable[[torch.Tensor], dict[int, torch.Tensor]],
                 layers: tuple[int, ...], d_model: int, capacity: int,
                 batch_size: int, device: str, dtype=torch.float16, seed: int = 0):
        self.tokens = token_batches
        self.harvest_fn = harvest_fn
        self.layers = layers
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.store = {L: torch.empty(capacity, d_model, dtype=dtype, device=device)
                      for L in layers}
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.n_valid = 0     # filled rows [0, n_valid)
        self.read = 0        # rows [0, read) already handed out
        self.tokens_seen = 0  # activations produced by GPT-2 (non-BOS)
        self.exhausted = False
        self._fill(0)

    def _fill(self, start: int) -> None:
        """Write fresh activations into rows [start, capacity)."""
        pos = start
        while pos < self.capacity:
            try:
                toks = next(self.tokens)
            except StopIteration:
                self.exhausted = True
                break
            acts = self.harvest_fn(toks)
            n = None
            for L in self.layers:
                a = acts[L][:, 1:].reshape(-1, acts[L].shape[-1])  # drop BOS
                n = min(a.shape[0], self.capacity - pos)
                self.store[L][pos:pos + n] = a[:n].to(self.store[L].dtype)
            pos += n
            self.tokens_seen += n
        self.n_valid = pos
        perm = torch.randperm(self.n_valid, generator=self.gen).to(self.device)
        for L in self.layers:
            self.store[L][: self.n_valid] = self.store[L][perm]
        self.read = 0

    def next_batch(self) -> dict[int, torch.Tensor] | None:
        """Next batch as float32 {layer: [batch_size, d_model]}, or None when done."""
        if self.n_valid - self.read < self.capacity // 2 and not self.exhausted:
            # move unread rows to the front, refill the rest, reshuffle all
            left = self.n_valid - self.read
            for L in self.layers:
                self.store[L][:left] = self.store[L][self.read:self.n_valid].clone()
            self._fill(left)
        if self.n_valid - self.read < self.batch_size:
            return None
        s = slice(self.read, self.read + self.batch_size)
        self.read += self.batch_size
        return {L: self.store[L][s].float() for L in self.layers}
