"""Logit filtering (temperature, top-k, top-p) and autoregressive generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import KVCache, Transformer


@dataclass
class SamplingConfig:
    temperature: float = 1.0  # 0 means greedy argmax
    top_k: int | None = None
    top_p: float | None = None


def filter_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Return logits with disallowed tokens set to -inf. logits: (B, V)."""
    logits = logits.float()
    if cfg.temperature > 0:
        logits = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < logits.size(-1):
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p is not None and cfg.top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Keep the smallest prefix whose mass reaches top_p: drop a token when the
        # mass strictly before it already reaches top_p. The top token always stays.
        mass_before = probs.cumsum(-1) - probs
        drop_sorted = mass_before >= cfg.top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return logits


def sample_next(logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next ids (B, 1)."""
    if cfg.temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(filter_logits(logits, cfg), dim=-1)
    # Sample on CPU so a seeded CPU generator gives the same stream on any device.
    return torch.multinomial(probs.cpu(), 1, generator=generator).to(logits.device)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    sampling: SamplingConfig = SamplingConfig(temperature=0.0),
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    stop_id: int | None = None,
    return_logits: bool = False,
):
    """Autoregressive decoding. prompt: (B, T0) ids on the model's device.

    use_cache=False recomputes the full prefix every step (O(n^2) total work);
    use_cache=True prefills once, then feeds one token per step. Both paths must
    produce the same logits; tests/test_kv_cache.py checks that.
    Total length must fit in max_seq_len (no sliding window in v0).
    """
    model.eval()
    B, T0 = prompt.shape
    total = T0 + max_new_tokens
    if total > model.cfg.max_seq_len:
        raise ValueError(f"prompt + new tokens = {total} > max_seq_len {model.cfg.max_seq_len}")
    dtype = next(model.parameters()).dtype
    cache = KVCache(model.cfg, B, prompt.device, dtype, max_len=total) if use_cache else None
    seq = prompt
    step_logits = []
    logits = model(prompt, cache)[:, -1]  # prefill
    for _ in range(max_new_tokens):
        step_logits.append(logits)
        nxt = sample_next(logits, sampling, generator)
        seq = torch.cat([seq, nxt], dim=1)
        if stop_id is not None and B == 1 and nxt.item() == stop_id:
            break
        if seq.shape[1] == total:
            break
        if use_cache:
            logits = model(nxt, cache)[:, -1]
        else:
            logits = model(seq)[:, -1]
    if return_logits:
        return seq, torch.stack(step_logits, dim=1)
    return seq
