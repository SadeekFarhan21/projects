"""A small GPT used as the single-process reference for every parallel mode.

Kept deliberately plain (explicit fused qkv, untied embedding/head, no dropout) so the
tensor-parallel and pipeline versions can load exactly the same weights and be
compared parameter by parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 64
    seq_len: int = 16
    d_model: int = 32
    n_head: int = 4
    n_layer: int = 4
    d_ff: int | None = None  # defaults to 4 * d_model

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_head == 0

    def to_dict(self) -> dict:
        return asdict(self)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v: [B, H, T, hd] -> [B, H, T, hd]."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        hd = D // self.n_head
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        y = causal_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, attn: nn.Module | None = None, mlp: nn.Module | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn if attn is not None else Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = mlp if mlp is not None else MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(init_weights)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.tok_emb(idx) + self.pos_emb(pos)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, lm_loss(logits, targets)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # float() so the loss is computed in fp32 even when logits come out of autocast.
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.view(-1))


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


def build_reference(cfg: GPTConfig, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(cfg)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_per_token(cfg: GPTConfig) -> float:
    """Approximate training FLOPs per token (6N rule plus attention term)."""
    n = 12 * cfg.n_layer * cfg.d_model ** 2
    return 6 * n + 12 * cfg.n_layer * cfg.d_model * cfg.seq_len
