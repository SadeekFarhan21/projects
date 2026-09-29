"""Decoder-only transformer: RMSNorm, RoPE, hand-written causal attention, SwiGLU.

Only tensor ops, nn.Linear/nn.Embedding and autograd come from PyTorch. The
attention softmax(QK^T/sqrt(d))V is spelled out; `attn_impl="sdpa"` exists
only as a reference path for tests and benchmarks.

Shapes: B batch, T new tokens this call, S total keys (cache + new),
H heads, D head dim, C = H * D model width.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 256
    ffn_hidden: int | None = None  # default: ~8/3 * d_model rounded to 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    attn_impl: str = "manual"  # "manual" (hand-written) or "sdpa" (reference)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dim must be even for RoPE")
        if self.ffn_hidden is None:
            # SwiGLU has 3 matrices instead of 2, so 8/3 * d keeps params ~= a 4d MLP.
            self.ffn_hidden = 64 * math.ceil(8 * self.d_model / 3 / 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """Upcast bf16/fp16 to fp32 for reductions, but never downcast fp64."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce in fp32 even under autocast: a bf16 mean of squares loses bits.
        xf = _at_least_fp32(x)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.to(xf.dtype)).to(x.dtype)


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim) in the rotate-half layout."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    pos = torch.arange(max_len, dtype=torch.float64)
    angles = torch.outer(pos, inv_freq)  # (max_len, D/2)
    angles = torch.cat([angles, angles], dim=-1)  # (max_len, D)
    return angles.cos().float(), angles.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[i], x[i + D/2]) by a position-dependent angle.

    x: (B, H, T, D); cos/sin: (T, D) already sliced to the absolute positions.
    Because q and k at positions m and n are rotated by m*w and n*w, their dot
    product depends only on m - n, which is what makes the KV cache valid: keys
    are rotated once when written and never need to change.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class KVCache:
    """Preallocated per-layer key/value buffers plus a shared write position.

    k[l], v[l]: (B, H, max_len, D). Invariant: slots [0, pos) hold the rotated
    keys and the values for the first `pos` tokens; slots >= pos are garbage
    and are never read because attention only looks at [0, pos + T).
    """

    def __init__(self, cfg: ModelConfig, batch: int, device, dtype, max_len: int | None = None):
        self.max_len = max_len or cfg.max_seq_len
        shape = (batch, cfg.n_heads, self.max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.pos = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        end = self.pos + t
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        self.k[layer][:, :, self.pos : end] = k
        self.v[layer][:, :, self.pos : end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)  # each (B,H,T,D)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        start = 0
        if cache is not None:
            start = cache.pos
            k, v = cache.update(self.layer_idx, k, v)  # now (B,H,S,D), S = start + T
        S = k.shape[2]

        # Query i sits at absolute position start + i and may see keys 0..start+i.
        # A single new token (T == 1) is the last position and may see every key,
        # so decode steps skip building and applying the mask entirely.
        mask = _causal_mask(T, S, start, x.device) if T > 1 else None
        if self.cfg.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))  # (B,H,T,S)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            # softmax in fp32: bf16 exp/sum over hundreds of keys is visibly lossy
            probs = torch.softmax(_at_least_fp32(scores), dim=-1).to(v.dtype)
            y = probs @ v  # (B,H,T,D)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)


def _causal_mask(T: int, S: int, start: int, device) -> torch.Tensor:
    q_pos = torch.arange(start, start + T, device=device)[:, None]
    k_pos = torch.arange(S, device=device)[None, :]
    return k_pos <= q_pos  # (T, S) bool, True = may attend


class SwiGLU(nn.Module):
    """out = W2 (silu(W1 x) * W3 x). W1 and W3 are fused into one matmul."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w13 = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    """Pre-norm residual block: x + Attn(Norm(x)), then x + MLP(Norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # GPT-2 style: shrink residual-branch outputs so the residual stream's
        # variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.out.weight") or name.endswith("mlp.w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weight counted once

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """idx: (B, T) token ids. Returns logits (B, T, vocab)."""
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        if start + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence of {start + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.pos += T  # advance once, after every layer has written its slots
        return self.lm_head(self.norm(x))

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
