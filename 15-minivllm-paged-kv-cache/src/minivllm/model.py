"""Qwen2 decoder written from scratch, reading and writing a paged KV cache.

The forward pass takes a *flattened* batch: the new tokens of every scheduled
sequence concatenated into one 1-D tensor. Linear layers (the bulk of the FLOPs)
run once over all tokens; only attention needs per-sequence structure, which
comes from `AttnMetadata`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .kv_cache import KVCache
from .weights import load_state_dict


@dataclass
class AttnMetadata:
    """Per-step attention layout, built once per step and shared by all layers.

    slot_mapping: [T] physical cache slot for each new token (where to write K/V).
    Prefill steps (every sequence starts from an empty cache in v0):
      seq_lens: python list of new-token counts per sequence, sums to T.
    Decode steps (exactly one new token per sequence, T == B):
      gather_slots: [B, Lmax] cache slots of each sequence's full context, padded.
      mask:         [B, 1, 1, Lmax] True where the key is a real context token.
    """

    is_prefill: bool
    slot_mapping: torch.Tensor
    seq_lens: list[int] | None = None
    gather_slots: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables, indexed by absolute position (non-interleaved RoPE)."""

    def __init__(self, head_dim: int, max_pos: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float64)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos().float(), persistent=False)
        self.register_buffer("sin", emb.sin().float(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [T, heads, head_dim]
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        xf = x.float()
        half = xf.shape[-1] // 2
        rot = torch.cat([-xf[..., half:], xf[..., :half]], dim=-1)
        return (xf * cos + rot * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        # q, k, v fused into one matmul: fewer kernel launches per layer.
        self.qkv_proj = nn.Linear(cfg.hidden_size, (self.nh + 2 * self.nkv) * self.hd, bias=True)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.scale = self.hd**-0.5

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        cache: KVCache,
        meta: AttnMetadata,
    ) -> torch.Tensor:
        T = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = rope(q.view(T, self.nh, self.hd), positions)
        k = rope(k.view(T, self.nkv, self.hd), positions)
        v = v.view(T, self.nkv, self.hd)

        k_cache, v_cache = cache.layer(self.layer_idx)
        k_cache.index_copy_(0, meta.slot_mapping, k)
        v_cache.index_copy_(0, meta.slot_mapping, v)

        if meta.is_prefill:
            # Every prefilled sequence starts from an empty cache, so its own
            # new K/V are its whole context: plain causal attention per sequence.
            outs = []
            start = 0
            for n in meta.seq_lens:
                qs = q[start : start + n].transpose(0, 1)  # [nh, n, hd]
                ks = k[start : start + n].transpose(0, 1)
                vs = v[start : start + n].transpose(0, 1)
                o = F.scaled_dot_product_attention(
                    qs.unsqueeze(0), ks.unsqueeze(0), vs.unsqueeze(0),
                    is_causal=True, scale=self.scale, enable_gqa=True,
                )
                outs.append(o[0].transpose(0, 1))
                start += n
            out = torch.cat(outs, dim=0)
        else:
            # Decode: gather each sequence's context from its (non-contiguous)
            # blocks, pad to the longest, and mask out padding.
            B, L = meta.gather_slots.shape
            ks = k_cache[meta.gather_slots.view(-1)].view(B, L, self.nkv, self.hd).transpose(1, 2)
            vs = v_cache[meta.gather_slots.view(-1)].view(B, L, self.nkv, self.hd).transpose(1, 2)
            qs = q.view(B, 1, self.nh, self.hd).transpose(1, 2)  # [B, nh, 1, hd]
            o = F.scaled_dot_product_attention(
                qs, ks, vs, attn_mask=meta.mask, scale=self.scale, enable_gqa=True
            )
            out = o.transpose(1, 2).reshape(B, self.nh, self.hd)
        return self.o_proj(out.reshape(T, self.nh * self.hd))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inter = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.inter, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, positions, rope, cache, meta):
        x = x + self.self_attn(self.input_layernorm(x), positions, rope, cache, meta)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig, max_pos: int = 4096):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(cfg.head_dim, max_pos, cfg.rope_theta)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, cache: KVCache, meta: AttnMetadata
    ) -> torch.Tensor:
        """Returns final hidden states [T, hidden]. Call compute_logits on the rows you need."""
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, self.rope, cache, meta)
        return self.norm(x)

    @torch.inference_mode()
    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        max_pos: int = 4096,
    ) -> "Qwen2Model":
        model_dir = Path(model_dir)
        cfg = ModelConfig.from_json(model_dir / "config.json")
        with torch.device("meta"):
            model = cls(cfg, max_pos=max_pos)
        sd = load_state_dict(model_dir)
        model.load_hf_state_dict(sd, dtype=dtype, device=device)
        return model

    def load_hf_state_dict(self, sd: dict[str, torch.Tensor], dtype, device) -> None:
        """Map HF Qwen2 names to ours, fusing q/k/v and gate/up."""
        cfg = self.cfg
        new: dict[str, torch.Tensor] = {}
        g = lambda k: sd[k].to(dtype)  # noqa: E731
        new["embed_tokens.weight"] = g("model.embed_tokens.weight")
        new["norm.weight"] = g("model.norm.weight")
        if not cfg.tie_word_embeddings:
            new["lm_head.weight"] = g("lm_head.weight")
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            new[f"layers.{i}.input_layernorm.weight"] = g(p + "input_layernorm.weight")
            new[f"layers.{i}.post_attention_layernorm.weight"] = g(p + "post_attention_layernorm.weight")
            new[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
                [g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")]
            )
            new[f"layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(
                [g(a + "q_proj.bias"), g(a + "k_proj.bias"), g(a + "v_proj.bias")]
            )
            new[f"layers.{i}.self_attn.o_proj.weight"] = g(a + "o_proj.weight")
            m = p + "mlp."
            new[f"layers.{i}.mlp.gate_up_proj.weight"] = torch.cat(
                [g(m + "gate_proj.weight"), g(m + "up_proj.weight")]
            )
            new[f"layers.{i}.mlp.down_proj.weight"] = g(m + "down_proj.weight")
        # assign=True swaps the meta tensors for real ones without a copy.
        missing, unexpected = self.load_state_dict(new, strict=False, assign=True)
        missing = [k for k in missing if not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
        assert not missing and not unexpected, (missing, unexpected)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        # Non-persistent RoPE buffers were created on meta; rebuild them for real.
        self.rope = RotaryEmbedding(cfg.head_dim, self.rope.cos.shape[0], cfg.rope_theta)
        self.to(device)
        self.eval()

    @classmethod
    def random_init(cls, cfg: ModelConfig, seed: int = 0, dtype=torch.float32, device="cpu", max_pos=1024):
        """Tiny random model for fast scheduler/cache tests."""
        torch.manual_seed(seed)
        model = cls(cfg, max_pos=max_pos)
        for name, p in model.named_parameters():
            if "norm" in name:
                nn.init.ones_(p)
            else:
                nn.init.normal_(p, std=0.05)
        return model.to(device=device, dtype=dtype).eval()
