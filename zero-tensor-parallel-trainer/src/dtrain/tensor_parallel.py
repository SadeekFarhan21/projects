"""Megatron-style tensor parallelism for the MLP and attention blocks.

Notation (Shoeybi et al. 2019): with T ranks in the TP group,

  ColumnParallelLinear   W = [W_0; W_1; ...]   (split output features)
      y_i = x W_i^T        each rank produces a slice of the output features
  RowParallelLinear      W = [W_0, W_1, ...]   (split input features)
      y = sum_i x_i W_i^T  partial sums, combined with one all_reduce

  MLP:        x -> f -> Column(fc1) -> GeLU -> Row(fc2) -> g -> y
  Attention:  x -> f -> Column(qkv, split by heads) -> local SDPA -> Row(proj) -> g -> y

  f: identity in forward, all_reduce of the gradient in backward
  g: all_reduce in forward, identity in backward

So each transformer block costs 2 all_reduces of [B, T, D] in forward and 2 in backward,
and GeLU / softmax need no communication because they act on local slices.

Replicated parameters (embeddings, LayerNorms, head, row-parallel biases) see
identical inputs and identical gradients on every TP rank, so they stay in sync
without any extra communication. A test checks that invariant.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import comm
from .model import GPT, GPTConfig, Block, causal_attention, init_weights


class _CopyToTP(torch.autograd.Function):
    """f: forward identity, backward all_reduce."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        comm.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    """g: forward all_reduce, backward identity."""

    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous().clone()
        comm.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def copy_to_tp(x, group=None):
    return _CopyToTP.apply(x, group)


def reduce_from_tp(x, group=None):
    return _ReduceFromTP.apply(x, group)


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, tp: int, bias: bool = True, group=None):
        super().__init__()
        assert out_f % tp == 0
        self.group = group
        self.weight = nn.Parameter(torch.empty(out_f // tp, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f // tp)) if bias else None

    def forward(self, x):
        # The caller applies copy_to_tp once per block input (shared by q/k/v).
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, tp: int, bias: bool = True, group=None):
        super().__init__()
        assert in_f % tp == 0
        self.group = group
        self.weight = nn.Parameter(torch.empty(out_f, in_f // tp))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x_shard):
        y = reduce_from_tp(F.linear(x_shard, self.weight), self.group)
        # Bias added after the reduction, otherwise it would be counted T times.
        return y + self.bias if self.bias is not None else y


class TPMLP(nn.Module):
    def __init__(self, cfg: GPTConfig, tp: int, group=None):
        super().__init__()
        self.group = group
        self.fc1 = ColumnParallelLinear(cfg.d_model, cfg.d_ff, tp, group=group)
        self.fc2 = RowParallelLinear(cfg.d_ff, cfg.d_model, tp, group=group)

    def forward(self, x):
        x = copy_to_tp(x, self.group)
        return self.fc2(F.gelu(self.fc1(x)))


class TPAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, tp: int, group=None):
        super().__init__()
        assert cfg.n_head % tp == 0, "heads must divide by TP size"
        self.group = group
        self.local_heads = cfg.n_head // tp
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = ColumnParallelLinear(cfg.d_model, 3 * cfg.d_model, tp, group=group)
        self.proj = RowParallelLinear(cfg.d_model, cfg.d_model, tp, group=group)

    def forward(self, x):
        B, T, _ = x.shape
        x = copy_to_tp(x, self.group)
        local = self.local_heads * self.hd
        q, k, v = self.qkv(x).split(local, dim=-1)
        q, k, v = (t.view(B, T, self.local_heads, self.hd).transpose(1, 2) for t in (q, k, v))
        y = causal_attention(q, k, v).transpose(1, 2).reshape(B, T, local)
        return self.proj(y)


class TPGPT(GPT):
    """GPT whose blocks use TP attention and MLP. Parameter names match GPT exactly;
    only the shapes of the sharded ones differ."""

    def __init__(self, cfg: GPTConfig, tp: int, group=None):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.tp = tp
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg, attn=TPAttention(cfg, tp, group), mlp=TPMLP(cfg, tp, group))
            for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(init_weights)


# --------------------------------------------------------------------------------------
# Sharding helpers: full reference state dict  <->  per-rank TP state dicts.
# qkv is [3D, D] = [Wq; Wk; Wv]; rank r takes heads r*H/T..(r+1)*H/T of each of q, k, v.
# --------------------------------------------------------------------------------------

def _kind(name: str) -> str:
    if ".attn.qkv." in name:
        return "qkv"
    if ".mlp.fc1." in name:
        return "col"
    if name.endswith(".attn.proj.weight") or name.endswith(".mlp.fc2.weight"):
        return "row"
    return "rep"


def shard_state_dict(full: dict, rank: int, tp: int) -> dict:
    out = {}
    for name, t in full.items():
        kind = _kind(name)
        if kind == "qkv":
            out[name] = torch.cat([part.chunk(tp, dim=0)[rank] for part in t.chunk(3, dim=0)]).clone()
        elif kind == "col":
            out[name] = t.chunk(tp, dim=0)[rank].clone()
        elif kind == "row":
            out[name] = t.chunk(tp, dim=1)[rank].clone()
        else:
            out[name] = t.clone()
    return out


def merge_state_dicts(shards: list[dict]) -> dict:
    tp = len(shards)
    out = {}
    for name in shards[0]:
        kind = _kind(name)
        ts = [s[name] for s in shards]
        if kind == "qkv":
            per = [t.chunk(3, dim=0) for t in ts]  # per[r] = (q_r, k_r, v_r)
            out[name] = torch.cat([torch.cat([per[r][j] for r in range(tp)]) for j in range(3)])
        elif kind == "col":
            out[name] = torch.cat(ts, dim=0)
        elif kind == "row":
            out[name] = torch.cat(ts, dim=1)
        else:
            out[name] = ts[0]
    return out


def build_tp_model(ref_state: dict, cfg: GPTConfig, group=None) -> TPGPT:
    tp, rank = dist.get_world_size(group), dist.get_rank(group)
    model = TPGPT(cfg, tp, group)
    model.load_state_dict(shard_state_dict(ref_state, rank, tp))
    return model
