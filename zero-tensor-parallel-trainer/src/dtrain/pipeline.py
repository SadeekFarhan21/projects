"""GPipe-style pipeline parallelism with microbatches, built on send/recv.

Stage s owns a contiguous range of transformer blocks; stage 0 also owns the
embeddings, the last stage owns ln_f + head + loss. The global batch is split into M
microbatches. Schedule (GPipe, Huang et al. 2019):

  time ->
  stage0  F0 F1 F2 F3                         B3 B2 B1 B0
  stage1     F0 F1 F2 F3                   B3 B2 B1 B0
  stage2        F0 F1 F2 F3             B3 B2 B1 B0
  stage3           F0 F1 F2 F3 B3 B2 B1 B0

All forwards, then all backwards. Every stage stores activations for all M
microbatches (the memory cost 1F1B fixes later). Bubble fraction = (S-1)/(M+S-1).

Wire protocol between stage s and s+1 per microbatch: forward sends the activation
[mb, T, D]; backward sends its gradient back. Shapes are static so no handshake.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from . import comm
from .model import GPTConfig, Block, lm_loss


def partition_layers(n_layer: int, n_stages: int) -> list[range]:
    """Balanced contiguous split; earlier stages get the extra layer when uneven."""
    base, extra = divmod(n_layer, n_stages)
    out, start = [], 0
    for s in range(n_stages):
        n = base + (1 if s < extra else 0)
        out.append(range(start, start + n))
        start += n
    return out


class PipelineStage(nn.Module):
    def __init__(self, cfg: GPTConfig, stage: int, n_stages: int):
        super().__init__()
        self.cfg, self.stage, self.n_stages = cfg, stage, n_stages
        self.is_first, self.is_last = stage == 0, stage == n_stages - 1
        self.layers = partition_layers(cfg.n_layer, n_stages)[stage]
        if self.is_first:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        # ModuleDict keyed by global layer index keeps names identical to GPT
        # ("blocks.5.attn.qkv.weight"), so loading from the reference is a plain filter.
        self.blocks = nn.ModuleDict({str(i): Block(cfg) for i in self.layers})
        if self.is_last:
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            pos = torch.arange(x.shape[1], device=x.device)
            x = self.tok_emb(x) + self.pos_emb(pos)
        for i in self.layers:
            x = self.blocks[str(i)](x)
        if self.is_last:
            x = self.head(self.ln_f(x))
        return x


def build_stage(ref_state: dict, cfg: GPTConfig, stage: int, n_stages: int) -> PipelineStage:
    m = PipelineStage(cfg, stage, n_stages)
    own = m.state_dict().keys()
    m.load_state_dict({k: v.clone() for k, v in ref_state.items() if k in own})
    return m


def gpipe_step(stage: PipelineStage, idx: torch.Tensor, targets: torch.Tensor,
               n_micro: int, autocast_ctx=None, loss_scale: float = 1.0) -> float:
    """One forward+backward over the global batch. Gradients accumulate into the
    stage's parameters. Returns the mean loss (valid on every rank: the last stage
    broadcasts it)."""
    rank, world = dist.get_rank(), dist.get_world_size()
    prev, nxt = rank - 1, rank + 1
    B, T = idx.shape
    assert B % n_micro == 0, "global batch must divide by number of microbatches"
    mb = B // n_micro
    D = stage.cfg.d_model
    dev = idx.device
    xs, ys = idx.split(mb), targets.split(mb)
    inputs, outputs, losses = [], [], []
    ctx = autocast_ctx or torch.autocast(dev.type, enabled=False)

    # ---- forward for every microbatch -----------------------------------------------
    for m in range(n_micro):
        if stage.is_first:
            inp = xs[m]
        else:
            inp = torch.empty(mb, T, D, device=dev)
            comm.recv(inp, prev)
            inp.requires_grad_(True)
        with ctx:
            out = stage(inp)
        if stage.is_last:
            loss = lm_loss(out, ys[m]) / n_micro
            losses.append(loss)
        else:
            comm.send(out.detach().float().contiguous(), nxt)
        inputs.append(inp)
        outputs.append(out)

    # ---- backward in reverse microbatch order ---------------------------------------
    for m in reversed(range(n_micro)):
        if stage.is_last:
            (losses[m] * loss_scale).backward()
        else:
            g = torch.empty(mb, T, D, device=dev)
            comm.recv(g, nxt)
            outputs[m].backward(g.to(outputs[m].dtype))
        if not stage.is_first:
            comm.send(inputs[m].grad.contiguous(), prev)

    # Loss is a metric, not training traffic: broadcast outside the comm counters.
    total = torch.tensor([sum(l.item() for l in losses)], device=dev)
    dist.broadcast(total, src=world - 1)
    return total.item()


def bubble_fraction(n_stages: int, n_micro: int) -> float:
    return (n_stages - 1) / (n_micro + n_stages - 1)
