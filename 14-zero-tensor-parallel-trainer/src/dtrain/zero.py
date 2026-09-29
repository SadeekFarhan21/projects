"""ZeRO stage 1: shard the optimizer state across data-parallel ranks.

Layout: all trainable parameters are viewed as one flat fp32 vector of length L,
padded to P = ceil(L / N) * N. Rank r owns the slice [r*S, (r+1)*S) with S = P / N.

Per step:
  1. flatten local grads  ->  g        (length P, zero padded)
  2. reduce_scatter(g)    ->  g_r      (sum of everyone's grads, only my slice)
  3. AdamW on my slice of the master weights using my slice of (m, v)
  4. all_gather(w_r)      ->  w        (everyone gets the full updated weights)

reduce_scatter + all_gather move the same bytes as one all_reduce, so ZeRO-1 costs no
extra communication over plain DP but divides Adam's 2 x fp32 state by N.

We reuse torch.optim.AdamW on a single nn.Parameter holding the shard. AdamW is
elementwise, so a shard update is bit-for-bit the corresponding slice of the full update.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn

from . import comm


class ZeroAdamW:
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, group=None, grad_clip: float | None = None):
        self.params = [p for p in params if p.requires_grad]
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.grad_clip = grad_clip
        self.numel = sum(p.numel() for p in self.params)
        self.padded = math.ceil(self.numel / self.world) * self.world
        self.shard_size = self.padded // self.world
        lo = self.rank * self.shard_size
        self.lo, self.hi = lo, lo + self.shard_size

        with torch.no_grad():
            full = self._flatten([p.data for p in self.params])
        self.master = nn.Parameter(full[self.lo:self.hi].clone().float())
        self.opt = torch.optim.AdamW([self.master], lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay)
        self.last_grad_norm: float | None = None

    def _flatten(self, tensors) -> torch.Tensor:
        flat = torch.cat([t.reshape(-1).float() for t in tensors])
        if self.padded > self.numel:
            flat = torch.cat([flat, flat.new_zeros(self.padded - self.numel)])
        return flat

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, loss_scale: float = 1.0) -> bool:
        """Returns True if the step was skipped because of inf/nan gradients."""
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        g = self._flatten(grads)
        g_shard = torch.empty(self.shard_size, dtype=g.dtype, device=g.device)
        comm.reduce_scatter(g_shard, g, group=self.group)
        g_shard.div_(self.world * loss_scale)

        # Overflow check and grad-norm both need a global view: tiny all_reduces.
        stats = torch.stack([(~torch.isfinite(g_shard)).any().float(), g_shard.pow(2).sum()])
        comm.all_reduce(stats, group=self.group)
        if stats[0].item() > 0:
            return True
        norm = stats[1].sqrt().item()
        self.last_grad_norm = norm
        if self.grad_clip is not None and norm > self.grad_clip:
            g_shard.mul_(self.grad_clip / (norm + 1e-6))

        self.master.grad = g_shard
        self.opt.step()
        self.master.grad = None

        full = torch.empty(self.padded, dtype=self.master.dtype, device=self.master.device)
        comm.all_gather(full, self.master.data, group=self.group)
        off = 0
        for p in self.params:
            n = p.numel()
            p.data.copy_(full[off:off + n].view_as(p))
            off += n
        return False

    # -- checkpointing: each rank saves only its shard -----------------------------------
    def shard_state_dict(self) -> dict:
        return {"master": self.master.data.clone(), "opt": self.opt.state_dict(),
                "lo": self.lo, "hi": self.hi, "numel": self.numel}

    def load_shard_state_dict(self, sd: dict) -> None:
        assert (sd["lo"], sd["hi"], sd["numel"]) == (self.lo, self.hi, self.numel), \
            "ZeRO shard layout changed; resharding across world sizes is not supported in v0"
        self.master.data.copy_(sd["master"])
        self.opt.load_state_dict(sd["opt"])

    def state_bytes_per_rank(self) -> int:
        """fp32 master shard + Adam m and v for the shard."""
        return 3 * self.shard_size * 4
