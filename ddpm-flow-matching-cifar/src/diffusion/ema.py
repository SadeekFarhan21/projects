"""Exponential moving average of weights with the usual warmup on the decay."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps = 0

    def current_decay(self) -> float:
        # (1 + n) / (10 + n) ramps the decay up so early EMA weights are not stuck at init
        return min(self.decay, (1 + self.steps) / (10 + self.steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        ema_p = list(self.model.parameters())
        cur_p = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ema_p, cur_p, 1.0 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
        self.steps += 1

    def state_dict(self):
        return {"decay": self.decay, "steps": self.steps, "model": self.model.state_dict()}
