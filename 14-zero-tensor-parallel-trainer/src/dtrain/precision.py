"""Mixed precision: autocast contexts and a distributed dynamic loss scaler.

Recipe (Micikevicius et al. 2018): keep fp32 master weights and optimizer state,
run matmuls in a 16-bit type via autocast, compute the loss in fp32.

fp16 has a 5-bit exponent: gradients below ~6e-8 flush to zero. Multiplying the loss
by a large scale S lifts them into range; gradients are divided by S before the
optimizer step. If any gradient overflowed (inf/nan), the step is skipped and S is
halved; after `growth_interval` clean steps S doubles.

bf16 has fp32's 8-bit exponent, so it does not need loss scaling; the scaler is
disabled for bf16 and fp32.

Distributed rule: every rank must make the same skip decision, otherwise replicas
diverge (DP) or pipeline stages disagree. The overflow flag is all_reduced with MAX
over the whole world before deciding.

CPU support (torch 2.14 on Apple M4 Pro): torch.autocast("cpu") accepts both bfloat16
and float16. Neither is faster than fp32 on this CPU for these sizes (see DEVLOG); the
point here is numerical correctness of the logic, which is identical on CUDA.
"""

from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist

from . import comm

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def autocast_ctx(precision: str, device: torch.device):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=DTYPES[precision])


class DynamicLossScaler:
    def __init__(self, enabled: bool, init_scale: float = 2.0 ** 16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 200):
        self.enabled = enabled
        self.scale = init_scale if enabled else 1.0
        self.growth_factor, self.backoff_factor = growth_factor, backoff_factor
        self.growth_interval = growth_interval
        self.good_steps = 0
        self.skipped = 0

    @classmethod
    def for_precision(cls, precision: str, **kw) -> "DynamicLossScaler":
        return cls(enabled=(precision == "fp16"), **kw)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale if self.enabled else loss

    @torch.no_grad()
    def unscale_and_check(self, params, group=None) -> bool:
        """Divide grads by the scale in place and return a globally agreed overflow flag."""
        params = [p for p in params if p.grad is not None]
        flag = torch.zeros(1)
        if params:
            dev = params[0].grad.device
            flag = flag.to(dev)
            for p in params:
                if self.enabled:
                    p.grad.div_(self.scale)
                if not torch.isfinite(p.grad).all():
                    flag.fill_(1.0)
        if dist.is_initialized() and dist.get_world_size(group) > 1:
            comm.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        return bool(flag.item() > 0)

    def update(self, found_inf: bool) -> None:
        if found_inf:
            self.skipped += 1
        if not self.enabled:
            return
        if found_inf:
            self.scale *= self.backoff_factor
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self.good_steps = 0

    def state_dict(self) -> dict:
        return {"scale": self.scale, "good_steps": self.good_steps, "skipped": self.skipped,
                "enabled": self.enabled}

    def load_state_dict(self, sd: dict) -> None:
        self.scale, self.good_steps = sd["scale"], sd["good_steps"]
        self.skipped, self.enabled = sd["skipped"], sd["enabled"]
