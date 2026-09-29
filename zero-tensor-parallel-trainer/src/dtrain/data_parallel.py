"""Data parallelism with gradient bucketing and compute/communication overlap.

The model is replicated; each rank runs forward/backward on its slice of the global
batch; gradients are averaged with all_reduce before the optimizer step.

Instead of one all_reduce per parameter (latency bound: ~100 small messages) or one
giant all_reduce after backward (no overlap), parameters are grouped into buckets of
roughly `bucket_cap_mb`. Buckets are filled in reverse parameter order, which is
roughly the order autograd produces gradients. When the last gradient of a bucket
arrives (post-accumulate-grad hook), the bucket is flattened and an async all_reduce
is launched while autograd keeps computing earlier layers' gradients.

Invariant: every rank launches buckets in the same order (0, 1, 2, ...). Collectives
are matched by call order, so a bucket that becomes ready early waits until all
lower-numbered buckets have launched. Without this, ranks whose autograd scheduled
gradients differently would pair up the wrong tensors.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn

from . import comm


class _Bucket:
    def __init__(self, params: list[nn.Parameter]):
        self.params = params
        self.numel = sum(p.numel() for p in params)
        self.pending = len(params)
        self.handle = None
        self.flat: torch.Tensor | None = None

    def reset(self) -> None:
        self.pending = len(self.params)
        self.handle = None
        self.flat = None


class BucketedDDP:
    """Wraps a module's gradient synchronisation. Usage per step:

        loss.backward()
        ddp.finish_grad_sync()   # waits for in-flight buckets, writes averaged grads
        opt.step()

    bucket_cap_mb=0 gives one bucket per parameter (the naive baseline).
    overlap=False launches all buckets after backward (bucketing without overlap).
    comm_dtype=torch.bfloat16 compresses gradients on the wire (halves bytes).
    """

    def __init__(self, model: nn.Module, bucket_cap_mb: float = 1.0, overlap: bool = True,
                 group=None, comm_dtype: torch.dtype | None = None,
                 broadcast_init: bool = True):
        self.model = model
        self.group = group
        self.world = dist.get_world_size(group)
        self.overlap = overlap
        self.comm_dtype = comm_dtype
        self.sync_enabled = True
        self.params = [p for p in model.parameters() if p.requires_grad]

        if broadcast_init:
            # Guarantee identical replicas even if seeds diverged.
            with torch.no_grad():
                for p in model.state_dict().values():
                    comm.broadcast(p, src=0, group=group)

        cap = bucket_cap_mb * 1024 * 1024
        self.buckets: list[_Bucket] = []
        cur, cur_bytes = [], 0
        for p in reversed(self.params):
            cur.append(p)
            cur_bytes += p.numel() * p.element_size()
            if cur_bytes >= cap:
                self.buckets.append(_Bucket(cur))
                cur, cur_bytes = [], 0
        if cur:
            self.buckets.append(_Bucket(cur))
        self.bucket_of = {id(p): i for i, b in enumerate(self.buckets) for p in b.params}
        self.next_launch = 0
        self.exposed_comm_s = 0.0  # time finish_grad_sync spent blocked on the network

        for p in self.params:
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    # -- hooks --------------------------------------------------------------------------
    def _on_grad_ready(self, p: nn.Parameter) -> None:
        if not self.sync_enabled:
            return
        b = self.buckets[self.bucket_of[id(p)]]
        b.pending -= 1
        if self.overlap:
            self._launch_ready()

    def _launch_ready(self) -> None:
        while self.next_launch < len(self.buckets) and self.buckets[self.next_launch].pending == 0:
            self._launch(self.buckets[self.next_launch])
            self.next_launch += 1

    def _launch(self, b: _Bucket) -> None:
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in b.params]
        flat = torch.cat([g.reshape(-1) for g in grads])
        if self.comm_dtype is not None:
            flat = flat.to(self.comm_dtype)
        b.flat = flat
        b.handle = comm.all_reduce(flat, group=self.group, async_op=True)

    # -- public -------------------------------------------------------------------------
    @contextmanager
    def no_sync(self):
        """Gradient accumulation: skip communication for the microbatches inside."""
        self.sync_enabled = False
        try:
            yield
        finally:
            self.sync_enabled = True

    def finish_grad_sync(self) -> None:
        # Buckets whose params got no gradient this step (unused params) are launched
        # here with zeros so every rank still issues the same sequence of collectives.
        for b in self.buckets[self.next_launch:]:
            b.pending = 0
        self._launch_ready()
        t0 = time.perf_counter()
        for b in self.buckets:
            b.handle.wait()
        self.exposed_comm_s += time.perf_counter() - t0
        for b in self.buckets:
            flat = b.flat.to(b.params[0].dtype) if self.comm_dtype is not None else b.flat
            flat.div_(self.world)
            off = 0
            for p in b.params:
                n = p.numel()
                g = flat[off:off + n].view_as(p)
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)
                off += n
            b.reset()
        self.next_launch = 0


def allreduce_grads_naive(model: nn.Module, group=None) -> None:
    """Reference implementation: one blocking all_reduce per parameter after backward."""
    world = dist.get_world_size(group)
    for p in model.parameters():
        if p.grad is not None:
            comm.all_reduce(p.grad, group=group)
            p.grad.div_(world)
