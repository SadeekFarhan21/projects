"""Collective wrappers that count communication volume, plus ring collectives built
from point-to-point send/recv.

Every parallel strategy in this package calls these wrappers instead of torch.distributed
directly. That gives one place to measure bytes moved per step.

Two numbers are recorded per call:
  payload_bytes : size of the tensor handed to the collective (what the caller "sees").
  wire_bytes    : bytes this rank *sends* under a bandwidth-optimal ring algorithm.
                  all_reduce     2 (N-1)/N * payload
                  reduce_scatter   (N-1)/N * payload   (payload = full input)
                  all_gather       (N-1)/N * payload   (payload = full output)
                  send           payload
The wire estimate is what NCCL's ring achieves and what gloo's ring allreduce does on
large messages, so it is the right number for comparing strategies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class CommStats:
    calls: dict = field(default_factory=lambda: defaultdict(int))
    payload_bytes: dict = field(default_factory=lambda: defaultdict(int))
    wire_bytes: dict = field(default_factory=lambda: defaultdict(int))

    def record(self, op: str, payload: int, wire: float) -> None:
        self.calls[op] += 1
        self.payload_bytes[op] += int(payload)
        self.wire_bytes[op] += int(wire)

    def reset(self) -> None:
        self.calls.clear()
        self.payload_bytes.clear()
        self.wire_bytes.clear()

    def total_wire(self) -> int:
        return sum(self.wire_bytes.values())

    def total_payload(self) -> int:
        return sum(self.payload_bytes.values())

    def as_dict(self) -> dict:
        return {"calls": dict(self.calls), "payload_bytes": dict(self.payload_bytes),
                "wire_bytes": dict(self.wire_bytes)}


STATS = CommStats()


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _size(group) -> int:
    return dist.get_world_size(group)


def all_reduce(t: torch.Tensor, op=dist.ReduceOp.SUM, group=None, async_op: bool = False):
    n = _size(group)
    STATS.record("all_reduce", _nbytes(t), 2 * (n - 1) / n * _nbytes(t))
    return dist.all_reduce(t, op=op, group=group, async_op=async_op)


def reduce_scatter(out: torch.Tensor, inp: torch.Tensor, op=dist.ReduceOp.SUM, group=None,
                   async_op: bool = False):
    """out has numel = inp.numel() / N; rank r receives the reduced r-th chunk."""
    n = _size(group)
    STATS.record("reduce_scatter", _nbytes(inp), (n - 1) / n * _nbytes(inp))
    return dist.reduce_scatter_tensor(out, inp, op=op, group=group, async_op=async_op)


def all_gather(out: torch.Tensor, inp: torch.Tensor, group=None, async_op: bool = False):
    """out has numel = N * inp.numel(); chunks are laid out in rank order."""
    n = _size(group)
    STATS.record("all_gather", _nbytes(out), (n - 1) / n * _nbytes(out))
    return dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)


def send(t: torch.Tensor, dst: int, group=None):
    STATS.record("send", _nbytes(t), _nbytes(t))
    return dist.send(t, dst=dst, group=group)


def recv(t: torch.Tensor, src: int, group=None):
    STATS.record("recv", _nbytes(t), 0)
    return dist.recv(t, src=src, group=group)


def isend(t: torch.Tensor, dst: int):
    STATS.record("send", _nbytes(t), _nbytes(t))
    return dist.isend(t, dst=dst)


def irecv(t: torch.Tensor, src: int):
    STATS.record("recv", _nbytes(t), 0)
    return dist.irecv(t, src=src)


def broadcast(t: torch.Tensor, src: int, group=None):
    n = _size(group)
    STATS.record("broadcast", _nbytes(t), _nbytes(t) if dist.get_rank() == src else 0)
    return dist.broadcast(t, src=src, group=group)


# ---------------------------------------------------------------------------------------
# Ring collectives from send/recv. These exist to show the algorithm the vendor
# collectives implement; tests check them against dist.all_reduce. They operate on the
# default (world) group.
# ---------------------------------------------------------------------------------------

def _exchange(send_buf: torch.Tensor, recv_buf: torch.Tensor, right: int, left: int) -> None:
    # Post both sides before waiting so a ring of blocking sends cannot deadlock.
    reqs = [isend(send_buf.contiguous(), right), irecv(recv_buf, left)]
    for r in reqs:
        r.wait()


def ring_reduce_scatter_(flat: torch.Tensor) -> torch.Tensor:
    """In-place ring reduce-scatter over the world. flat.numel() must divide by N.

    After N-1 steps rank r holds the full sum of chunk (r + 1) % N. Returns that chunk.
    """
    n, r = dist.get_world_size(), dist.get_rank()
    chunks = flat.view(n, -1)
    right, left = (r + 1) % n, (r - 1) % n
    tmp = torch.empty_like(chunks[0])
    for step in range(n - 1):
        send_idx = (r - step) % n
        recv_idx = (r - step - 1) % n
        _exchange(chunks[send_idx], tmp, right, left)
        chunks[recv_idx] += tmp
    return chunks[(r + 1) % n]


def ring_all_gather_(flat: torch.Tensor) -> None:
    """In-place ring all-gather, assuming rank r owns chunk (r + 1) % N (the layout
    ring_reduce_scatter_ leaves behind)."""
    n, r = dist.get_world_size(), dist.get_rank()
    chunks = flat.view(n, -1)
    right, left = (r + 1) % n, (r - 1) % n
    tmp = torch.empty_like(chunks[0])
    for step in range(n - 1):
        send_idx = (r + 1 - step) % n
        recv_idx = (r - step) % n
        _exchange(chunks[send_idx], tmp, right, left)
        chunks[recv_idx].copy_(tmp)


def ring_all_reduce_(t: torch.Tensor) -> torch.Tensor:
    """Bandwidth-optimal ring all-reduce (sum) built only from isend/irecv."""
    n = dist.get_world_size()
    if n == 1:
        return t
    flat = t.reshape(-1)
    pad = (-flat.numel()) % n
    buf = torch.cat([flat, flat.new_zeros(pad)]) if pad else flat.clone()
    ring_reduce_scatter_(buf)
    ring_all_gather_(buf)
    t.copy_(buf[: flat.numel()].view_as(t))
    return t
