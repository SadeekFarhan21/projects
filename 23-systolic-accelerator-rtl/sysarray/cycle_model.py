"""Analytical cycle model of sa_top.

Derived from the RTL phase structure, not fitted to measurements:

  per command
    IDLE       1 cycle (command handshake)
    LOAD       k_in = n*load_w + m input beats of n bytes each
    PRELOAD    n + 1 cycles if load_w
    COMPUTE    m + 2n cycles (m issue slots plus skew, array and deskew latency)
    DRAIN      m output beats of 4n bytes, plus 2 cycles of SRAM and FIFO latency

A stream phase of k beats of S bytes over a channel of `bw` bytes per cycle
(one channel per direction, token bucket with one beat of burst) finishes its
last beat at cycle max(k - 1, ceil((k - 1) * S / bw)) after the first.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .schedule import schedule

INF_BW = float("inf")


def stream_cycles(beats: int, beat_bytes: int, bw: float) -> int:
    if beats == 0:
        return 0
    if bw == INF_BW or bw >= beat_bytes:
        return beats
    return ceil((beats - 1) * beat_bytes / bw - 1e-9) + 1


@dataclass
class Prediction:
    cycles: int
    macs: int                # useful MACs M*K*Nout
    bytes_in: int            # activation + weight bytes over the input channel
    bytes_out: int           # output bytes over the output channel
    commands: int
    phase: dict              # cycles per phase

    def utilization(self, n: int) -> float:
        return self.macs / (n * n * self.cycles)


def predict(M: int, K: int, Nout: int, n: int = 8, buf_rows: int = 64,
            bw: float = INF_BW) -> Prediction:
    ph = {"idle": 0, "load": 0, "preload": 0, "compute": 0, "drain": 0}
    bytes_in = bytes_out = 0
    cmds = schedule(M, K, Nout, n, buf_rows)
    for c in cmds:
        ph["idle"] += 1
        k_in = (n if c.load_w else 0) + c.m
        ph["load"] += stream_cycles(k_in, n, bw)
        bytes_in += k_in * n
        if c.load_w:
            ph["preload"] += n + 1
        ph["compute"] += c.m + 2 * n
        if c.drain:
            ph["drain"] += 2 + stream_cycles(c.m, 4 * n, bw)
            bytes_out += c.m * 4 * n
    return Prediction(sum(ph.values()), M * K * Nout, bytes_in, bytes_out, len(cmds), ph)
