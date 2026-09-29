"""Tiling schedule shared by the testbench driver and the cycle model.

A GEMM C[M, Nout] = A[M, K] @ B[K, Nout] is cut into commands for an n x n
array whose activation and output buffers hold `buf_rows` rows each.

Loop order (outer to inner): output column tile, M chunk, K tile.
The output buffer accumulates over the K tiles of one (M chunk, column tile)
pair and is drained after the last K tile. Weights are reloaded unless the
previous command used the same (K tile, column tile), which happens only when
K fits in a single tile.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Cmd:
    m: int          # rows in this command
    load_w: bool
    acc: bool
    drain: bool
    m0: int         # first A row
    k0: int         # first K index (weight tile row)
    n0: int         # first output column


def schedule(M: int, K: int, Nout: int, n: int, buf_rows: int) -> list[Cmd]:
    if min(M, K, Nout, n, buf_rows) < 1:
        raise ValueError("all dimensions must be positive")
    kt = ceil(K / n)
    nt = ceil(Nout / n)
    cmds: list[Cmd] = []
    loaded: tuple[int, int] | None = None
    for ni in range(nt):
        for m0 in range(0, M, buf_rows):
            mc = min(buf_rows, M - m0)
            for ki in range(kt):
                load = loaded != (ki, ni)
                loaded = (ki, ni)
                cmds.append(Cmd(mc, load, ki > 0, ki == kt - 1, m0, ki * n, ni * n))
    return cmds


def pad_to(x, rows: int, cols: int):
    import numpy as np
    out = np.zeros((rows, cols), dtype=x.dtype)
    out[: x.shape[0], : x.shape[1]] = x
    return out


def golden(A, B):
    """int8 x int8 GEMM with int32 wraparound, the RTL's arithmetic."""
    import numpy as np
    return (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)
