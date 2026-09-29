"""Hypothesis strategy for random well-conditioned graphs.

Values stay bounded: transcendental ops are wrapped so their inputs stay in a
safe domain (exp(tanh(x)), log(|x| + 1), x / (|y| + 1)), which keeps the
comparison about the compiler and not about overflow semantics.
"""

import numpy as np
from hypothesis import strategies as st

import tcm
from tcm.ir import broadcast_shapes

DIMS = [1, 2, 3, 4, 5, 8, 16, 33]


@st.composite
def random_graph(draw, dtype="f32", max_ops=10):
    rank = draw(st.integers(1, 3))
    base = tuple(draw(st.sampled_from(DIMS)) for _ in range(rank))
    n_in = draw(st.integers(1, 3))
    specs = []
    for _ in range(n_in):
        # an input is the base shape with some dims broadcast (1) or dropped from the front
        drop = draw(st.integers(0, rank - 1))
        shape = tuple(1 if draw(st.booleans()) and draw(st.booleans()) else d for d in base[drop:])
        specs.append((shape, dtype))
    specs[0] = (base, dtype)
    program = draw(st.lists(st.tuples(st.integers(0, 11), st.integers(0, 1000), st.integers(0, 1000)), min_size=1, max_size=max_ops))
    seed = draw(st.integers(0, 2**31 - 1))
    return specs, program, seed


def build(specs, program):
    """Returns fn; trailing arguments beyond specs are matmul weights."""

    def fn(*args):
        pool = list(args[: len(specs)])
        weights = iter(args[len(specs):])
        for code, i, j in program:
            a = pool[i % len(pool)]
            b = pool[j % len(pool)]
            try:
                bs = broadcast_shapes(a.shape, b.shape)
            except ValueError:
                bs = None
            if code == 0 and bs is not None:
                pool.append(a + b)
            elif code == 1 and bs is not None:
                pool.append(a * b)
            elif code == 2 and bs is not None:
                pool.append(a - b)
            elif code == 3 and bs is not None:
                pool.append(a / (b.abs() + 1.0))
            elif code == 4 and bs is not None:
                pool.append(a.maximum(b) if i % 2 else a.minimum(b))
            elif code == 5:
                pool.append(a.tanh().exp())
            elif code == 6:
                pool.append((a.abs() + 1.0).log() if i % 2 else (a.abs() + 0.5).sqrt())
            elif code == 7:
                pool.append(a.sum(-1) if i % 2 else a.max(-1))
            elif code == 8 and a.ndim >= 2:
                perm = list(range(a.ndim))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                pool.append(a.transpose(perm))
            elif code == 9:
                pool.append(a.reshape(-1) if a.ndim > 1 else a.reshape(1, -1))
            elif code == 10 and a.ndim in (2, 3):
                w = next(weights, None)
                if w is not None and w.shape[0] == a.shape[-1]:
                    pool.append((a @ w) * (1.0 / a.shape[-1]))
            elif code == 11:
                pool.append(tcm.softmax(a) if i % 2 else -a)
        outs = [pool[-1]]
        if len(pool) > len(args) + 1 and program[0][1] % 3 == 0:
            outs.append(pool[len(args)])
        return tuple(outs)

    return fn


def trace_random(specs, program):
    """Trace once to learn K at each matmul site, then retrace with (K, 5) weights as inputs."""
    fn = build(specs, program)
    # figure out weight shapes by tracing without weights, recording K at each matmul site
    ks = []

    def probe(*args):
        pool = list(args)
        for code, i, j in program:
            a = pool[i % len(pool)]
            b = pool[j % len(pool)]
            try:
                bs = broadcast_shapes(a.shape, b.shape)
            except ValueError:
                bs = None
            if code in (0, 1, 2, 3, 4) and bs is not None:
                pool.append(a + b)
            elif code in (5, 6, 11):
                pool.append(a.tanh())
            elif code == 7:
                pool.append(a.sum(-1))
            elif code == 8 and a.ndim >= 2:
                perm = list(range(a.ndim))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                pool.append(a.transpose(perm))
            elif code == 9:
                pool.append(a.reshape(-1) if a.ndim > 1 else a.reshape(1, -1))
            elif code == 10 and a.ndim in (2, 3):
                ks.append(a.shape[-1])
                pool.append(a @ tcm.constant(np.zeros((a.shape[-1], 5), np.float32), a.dtype))
        return pool[-1]

    tcm.trace(probe, specs)
    dtype = specs[0][1]
    weight_specs = [((k, 5), dtype) for k in ks]
    return tcm.trace(fn, list(specs) + weight_specs)
