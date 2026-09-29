"""numpy reference interpreter, the correctness oracle.

Values are carried in float64. After every node the value is rounded to the
node's storage dtype when `round_every_node` is set (used for constant folding
so the folded constant is exactly what the device would have stored). For the
oracle we keep f32 nodes in float64 (a better estimate of the true result) and
round f16 nodes to half precision, which mimics f16 storage between kernels.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ir import F16, NP_DTYPE, Graph, Node

_UNARY = {
    "neg": np.negative,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "tanh": np.tanh,
    "abs": np.abs,
    "recip": lambda x: 1.0 / x,
    "copy": lambda x: x.copy(),
}
_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "maximum": np.maximum,
    "minimum": np.minimum,
}


def eval_node(n: Node, args: Sequence[np.ndarray]) -> np.ndarray:
    op = n.op
    if op == "const":
        return np.asarray(n.attrs["value"], dtype=np.float64).reshape(n.shape)
    if op in _UNARY:
        return _UNARY[op](args[0])
    if op == "cast":
        return args[0]
    if op in _BINARY:
        return _BINARY[op](args[0], args[1])
    if op == "reduce_sum":
        return args[0].sum(axis=-1, keepdims=True)
    if op == "reduce_max":
        return args[0].max(axis=-1, keepdims=True)
    if op == "matmul":
        return np.matmul(args[0], args[1])
    if op == "reshape":
        return args[0].reshape(n.shape)
    if op == "transpose":
        return np.transpose(args[0], n.attrs["perm"])
    raise NotImplementedError(op)


def _round(v: np.ndarray, dtype: str, round_f32: bool) -> np.ndarray:
    if dtype == F16:
        return v.astype(np.float16).astype(np.float64)
    if round_f32:
        return v.astype(np.float32).astype(np.float64)
    return v


def interpret(
    g: Graph,
    feeds: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    round_f32: bool = False,
) -> list[np.ndarray]:
    """Evaluate the graph; returns outputs cast to their storage dtype."""
    if not isinstance(feeds, Mapping):
        feeds = dict(zip(g.input_names(), feeds))
    env: dict[int, np.ndarray] = {}
    with np.errstate(all="ignore"):
        for n in g:
            if n.op == "input":
                v = np.asarray(feeds[n.attrs["name"]], dtype=np.float64)
                if v.shape != n.shape:
                    raise ValueError(f"input {n.attrs['name']} has shape {v.shape}, expected {n.shape}")
                env[n.id] = _round(v, n.dtype, round_f32)
                continue
            v = eval_node(n, [env[i] for i in n.inputs])
            env[n.id] = _round(np.asarray(v, dtype=np.float64), n.dtype, round_f32)
    return [env[o].astype(NP_DTYPE[g[o].dtype]) for o in g.outputs]
