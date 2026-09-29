"""Tracing frontend: a numpy-like Tensor that records ops into a Graph.

    def f(x, g, b):
        return layernorm(x, g, b)

    graph = trace(f, [((64, 768), "f32"), ((768,), "f32"), ((768,), "f32")])

Composite ops (softmax, layernorm, gelu) are written in terms of primitives
here, so the compiler only ever sees primitives and has to rediscover the
fused kernels itself.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .ir import F32, NP_DTYPE, Graph, broadcast_shapes, normalize_dtype

_ACTIVE: list[Graph] = []


def _graph() -> Graph:
    if not _ACTIVE:
        raise RuntimeError("no active trace; use tcm.trace(...) or tcm.Tracer()")
    return _ACTIVE[-1]


class Tensor:
    __array_priority__ = 1000  # make numpy defer to our operators

    def __init__(self, graph: Graph, nid: int) -> None:
        self.graph = graph
        self.id = nid

    # metadata -------------------------------------------------------------
    @property
    def node(self):
        return self.graph[self.id]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.node.shape

    @property
    def dtype(self) -> str:
        return self.node.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"Tensor(%{self.id}, {self.dtype}{list(self.shape)})"

    # helpers --------------------------------------------------------------
    def _lift(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return constant(other, self.dtype)

    def _unary(self, op: str) -> "Tensor":
        return Tensor(self.graph, self.graph.add(op, [self.id], self.shape, self.dtype))

    def _binary(self, op: str, other, reverse: bool = False) -> "Tensor":
        o = self._lift(other)
        a, b = (o, self) if reverse else (self, o)
        if a.dtype != b.dtype:
            raise TypeError(f"dtype mismatch {a.dtype} vs {b.dtype}; insert an explicit cast")
        shape = broadcast_shapes(a.shape, b.shape)
        return Tensor(self.graph, self.graph.add(op, [a.id, b.id], shape, a.dtype))

    # arithmetic -----------------------------------------------------------
    def __add__(self, o):
        return self._binary("add", o)

    def __radd__(self, o):
        return self._binary("add", o, True)

    def __sub__(self, o):
        return self._binary("sub", o)

    def __rsub__(self, o):
        return self._binary("sub", o, True)

    def __mul__(self, o):
        return self._binary("mul", o)

    def __rmul__(self, o):
        return self._binary("mul", o, True)

    def __truediv__(self, o):
        return self._binary("div", o)

    def __rtruediv__(self, o):
        return self._binary("div", o, True)

    def __neg__(self):
        return self._unary("neg")

    def __pow__(self, p):
        if isinstance(p, int) and 1 <= p <= 4:
            out = self
            for _ in range(p - 1):
                out = out * self
            return out
        if p == 0.5:
            return self.sqrt()
        raise NotImplementedError("only small integer powers and 0.5 are supported")

    def __matmul__(self, o):
        return matmul(self, o)

    def exp(self):
        return self._unary("exp")

    def log(self):
        return self._unary("log")

    def sqrt(self):
        return self._unary("sqrt")

    def rsqrt(self):
        return self._unary("rsqrt")

    def tanh(self):
        return self._unary("tanh")

    def abs(self):
        return self._unary("abs")

    def recip(self):
        return self._unary("recip")

    def maximum(self, o):
        return self._binary("maximum", o)

    def minimum(self, o):
        return self._binary("minimum", o)

    def astype(self, dtype) -> "Tensor":
        dt = normalize_dtype(dtype)
        if dt == self.dtype:
            return self
        return Tensor(self.graph, self.graph.add("cast", [self.id], self.shape, dt))

    # reductions (last axis only, keepdims) --------------------------------
    def _reduce(self, op: str, axis: int, keepdims: bool) -> "Tensor":
        if axis not in (-1, self.ndim - 1):
            raise NotImplementedError("only last-axis reductions are supported")
        out = Tensor(self.graph, self.graph.add(op, [self.id], self.shape[:-1] + (1,), self.dtype))
        if not keepdims:
            out = out.reshape(self.shape[:-1] if self.ndim > 1 else (1,))
        return out

    def sum(self, axis: int = -1, keepdims: bool = True):
        return self._reduce("reduce_sum", axis, keepdims)

    def max(self, axis: int = -1, keepdims: bool = True):
        return self._reduce("reduce_max", axis, keepdims)

    def mean(self, axis: int = -1, keepdims: bool = True):
        return self.sum(axis, keepdims) * (1.0 / self.shape[-1])

    # views ----------------------------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            k = shape.index(-1)
            rest = int(np.prod([s for i, s in enumerate(shape) if i != k]))
            shape[k] = int(np.prod(self.shape)) // rest
        shape = tuple(int(s) for s in shape)
        if int(np.prod(shape)) != int(np.prod(self.shape)):
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        return Tensor(self.graph, self.graph.add("reshape", [self.id], shape, self.dtype))

    def transpose(self, *perm) -> "Tensor":
        if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
            perm = tuple(perm[0])
        if not perm:
            perm = tuple(reversed(range(self.ndim)))
        perm = tuple(p % self.ndim for p in perm)
        if sorted(perm) != list(range(self.ndim)):
            raise ValueError(f"bad permutation {perm}")
        shape = tuple(self.shape[p] for p in perm)
        return Tensor(self.graph, self.graph.add("transpose", [self.id], shape, self.dtype, perm=perm))

    @property
    def T(self) -> "Tensor":
        if self.ndim < 2:
            return self
        perm = list(range(self.ndim))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        return self.transpose(perm)


def constant(value, dtype=F32) -> Tensor:
    g = _graph()
    dt = normalize_dtype(dtype)
    arr = np.asarray(value, dtype=NP_DTYPE[dt])
    shape = arr.shape if arr.ndim else (1,)
    arr = arr.reshape(shape)
    return Tensor(g, g.add("const", [], shape, dt, value=arr))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.dtype != b.dtype:
        raise TypeError("matmul dtype mismatch")
    if a.ndim not in (2, 3) or b.ndim not in (2, 3):
        raise NotImplementedError("matmul supports 2D and batched 3D operands")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"matmul inner dims differ: {a.shape} @ {b.shape}")
    m, n = a.shape[-2], b.shape[-1]
    if a.ndim == 3 and b.ndim == 3:
        if a.shape[0] != b.shape[0]:
            raise ValueError("batched matmul needs equal batch sizes")
        shape = (a.shape[0], m, n)
    elif a.ndim == 3:
        shape = (a.shape[0], m, n)
    elif b.ndim == 3:
        shape = (b.shape[0], m, n)
    else:
        shape = (m, n)
    g = a.graph
    return Tensor(g, g.add("matmul", [a.id, b.id], shape, a.dtype))


# numpy-style free functions ---------------------------------------------
def exp(x):
    return x.exp()


def log(x):
    return x.log()


def sqrt(x):
    return x.sqrt()


def tanh(x):
    return x.tanh()


def maximum(a, b):
    return a.maximum(b) if isinstance(a, Tensor) else b.maximum(a)


# composites (defined only via primitives) --------------------------------
def softmax(x: Tensor) -> Tensor:
    m = x.max(-1)
    e = (x - m).exp()
    return e / e.sum(-1)


def layernorm(x: Tensor, gamma: Tensor | None = None, beta: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    mu = x.mean(-1)
    xc = x - mu
    var = (xc * xc).mean(-1)
    y = xc * (var + eps).rsqrt()
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def gelu(x: Tensor) -> Tensor:
    """tanh approximation of GELU, as used by GPT-2."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + (c * (x + 0.044715 * (x * x * x))).tanh())


def relu(x: Tensor) -> Tensor:
    return x.maximum(0.0)


# tracing entry points ------------------------------------------------------
class Tracer:
    def __init__(self) -> None:
        self.graph = Graph()

    def __enter__(self) -> "Tracer":
        _ACTIVE.append(self.graph)
        return self

    def __exit__(self, *exc) -> None:
        _ACTIVE.pop()

    def input(self, name: str, shape: Sequence[int], dtype=F32) -> Tensor:
        dt = normalize_dtype(dtype)
        return Tensor(self.graph, self.graph.add("input", [], tuple(shape), dt, name=name))

    def output(self, *ts: Tensor) -> None:
        for t in ts:
            self.graph.outputs.append(t.id)


def trace(fn: Callable, specs: Sequence[tuple], names: Sequence[str] | None = None) -> Graph:
    """Trace fn on symbolic inputs described by (shape, dtype) pairs."""
    with Tracer() as t:
        args = []
        for i, spec in enumerate(specs):
            shape, dtype = spec
            name = names[i] if names else f"in{i}"
            args.append(t.input(name, shape, dtype))
        out = fn(*args)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        t.output(*outs)
    t.graph.verify()
    return t.graph
