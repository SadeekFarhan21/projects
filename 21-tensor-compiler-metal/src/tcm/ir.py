"""SSA graph IR.

A Graph is a list of Nodes in topological order. Every Node defines exactly one
value (SSA): it has an integer id, an op name, the ids of its inputs, a static
shape, a dtype and a small dict of attributes. Nodes are immutable once built;
passes produce new graphs through a GraphBuilder instead of mutating in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

F32 = "f32"
F16 = "f16"
DTYPES = (F32, F16)
NP_DTYPE = {F32: np.float32, F16: np.float16}
ITEMSIZE = {F32: 4, F16: 2}

UNARY_OPS = ("neg", "exp", "log", "sqrt", "rsqrt", "tanh", "abs", "recip", "copy")
BINARY_OPS = ("add", "sub", "mul", "div", "maximum", "minimum")
REDUCE_OPS = ("reduce_sum", "reduce_max")
VIEW_OPS = ("reshape", "transpose")
ELEMENTWISE_OPS = UNARY_OPS + BINARY_OPS + ("cast",)
ALL_OPS = ("input", "const") + ELEMENTWISE_OPS + REDUCE_OPS + ("matmul",) + VIEW_OPS


def normalize_dtype(dt: Any) -> str:
    if dt in DTYPES:
        return dt
    d = np.dtype(dt)
    if d == np.float32:
        return F32
    if d == np.float16:
        return F16
    raise TypeError(f"unsupported dtype {dt!r}")


def broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    n = max(len(a), len(b))
    pa = (1,) * (n - len(a)) + tuple(a)
    pb = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(pa, pb):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"shapes {a} and {b} do not broadcast")
    return tuple(out)


def numel(shape: Iterable[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


@dataclass(frozen=True)
class Node:
    id: int
    op: str
    inputs: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def nbytes(self) -> int:
        return numel(self.shape) * ITEMSIZE[self.dtype]

    def is_elementwise(self) -> bool:
        return self.op in ELEMENTWISE_OPS

    def is_reduce(self) -> bool:
        return self.op in REDUCE_OPS

    def is_view(self) -> bool:
        return self.op in VIEW_OPS

    def attr_key(self) -> tuple:
        """Hashable form of attrs, used by CSE. Constants hash by content."""
        items = []
        for k in sorted(self.attrs):
            v = self.attrs[k]
            if isinstance(v, np.ndarray):
                v = ("nd", v.dtype.str, v.shape, v.tobytes())
            elif isinstance(v, list):
                v = tuple(v)
            items.append((k, v))
        return tuple(items)


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.order: list[int] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self._next = 0

    # construction -------------------------------------------------------
    def add(self, op: str, inputs: Iterable[int], shape, dtype: str, **attrs) -> int:
        if op not in ALL_OPS:
            raise ValueError(f"unknown op {op}")
        nid = self._next
        self._next += 1
        for i in inputs:
            if i not in self.nodes:
                raise ValueError(f"input {i} of new node is not defined (SSA violation)")
        node = Node(nid, op, tuple(inputs), tuple(int(s) for s in shape), dtype, dict(attrs))
        self.nodes[nid] = node
        self.order.append(nid)
        if op == "input":
            self.inputs.append(nid)
        return nid

    def __getitem__(self, nid: int) -> Node:
        return self.nodes[nid]

    def __iter__(self):
        for nid in self.order:
            yield self.nodes[nid]

    def __len__(self) -> int:
        return len(self.order)

    # analysis -----------------------------------------------------------
    def users(self) -> dict[int, list[int]]:
        u: dict[int, list[int]] = {nid: [] for nid in self.order}
        for n in self:
            for i in n.inputs:
                u[i].append(n.id)
        return u

    def input_names(self) -> list[str]:
        return [self.nodes[i].attrs["name"] for i in self.inputs]

    def verify(self) -> None:
        """Check SSA, topological order and that outputs exist."""
        seen: set[int] = set()
        for nid in self.order:
            n = self.nodes[nid]
            for i in n.inputs:
                if i not in seen:
                    raise AssertionError(f"node {nid} ({n.op}) uses {i} before definition")
            if n.dtype not in DTYPES:
                raise AssertionError(f"node {nid} has bad dtype {n.dtype}")
            seen.add(nid)
        for o in self.outputs:
            if o not in seen:
                raise AssertionError(f"output {o} not defined")

    def count_ops(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for n in self:
            c[n.op] = c.get(n.op, 0) + 1
        return c

    def __str__(self) -> str:
        lines = []
        for n in self:
            args = ", ".join(f"%{i}" for i in n.inputs)
            attrs = {k: v for k, v in n.attrs.items() if k != "value"}
            if n.op == "const":
                v = n.attrs["value"]
                attrs["value"] = float(v) if v.size == 1 else f"<{v.shape}>"
            a = f" {attrs}" if attrs else ""
            lines.append(f"%{n.id}: {n.dtype}{list(n.shape)} = {n.op}({args}){a}")
        lines.append("return " + ", ".join(f"%{o}" for o in self.outputs))
        return "\n".join(lines)


class GraphBuilder:
    """Rebuilds a graph node by node while remapping ids. Used by passes."""

    def __init__(self, src: Graph) -> None:
        self.src = src
        self.g = Graph()
        self.map: dict[int, int] = {}

    def copy_node(self, n: Node, inputs: tuple[int, ...] | None = None) -> int:
        ins = tuple(self.map[i] for i in n.inputs) if inputs is None else inputs
        nid = self.g.add(n.op, ins, n.shape, n.dtype, **n.attrs)
        return nid

    def finish(self) -> Graph:
        self.g.outputs = [self.map[o] for o in self.src.outputs]
        # keep graph inputs in their original order even if reordered
        order = [self.map[i] for i in self.src.inputs if i in self.map]
        self.g.inputs = order
        self.g.verify()
        return self.g
