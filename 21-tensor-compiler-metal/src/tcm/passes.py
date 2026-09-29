"""Graph-level passes: shape/dtype inference, constant folding, CSE, algebraic
simplification and dead code elimination. Every pass returns a new Graph.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .interp import eval_node
from .ir import (
    BINARY_OPS,
    NP_DTYPE,
    UNARY_OPS,
    Graph,
    GraphBuilder,
    Node,
    broadcast_shapes,
    numel,
)

FOLD_LIMIT = 1 << 20  # do not fold constants larger than this many elements


# ---------------------------------------------------------------- inference
def infer(op: str, ins: list[Node], attrs: dict) -> tuple[tuple[int, ...], str]:
    """Compute (shape, dtype) of an op from its inputs. Raises on type errors."""
    if op in UNARY_OPS:
        return ins[0].shape, ins[0].dtype
    if op == "cast":
        return ins[0].shape, attrs["_dtype"]
    if op in BINARY_OPS:
        a, b = ins
        if a.dtype != b.dtype:
            raise TypeError(f"{op}: dtype mismatch {a.dtype} {b.dtype}")
        return broadcast_shapes(a.shape, b.shape), a.dtype
    if op in ("reduce_sum", "reduce_max"):
        return ins[0].shape[:-1] + (1,), ins[0].dtype
    if op == "matmul":
        a, b = ins
        if a.dtype != b.dtype:
            raise TypeError("matmul dtype mismatch")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError("matmul inner dimension mismatch")
        batch = ()
        if len(a.shape) == 3:
            batch = (a.shape[0],)
        if len(b.shape) == 3:
            if batch and batch[0] != b.shape[0]:
                raise ValueError("matmul batch mismatch")
            batch = (b.shape[0],)
        return batch + (a.shape[-2], b.shape[-1]), a.dtype
    if op == "reshape":
        shape = tuple(attrs["_shape"])
        if numel(shape) != numel(ins[0].shape):
            raise ValueError("reshape changes element count")
        return shape, ins[0].dtype
    if op == "transpose":
        perm = attrs["perm"]
        return tuple(ins[0].shape[p] for p in perm), ins[0].dtype
    raise NotImplementedError(op)


def infer_shapes(g: Graph) -> Graph:
    """Re-derive every node's shape and dtype and check it against the stored one."""
    for n in g:
        if n.op in ("input", "const"):
            if n.op == "const" and tuple(n.attrs["value"].shape) != n.shape:
                raise AssertionError(f"const %{n.id} shape mismatch")
            continue
        attrs = dict(n.attrs)
        attrs["_dtype"] = n.dtype
        attrs["_shape"] = n.shape
        shape, dtype = infer(n.op, [g[i] for i in n.inputs], attrs)
        if n.op == "cast":
            dtype = n.dtype
        if tuple(shape) != n.shape or dtype != n.dtype:
            raise AssertionError(f"%{n.id} {n.op}: stored {n.dtype}{n.shape} inferred {dtype}{shape}")
    return g


# ---------------------------------------------------------------- helpers
def _const_value(n: Node) -> np.ndarray | None:
    return n.attrs["value"] if n.op == "const" else None


def _is_scalar_const(n: Node, value: float) -> bool:
    v = _const_value(n)
    return v is not None and v.size >= 1 and bool(np.all(v == value))


def _rewrite(g: Graph, rule: Callable[[GraphBuilder, Node], int | None]) -> Graph:
    """Generic rewriter: for each node, rule may return a replacement id in the
    new graph; otherwise the node is copied."""
    b = GraphBuilder(g)
    for n in g:
        new = rule(b, n)
        if new is None:
            new = b.copy_node(n)
        b.map[n.id] = new
    return b.finish()


# ---------------------------------------------------------------- DCE
def dce(g: Graph) -> Graph:
    live: set[int] = set(g.outputs)
    for nid in reversed(g.order):
        if nid in live:
            live.update(g[nid].inputs)
    live.update(g.inputs)  # inputs are part of the signature
    b = GraphBuilder(g)
    for n in g:
        if n.id in live:
            b.map[n.id] = b.copy_node(n)
    return b.finish()


# ---------------------------------------------------------------- CSE
def cse(g: Graph) -> Graph:
    seen: dict[tuple, int] = {}

    def rule(b: GraphBuilder, n: Node):
        if n.op == "input":
            return None
        key = (n.op, tuple(b.map[i] for i in n.inputs), n.shape, n.dtype, n.attr_key())
        if key in seen:
            return seen[key]
        nid = b.copy_node(n)
        seen[key] = nid
        return nid

    return _rewrite(g, rule)


# ---------------------------------------------------------------- constant folding
def constant_fold(g: Graph) -> Graph:
    def rule(b: GraphBuilder, n: Node):
        if n.op in ("input", "const") or not n.inputs:
            return None
        srcs = [b.g[b.map[i]] for i in n.inputs]
        if not all(s.op == "const" for s in srcs) or numel(n.shape) > FOLD_LIMIT:
            return None
        with np.errstate(all="ignore"):
            args = [np.asarray(s.attrs["value"], dtype=np.float64) for s in srcs]
            v = np.asarray(eval_node(n, args), dtype=np.float64)
        v = v.astype(NP_DTYPE[n.dtype]).reshape(n.shape)
        return b.g.add("const", [], n.shape, n.dtype, value=v)

    return _rewrite(g, rule)


# ---------------------------------------------------------------- algebraic simplification
def _compose_perm(p_inner, p_outer):
    # y = transpose(transpose(x, p_inner), p_outer) == transpose(x, p_inner[p_outer])
    return tuple(p_inner[i] for i in p_outer)


def simplify(g: Graph) -> Graph:
    def rule(b: GraphBuilder, n: Node):
        ng = b.g
        ins = [ng[b.map[i]] for i in n.inputs]
        op = n.op
        # identities that keep the output shape equal to x's shape
        if op in ("add", "sub", "mul", "div"):
            x, y = ins
            if op in ("add", "sub") and _is_scalar_const(y, 0.0) and x.shape == n.shape:
                return x.id
            if op == "add" and _is_scalar_const(x, 0.0) and y.shape == n.shape:
                return y.id
            if op in ("mul", "div") and _is_scalar_const(y, 1.0) and x.shape == n.shape:
                return x.id
            if op == "mul" and _is_scalar_const(x, 1.0) and y.shape == n.shape:
                return y.id
            # x / c  ->  x * (1/c)   (strength reduction, one extra rounding)
            if op == "div" and y.op == "const":
                inv = (1.0 / y.attrs["value"].astype(np.float64)).astype(NP_DTYPE[y.dtype])
                c = ng.add("const", [], y.shape, y.dtype, value=inv)
                return ng.add("mul", [x.id, c], n.shape, n.dtype)
            # 1 / y -> recip(y);  1 / sqrt(z) -> rsqrt(z)
            if op == "div" and _is_scalar_const(x, 1.0) and y.shape == n.shape:
                if y.op == "sqrt":
                    return ng.add("rsqrt", [y.inputs[0]], n.shape, n.dtype)
                return ng.add("recip", [y.id], n.shape, n.dtype)
            # x * -1 -> neg x
            if op == "mul" and _is_scalar_const(y, -1.0) and x.shape == n.shape:
                return ng.add("neg", [x.id], n.shape, n.dtype)
        if op == "neg" and ins[0].op == "neg":
            return ins[0].inputs[0]
        if op == "recip" and ins[0].op == "sqrt":
            return ng.add("rsqrt", [ins[0].inputs[0]], n.shape, n.dtype)
        if op == "cast" and ins[0].dtype == n.dtype:
            return ins[0].id
        if op in ("reduce_sum", "reduce_max") and ins[0].shape[-1] == 1:
            return ins[0].id
        if op == "reshape":
            x = ins[0]
            if x.shape == n.shape:
                return x.id
            if x.op == "reshape":
                src = ng[x.inputs[0]]
                if src.shape == n.shape:
                    return src.id
                return ng.add("reshape", [src.id], n.shape, n.dtype)
        if op == "transpose":
            x = ins[0]
            perm = tuple(n.attrs["perm"])
            if perm == tuple(range(len(perm))):
                return x.id
            if x.op == "transpose":
                p = _compose_perm(x.attrs["perm"], perm)
                if p == tuple(range(len(p))):
                    return x.inputs[0]
                return ng.add("transpose", [x.inputs[0]], n.shape, n.dtype, perm=p)
        return None

    return _rewrite(g, rule)


# ---------------------------------------------------------------- pipeline
def optimize(g: Graph, max_iters: int = 8, log: list | None = None) -> Graph:
    """Run fold, simplify, CSE and DCE to a fixed point."""
    infer_shapes(g)
    for it in range(max_iters):
        before = (len(g), tuple(sorted(g.count_ops().items())))
        g = constant_fold(g)
        g = simplify(g)
        g = cse(g)
        g = dce(g)
        infer_shapes(g)
        after = (len(g), tuple(sorted(g.count_ops().items())))
        if log is not None:
            log.append({"iter": it, "nodes": len(g)})
        if after == before:
            break
    return g
