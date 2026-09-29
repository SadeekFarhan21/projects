"""Lowering from an optimized Graph to a kernel schedule.

Steps
  1. legalize: insert `copy` nodes where a reshape cannot be expressed as a
     strided view of its source, and where a graph output is a view, input or
     constant (outputs must be fresh contiguous buffers).
  2. group: fusion. Walk nodes in reverse topological order and put a node in
     its consumers' group when every consumer is in the same group and the
     group's iteration domain is compatible. Reductions turn an elementwise
     group into a row group, which is how softmax and layernorm become one
     kernel each.
  3. plan: order kernels, compute liveness of every materialized value and
     assign intermediate values to reusable buffers (best-fit free list).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Graph, GraphBuilder, Node, broadcast_shapes, numel


# ------------------------------------------------------------------ views
def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    st = []
    acc = 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return tuple(reversed(st))


def restride(oshape, ostrides, nshape) -> tuple[int, ...] | None:
    """Strides for viewing (oshape, ostrides) as nshape without a copy, or None."""
    old = [(d, s) for d, s in zip(oshape, ostrides) if d != 1]
    nidx = [i for i, d in enumerate(nshape) if d != 1]
    nstrides = [0] * len(nshape)
    oi = nj = 0
    while oi < len(old) and nj < len(nidx):
        op, npd = old[oi][0], nshape[nidx[nj]]
        oi2, nj2 = oi + 1, nj + 1
        while op != npd:
            if op < npd:
                op *= old[oi2][0]
                oi2 += 1
            else:
                npd *= nshape[nidx[nj2]]
                nj2 += 1
        for k in range(oi, oi2 - 1):
            if old[k][1] != old[k + 1][1] * old[k + 1][0]:
                return None
        st = old[oi2 - 1][1]
        for k in range(nj2 - 1, nj - 1, -1):
            nstrides[nidx[k]] = st
            st *= nshape[nidx[k]]
        oi, nj = oi2, nj2
    return tuple(nstrides)


def resolve_view(g: Graph, nid: int) -> tuple[int, tuple[int, ...]]:
    """Follow reshape/transpose chains to (materialized base id, strides)."""
    n = g[nid]
    if n.op == "transpose":
        b, s = resolve_view(g, n.inputs[0])
        return b, tuple(s[p] for p in n.attrs["perm"])
    if n.op == "reshape":
        b, s = resolve_view(g, n.inputs[0])
        ns = restride(g[n.inputs[0]].shape, s, n.shape)
        if ns is None:
            raise ValueError(f"reshape %{nid} is not a view of its source")
        return b, ns
    return nid, contiguous_strides(n.shape)


def legalize(g: Graph) -> Graph:
    b = GraphBuilder(g)
    for n in g:
        if n.op == "reshape":
            src = b.map[n.inputs[0]]
            try:
                _, s = resolve_view(b.g, src)
                ok = restride(b.g[src].shape, s, n.shape) is not None
            except ValueError:
                ok = False
            if not ok:
                sn = b.g[src]
                src = b.g.add("copy", [src], sn.shape, sn.dtype)
            b.map[n.id] = b.g.add("reshape", [src], n.shape, n.dtype)
            continue
        b.map[n.id] = b.copy_node(n)
    out = b.finish()
    # outputs must be fresh buffers produced by a kernel
    new_outputs = []
    for o in out.outputs:
        n = out[o]
        if n.op in ("input", "const", "reshape", "transpose") or o in new_outputs:
            o = out.add("copy", [o], n.shape, n.dtype)
        new_outputs.append(o)
    out.outputs = new_outputs
    out.verify()
    return out


# ------------------------------------------------------------------ fusion
def is_inline_const(n: Node) -> bool:
    return n.op == "const" and numel(n.shape) == 1


@dataclass
class Group:
    kind: str  # "ew" | "row" | "matmul"
    root: int
    nodes: set[int] = field(default_factory=set)
    domain: tuple[int, ...] = ()  # ew: output shape; row: lead + (N,)

    def ordered(self) -> list[int]:
        return sorted(self.nodes)


def _pad(shape, rank):
    return (1,) * (rank - len(shape)) + tuple(shape)


def _try_join(grp: Group, n: Node, g: Graph) -> bool:
    D = grp.domain
    if n.is_elementwise():
        if len(n.shape) > len(D):
            return False
        try:
            return broadcast_shapes(n.shape, D) == D
        except ValueError:
            return False
    if n.is_reduce():
        ishape = g[n.inputs[0]].shape
        if len(ishape) != len(D) or ishape[:-1] != D[:-1]:
            return False
        if grp.kind == "ew" and D[-1] in (1, ishape[-1]):
            grp.kind = "row"
            grp.domain = ishape
            return True
        if grp.kind == "row" and D == ishape:
            return True
    return False


def fuse_groups(g: Graph, fuse: bool = True) -> list[Group]:
    users = g.users()
    outputs = set(g.outputs)
    group_of: dict[int, Group] = {}
    groups: list[Group] = []
    for nid in reversed(g.order):
        n = g[nid]
        if n.op in ("input", "const") or n.is_view():
            continue
        if n.op == "matmul":
            grp = Group("matmul", nid, {nid}, n.shape)
            groups.append(grp)
            group_of[nid] = grp
            continue
        joined = False
        us = users[nid]
        if fuse and nid not in outputs and us:
            gs = {id(group_of.get(u)) for u in us}
            first = group_of.get(us[0])
            if len(gs) == 1 and first is not None and first.kind in ("ew", "row"):
                if _try_join(first, n, g):
                    first.nodes.add(nid)
                    group_of[nid] = first
                    joined = True
        if not joined:
            if n.is_reduce():
                grp = Group("row", nid, {nid}, g[n.inputs[0]].shape)
            else:
                grp = Group("ew", nid, {nid}, n.shape)
            groups.append(grp)
            group_of[nid] = grp
    groups.sort(key=lambda gr: gr.root)
    return groups


def group_leaves(g: Graph, grp: Group) -> list[int]:
    """Node ids read by the group that are not produced inside it (in first-use order)."""
    leaves: list[int] = []
    for nid in grp.ordered():
        for i in g[nid].inputs:
            if i not in grp.nodes and i not in leaves:
                leaves.append(i)
    return leaves


# ------------------------------------------------------------------ buffer planning
@dataclass
class BufferPlan:
    slot_of: dict[int, int]  # materialized value id -> slot index
    slot_bytes: list[int]
    kind_of_slot: list[str]  # "input" | "const" | "output" | "temp"
    naive_temp_bytes: int
    pooled_temp_bytes: int


def plan_buffers(g: Graph, groups: list[Group]) -> BufferPlan:
    # last kernel index reading each materialized value
    last_use: dict[int, int] = {}
    for k, grp in enumerate(groups):
        for leaf in group_leaves(g, grp):
            n = g[leaf]
            if is_inline_const(n):
                continue
            base, _ = resolve_view(g, leaf)
            last_use[base] = k
    slot_of: dict[int, int] = {}
    slot_bytes: list[int] = []
    kinds: list[str] = []

    def new_slot(nbytes: int, kind: str) -> int:
        slot_bytes.append(nbytes)
        kinds.append(kind)
        return len(slot_bytes) - 1

    for n in g:
        if n.op == "input":
            slot_of[n.id] = new_slot(n.nbytes, "input")
        elif n.op == "const" and n.id in last_use:
            slot_of[n.id] = new_slot(n.nbytes, "const")
    outputs = set(g.outputs)
    free: list[int] = []
    naive = 0
    for k, grp in enumerate(groups):
        root = g[grp.root]
        if grp.root in outputs:
            slot_of[grp.root] = new_slot(root.nbytes, "output")
        else:
            naive += root.nbytes
            best = None
            for s in free:
                if slot_bytes[s] >= root.nbytes and (best is None or slot_bytes[s] < slot_bytes[best]):
                    best = s
            if best is None:
                best = new_slot(root.nbytes, "temp")
            else:
                free.remove(best)
            slot_of[grp.root] = best
        # release temps whose last use is this kernel (after the output is
        # allocated, so a kernel never reads and writes the same buffer)
        for v, lu in last_use.items():
            if lu == k and v in slot_of and kinds[slot_of[v]] == "temp":
                free.append(slot_of[v])
        # values with no reader at all are dead right away
        if grp.root not in last_use and grp.root not in outputs:
            free.append(slot_of[grp.root])
    pooled = sum(b for b, kd in zip(slot_bytes, kinds) if kd == "temp")
    return BufferPlan(slot_of, slot_bytes, kinds, naive, pooled)
