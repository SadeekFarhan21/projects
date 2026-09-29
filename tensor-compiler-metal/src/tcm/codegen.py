"""Metal Shading Language code generation for the three kernel families.

  ew      fused elementwise kernel over an output domain, vectorized 4 wide
          along the last axis when it divides evenly (float4/half4 loads).
  row     fused reduction kernel: TPR threads cooperate on one row of the last
          axis, reductions use simd_sum/simd_max plus a threadgroup exchange
          when TPR > 32. Elementwise producers are recomputed inside each row
          loop instead of being stored.
  matmul  tiled matmul with threadgroup memory. Tile sizes (BM, BN, BK) and the
          per-thread register tile (TM, TN) are parameters the autotuner picks.

All arithmetic is done in float; f16 values are converted on load and store.
Shapes are static, so every size and stride is baked into the source as a
literal and the pipeline cache is keyed by source text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ir import F16, Graph, numel
from .schedule import Group, _pad, group_leaves, is_inline_const, resolve_view

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n\n"

UNARY = {
    "neg": "(-{0})",
    "exp": "exp({0})",
    "log": "log({0})",
    "sqrt": "sqrt({0})",
    "rsqrt": "rsqrt({0})",
    "tanh": "tanh({0})",
    "abs": "abs({0})",
    "recip": "(1.0f / {0})",
    "copy": "{0}",
}
BINARY = {
    "add": "({0} + {1})",
    "sub": "({0} - {1})",
    "mul": "({0} * {1})",
    "div": "({0} / {1})",
    "maximum": "max({0}, {1})",
    "minimum": "min({0}, {1})",
}


def ctype(dtype: str) -> str:
    return "half" if dtype == F16 else "float"


def flit(v: float) -> str:
    v = float(v)
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(-INFINITY)"
    return f"{np.float32(v)!r}".replace("np.float32(", "").rstrip(")") + "f"


@dataclass
class KernelSpec:
    kind: str
    name: str
    root: int
    nodes: list[int]
    args: list[int]  # materialized base value ids bound to buffer(0..n-2); output is last
    source: str = ""
    groups: tuple[int, int, int] = (1, 1, 1)
    threads: tuple[int, int, int] = (1, 1, 1)
    config: dict = field(default_factory=dict)
    flops: float = 0.0
    bytes: float = 0.0
    tune_key: str = ""


@dataclass
class Leaf:
    nid: int
    literal: str | None  # inline scalar constant
    arg: int  # buffer argument index (-1 for literals)
    strides: tuple[int, ...]  # aligned to the kernel domain, 0 on broadcast dims
    dtype: str


def _leaf_table(g: Graph, grp: Group, domain: tuple[int, ...]):
    leaves = group_leaves(g, grp)
    args: list[int] = []
    table: dict[int, Leaf] = {}
    R = len(domain)
    for nid in leaves:
        n = g[nid]
        if is_inline_const(n):
            table[nid] = Leaf(nid, flit(n.attrs["value"].reshape(-1)[0]), -1, (0,) * R, n.dtype)
            continue
        base, strides = resolve_view(g, nid)
        if base not in args:
            args.append(base)
        vshape = _pad(n.shape, R)
        vstr = (0,) * (R - len(strides)) + tuple(strides)
        st = tuple(0 if (vs == 1) else s for vs, s, d in zip(vshape, vstr, domain))
        table[nid] = Leaf(nid, None, args.index(base), st, g[base].dtype)
    return table, args


def _offset(idx_names: list[str], strides: tuple[int, ...], domain: tuple[int, ...], extra: str = "") -> str:
    terms = []
    for name, s, d in zip(idx_names, strides, domain):
        if s == 0 or d == 1 or name is None:
            continue
        terms.append(name if s == 1 else f"{name} * {s}u")
    if extra:
        terms.append(extra)
    return " + ".join(terms) if terms else "0u"


def _load(leaf: Leaf, off: str, vec: int, last_stride: int) -> str:
    T = ctype(leaf.dtype)
    p = f"a{leaf.arg}"
    if vec == 1:
        return f"float({p}[{off}])"
    if last_stride == 1:
        return f"float4(*((device const packed_{T}4*)({p} + {off})))"
    if last_stride == 0:
        return f"float4(float({p}[{off}]))"
    s = last_stride
    return (
        f"float4(float({p}[{off}]), float({p}[{off} + {s}u]), "
        f"float({p}[{off} + {2 * s}u]), float({p}[{off} + {3 * s}u]))"
    )


def _expr(g: Graph, nid: int, env: dict[int, str], vec: int) -> str:
    n = g[nid]
    a = [env[i] for i in n.inputs]
    if n.op in UNARY:
        return UNARY[n.op].format(*a)
    if n.op in BINARY:
        return BINARY[n.op].format(*a)
    if n.op == "cast":
        if n.dtype == F16:
            return f"float4(half4({a[0]}))" if vec == 4 else f"float(half({a[0]}))"
        return a[0]
    raise NotImplementedError(n.op)


def _args_decl(g: Graph, args: list[int], out_dtype: str) -> list[str]:
    decl = []
    for i, base in enumerate(args):
        decl.append(f"device const {ctype(g[base].dtype)}* a{i} [[buffer({i})]]")
    decl.append(f"device {ctype(out_dtype)}* out [[buffer({len(args)})]]")
    return decl


def _vt(vec: int) -> str:
    return "float4" if vec == 4 else "float"


# ------------------------------------------------------------------ elementwise
def gen_ew(g: Graph, grp: Group, name: str, config: dict | None = None) -> KernelSpec:
    config = dict(config or {})
    root = g[grp.root]
    D = root.shape
    R = len(D)
    total = numel(D)
    vec = 4 if (D[-1] % 4 == 0 and config.get("vec", 4) == 4) else 1
    tg = int(config.get("tg", 256))
    config.update(vec=vec, tg=tg)
    table, args = _leaf_table(g, grp, D)
    nthreads = total // vec
    lines = [f"  if (gid >= {nthreads}u) return;", f"  uint rem = gid * {vec}u;"]
    idx = [f"i{d}" for d in range(R)]
    for d in reversed(range(R)):
        if d == 0:
            lines.append(f"  uint i0 = rem;")
        else:
            lines.append(f"  uint i{d} = rem % {D[d]}u; rem /= {D[d]}u;")
    env: dict[int, str] = {}
    for nid, leaf in table.items():
        if leaf.literal is not None:
            env[nid] = f"float4({leaf.literal})" if vec == 4 else leaf.literal
        else:
            off = _offset(idx, leaf.strides, D)
            lines.append(f"  {_vt(vec)} l{nid} = {_load(leaf, off, vec, leaf.strides[-1])};")
            env[nid] = f"l{nid}"
    for nid in grp.ordered():
        lines.append(f"  {_vt(vec)} v{nid} = {_expr(g, nid, env, vec)};")
        env[nid] = f"v{nid}"
    OT = ctype(root.dtype)
    if vec == 4:
        lines.append(f"  *((device packed_{OT}4*)(out + gid * 4u)) = {OT}4(v{root.id});")
    else:
        lines.append(f"  out[gid] = {OT}(v{root.id});")
    decl = _args_decl(g, args, root.dtype) + ["uint gid [[thread_position_in_grid]]"]
    src = HEADER + f"kernel void {name}(\n    " + ",\n    ".join(decl) + ") {\n" + "\n".join(lines) + "\n}\n"
    ngroups = (nthreads + tg - 1) // tg
    nbytes = sum(g[b].nbytes for b in args) + root.nbytes
    flops = total * len(grp.nodes)
    spec = KernelSpec("ew", name, root.id, grp.ordered(), args + [root.id], src, (ngroups, 1, 1), (tg, 1, 1), config)
    spec.flops, spec.bytes = float(flops), float(nbytes)
    return spec


# ------------------------------------------------------------------ row reductions
def default_tpr(n: int, vec: int) -> int:
    per_thread = 4
    t = 32
    while t < 1024 and t * vec * per_thread < n:
        t *= 2
    return t


def gen_row(g: Graph, grp: Group, name: str, config: dict | None = None) -> KernelSpec:
    config = dict(config or {})
    root = g[grp.root]
    RD = grp.domain
    R = len(RD)
    N = RD[-1]
    L = RD[:-1]
    rows = numel(L)
    vec = 4 if (N % 4 == 0 and config.get("vec", 4) == 4) else 1
    tpr = int(config.get("tpr", default_tpr(N, vec)))
    rpg = int(config.get("rpg", 4 if tpr == 32 else 1)) if tpr == 32 else 1
    config.update(vec=vec, tpr=tpr, rpg=rpg)
    nsg = tpr // 32
    table, args = _leaf_table(g, grp, RD)

    def rowvalued(nid: int) -> bool:
        return _pad(g[nid].shape, R)[-1] == N and N != 1

    lines = [
        f"  uint sub = tid_tg / {tpr}u;",
        f"  uint tid = tid_tg % {tpr}u;",
        f"  uint row = tgid * {rpg}u + sub;",
        f"  if (row >= {rows}u) return;",
        "  uint rem = row;",
    ]
    idx: list[str | None] = [f"i{d}" for d in range(R - 1)] + [None]
    for d in reversed(range(R - 1)):
        if d == 0:
            lines.append("  uint i0 = rem;")
        else:
            lines.append(f"  uint i{d} = rem % {RD[d]}u; rem /= {RD[d]}u;")
    lines.append("  (void)rem;")
    if nsg > 1:
        lines.append(f"  threadgroup float sh[{nsg}];")
    # per-leaf lead offsets
    scal: dict[int, str] = {}  # per-row scalars (float)
    for nid, leaf in table.items():
        if leaf.literal is not None:
            continue
        lines.append(f"  uint b{nid} = {_offset(idx, leaf.strides, RD)};")
        if not rowvalued(nid):
            lines.append(f"  float l{nid} = float(a{leaf.arg}[b{nid}]);")
            scal[nid] = f"l{nid}"
    for nid, leaf in table.items():
        if leaf.literal is not None:
            scal[nid] = leaf.literal

    members = set(grp.nodes)

    def emit_loop_body(target: int, body: list[str]) -> str:
        """Emit row-valued computation of `target` inside a column loop."""
        env: dict[int, str] = {}
        for k, v in scal.items():
            env[k] = f"float4({v})" if vec == 4 else v
        needed: list[int] = []

        def visit(nid: int) -> None:
            if nid in env or nid in needed:
                return
            if nid in table:  # row-valued leaf
                leaf = table[nid]
                st = leaf.strides[-1]
                off = f"b{nid} + c" if st == 1 else f"b{nid} + c * {st}u"
                body.append(f"    {_vt(vec)} l{nid} = {_load(leaf, off, vec, st)};")
                env[nid] = f"l{nid}"
                return
            assert nid in members and rowvalued(nid), nid
            for i in g[nid].inputs:
                visit(i)
            needed.append(nid)
            body.append(f"    {_vt(vec)} v{nid} = {_expr(g, nid, env, vec)};")
            env[nid] = f"v{nid}"

        visit(target)
        return env[target]

    loop_head = f"  for (uint c = tid * {vec}u; c < {N}u; c += {tpr * vec}u) {{"
    for nid in grp.ordered():
        n = g[nid]
        if n.is_reduce():
            is_max = n.op == "reduce_max"
            init = "(-INFINITY)" if is_max else "0.0f"
            lines.append(f"  {_vt(vec)} acc{nid} = {_vt(vec)}({init});")
            lines.append(loop_head)
            body: list[str] = []
            v = emit_loop_body(n.inputs[0], body)
            lines.extend(body)
            lines.append(f"    acc{nid} = max(acc{nid}, {v});" if is_max else f"    acc{nid} += {v};")
            lines.append("  }")
            a = f"acc{nid}"
            if vec == 4:
                part = f"max(max({a}.x, {a}.y), max({a}.z, {a}.w))" if is_max else f"(({a}.x + {a}.y) + ({a}.z + {a}.w))"
            else:
                part = a
            red = "simd_max" if is_max else "simd_sum"
            lines.append(f"  float s{nid} = {red}({part});")
            if nsg > 1:
                lines += [
                    f"  if ((tid & 31u) == 0u) sh[tid >> 5] = s{nid};",
                    "  threadgroup_barrier(mem_flags::mem_threadgroup);",
                    f"  s{nid} = {red}((tid & 31u) < {nsg}u ? sh[tid & 31u] : {init});",
                    "  threadgroup_barrier(mem_flags::mem_threadgroup);",
                ]
            scal[nid] = f"s{nid}"
        elif not rowvalued(nid):
            env = dict(scal)
            lines.append(f"  float s{nid} = {_expr(g, nid, env, 1)};")
            scal[nid] = f"s{nid}"
    OT = ctype(root.dtype)
    if rowvalued(root.id):
        lines.append(loop_head)
        body = []
        v = emit_loop_body(root.id, body)
        lines.extend(body)
        if vec == 4:
            lines.append(f"    *((device packed_{OT}4*)(out + row * {N}u + c)) = {OT}4({v});")
        else:
            lines.append(f"    out[row * {N}u + c] = {OT}({v});")
        lines.append("  }")
    else:
        lines.append(f"  if (tid == 0u) out[row] = {OT}({scal[root.id]});")
    decl = _args_decl(g, args, root.dtype) + [
        "uint tid_tg [[thread_index_in_threadgroup]]",
        "uint tgid [[threadgroup_position_in_grid]]",
    ]
    src = HEADER + f"kernel void {name}(\n    " + ",\n    ".join(decl) + ") {\n" + "\n".join(lines) + "\n}\n"
    ngroups = (rows + rpg - 1) // rpg
    spec = KernelSpec("row", name, root.id, grp.ordered(), args + [root.id], src, (ngroups, 1, 1), (tpr * rpg, 1, 1), config)
    spec.bytes = float(sum(g[b].nbytes for b in args) + root.nbytes)
    spec.flops = float(numel(RD) * len(grp.nodes))
    return spec


# ------------------------------------------------------------------ matmul
DEFAULT_MM = {"BM": 64, "BN": 64, "BK": 16, "TM": 4, "TN": 4}


def mm_config_ok(c: dict, max_threads: int = 1024, max_tg_mem: int = 32768) -> bool:
    BM, BN, BK, TM, TN = (c[k] for k in ("BM", "BN", "BK", "TM", "TN"))
    if BM % TM or BN % TN:
        return False
    nt = (BM // TM) * (BN // TN)
    if nt < 32 or nt > max_threads or nt % 32:
        return False
    return (BM * BK + BK * BN) * 4 <= max_tg_mem


def gen_matmul(g: Graph, grp: Group, name: str, config: dict | None = None) -> KernelSpec:
    c = dict(DEFAULT_MM)
    c.update(config or {})
    if not mm_config_ok(c):
        raise ValueError(f"bad matmul config {c}")
    BM, BN, BK, TM, TN = (c[k] for k in ("BM", "BN", "BK", "TM", "TN"))
    root = g[grp.root]
    a_id, b_id = g[root.id].inputs
    an, bn = g[a_id], g[b_id]
    M, K = an.shape[-2], an.shape[-1]
    N = bn.shape[-1]
    batch = root.shape[0] if len(root.shape) == 3 else 1
    a_base, a_st = resolve_view(g, a_id)
    b_base, b_st = resolve_view(g, b_id)
    sab = a_st[0] if len(a_st) == 3 else 0
    sbb = b_st[0] if len(b_st) == 3 else 0
    sam, sak = a_st[-2], a_st[-1]
    sbk, sbn = b_st[-2], b_st[-1]
    args = [a_base] if a_base == b_base else [a_base, b_base]
    ai, bi = 0, (0 if a_base == b_base else 1)
    TA, TB, TO = ctype(g[a_base].dtype), ctype(g[b_base].dtype), ctype(root.dtype)
    NT = (BM // TM) * (BN // TN)
    full = M % BM == 0 and N % BN == 0 and K % BK == 0
    # choose the cooperative load order that walks the unit-stride axis fastest
    a_kfast = sak == 1 or sam != 1
    b_nfast = sbn == 1 or sbk != 1

    def guard(cond: str, expr: str) -> str:
        return expr if full else f"(({cond}) ? {expr} : 0.0f)"

    a_idx = "uint r = l / {BK}u, cc = l % {BK}u;" if a_kfast else "uint r = l % {BM}u, cc = l / {BM}u;"
    b_idx = "uint r = l / {BN}u, cc = l % {BN}u;" if b_nfast else "uint r = l % {BK}u, cc = l / {BK}u;"
    a_idx = a_idx.format(BK=BK, BM=BM)
    b_idx = b_idx.format(BN=BN, BK=BK)
    src = HEADER + f"""kernel void {name}(
    device const {TA}* a{ai} [[buffer({ai})]],
""" + (f"    device const {TB}* a{bi} [[buffer({bi})]],\n" if bi != ai else "") + f"""    device {TO}* out [[buffer({len(args)})]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {{
  threadgroup float As[{BK}][{BM}];
  threadgroup float Bs[{BK}][{BN}];
  const uint tx = tid % {BN // TN}u, ty = tid / {BN // TN}u;
  const uint m0 = tg.y * {BM}u, n0 = tg.x * {BN}u;
  device const {TA}* A = a{ai} + tg.z * {sab}u;
  device const {TB}* B = a{bi} + tg.z * {sbb}u;
  float acc[{TM}][{TN}];
  for (uint i = 0; i < {TM}u; ++i) for (uint j = 0; j < {TN}u; ++j) acc[i][j] = 0.0f;
  for (uint k0 = 0; k0 < {K}u; k0 += {BK}u) {{
    for (uint l = tid; l < {BM * BK}u; l += {NT}u) {{
      {a_idx}
      uint gm = m0 + r, gk = k0 + cc;
      As[cc][r] = {guard(f"gm < {M}u && gk < {K}u", f"float(A[gm * {sam}u + gk * {sak}u])")};
    }}
    for (uint l = tid; l < {BK * BN}u; l += {NT}u) {{
      {b_idx}
      uint gk = k0 + r, gn = n0 + cc;
      Bs[r][cc] = {guard(f"gk < {K}u && gn < {N}u", f"float(B[gk * {sbk}u + gn * {sbn}u])")};
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < {BK}u; ++kk) {{
      float av[{TM}], bv[{TN}];
      for (uint i = 0; i < {TM}u; ++i) av[i] = As[kk][ty + i * {BM // TM}u];
      for (uint j = 0; j < {TN}u; ++j) bv[j] = Bs[kk][tx + j * {BN // TN}u];
      for (uint i = 0; i < {TM}u; ++i)
        for (uint j = 0; j < {TN}u; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  device {TO}* C = out + tg.z * {M * N}u;
  for (uint i = 0; i < {TM}u; ++i) {{
    uint gm = m0 + ty + i * {BM // TM}u;
    for (uint j = 0; j < {TN}u; ++j) {{
      uint gn = n0 + tx + j * {BN // TN}u;
      {"" if full else f"if (gm < {M}u && gn < {N}u) "}C[gm * {N}u + gn] = {TO}(acc[i][j]);
    }}
  }}
}}
"""
    groups = ((N + BN - 1) // BN, (M + BM - 1) // BM, batch)
    spec = KernelSpec("matmul", name, root.id, [root.id], args + [root.id], src, groups, (NT, 1, 1), c)
    spec.flops = 2.0 * batch * M * N * K
    spec.bytes = float(sum(g[b].nbytes for b in args) + root.nbytes)
    spec.tune_key = f"matmul|b{batch}m{M}n{N}k{K}|{root.dtype}|a{sab},{sam},{sak}|b{sbb},{sbk},{sbn}"
    return spec


def generate(g: Graph, grp: Group, name: str, config: dict | None = None) -> KernelSpec:
    if grp.kind == "ew":
        spec = gen_ew(g, grp, name, config)
        spec.tune_key = f"ew|{g[grp.root].shape}|{g[grp.root].dtype}"
    elif grp.kind == "row":
        spec = gen_row(g, grp, name, config)
        spec.tune_key = f"row|{grp.domain}|{g[grp.root].dtype}|n{len(grp.nodes)}"
    else:
        spec = gen_matmul(g, grp, name, config)
    return spec
