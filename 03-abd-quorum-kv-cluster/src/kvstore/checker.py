"""Linearizability checker for a read/write register (Knossos-style, brute force).

A history is a list of completed or pending operations with real-time
intervals [call, ret]. The history is linearizable if there is a total order
of the operations that (1) respects real time: if a returned before b was
invoked, a comes first, and (2) is legal for a register: every read returns
the value of the latest preceding write (or the initial value).

Algorithm: Wing and Gong's search with Lowe's memoization (the core of
Knossos's "linear" / WGL checker). We repeatedly pick a *minimal* operation
(one that no other remaining operation must precede), apply it to the model,
and backtrack on failure. The cache of (set of linearized ops, register value)
states that already failed turns a factorial search into something that is
fine for the few hundred ops per key our tests produce.

Operation outcomes, following Jepsen:
  ok   - the op happened, with the observed result.
  fail - the op definitely did not happen; dropped.
  info - unknown (e.g. timeout). A write may or may not have taken effect, at
         any point after its invocation; we model it as ret = +inf and allow
         the search to leave it out entirely. Info reads carry no information
         and are dropped.

Linearizability is compositional (local), so a multi-key history is checked
one key at a time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Op:
    process: int
    f: str  # "read" | "write"
    key: str
    value: Any  # written value, or value returned by a read
    call: float
    ret: float  # math.inf for info ops
    status: str = "ok"  # "ok" | "fail" | "info"


@dataclass
class CheckResult:
    ok: bool
    key: str | None = None
    ops_checked: int = 0
    states_explored: int = 0
    detail: str = ""


def _prepare(ops: list[Op]) -> list[Op]:
    out = []
    for o in ops:
        if o.status == "fail":
            continue
        if o.status == "info":
            if o.f == "read":
                continue
            o = Op(o.process, o.f, o.key, o.value, o.call, math.inf, "info")
        out.append(o)
    return out


def check_register(ops: list[Op], initial: Any = None, max_states: int = 5_000_000) -> CheckResult:
    """Check a single-key history. Returns CheckResult(ok=...)."""
    ops = sorted(_prepare(ops), key=lambda o: o.call)
    n = len(ops)
    if n == 0:
        return CheckResult(True)
    required = 0
    for i, o in enumerate(ops):
        if o.status == "ok":
            required |= 1 << i
    full = (1 << n) - 1

    failed: set[tuple[int, Any]] = set()
    explored = 0
    best_depth, best_mask = -1, 0

    # Explicit DFS stack of (mask, value, iterator-over-candidates).
    def candidates(mask: int) -> list[int]:
        # Minimal ops: invoked no later than the earliest return among the
        # remaining ops. Ops are sorted by call time, so we start at the first
        # unlinearized op and stop as soon as calls pass the running min_ret:
        # every later op has ret >= call > min_ret and cannot lower it. This
        # keeps the scan O(concurrency window) instead of O(history).
        i = ((~mask) & (mask + 1)).bit_length() - 1
        min_ret = math.inf
        out = []
        while i < n and ops[i].call <= min_ret:
            if not (mask >> i) & 1:
                out.append(i)
                if ops[i].ret < min_ret:
                    min_ret = ops[i].ret
            i += 1
        # Reversed so that stack.pop() tries the earliest-invoked op first,
        # which is usually the right guess and keeps the search shallow.
        return [j for j in reversed(out) if ops[j].call <= min_ret]

    stack: list[tuple[int, Any, list[int]]] = [(0, initial, candidates(0))]
    while stack:
        mask, value, cands = stack[-1]
        if mask & required == required:
            return CheckResult(True, ops[0].key, n, explored)
        if not cands:
            failed.add((mask, _h(value)))
            stack.pop()
            continue
        j = cands.pop()
        o = ops[j]
        if o.f == "write":
            nv = o.value
        elif o.value == value:
            nv = value
        else:
            continue  # read does not match the model: prune
        nmask = mask | (1 << j)
        if (nmask, _h(nv)) in failed:
            continue
        explored += 1
        if explored > max_states:
            return CheckResult(False, ops[0].key, n, explored, "search budget exhausted (unknown)")
        depth = bin(nmask & required).count("1")
        if depth > best_depth:
            best_depth, best_mask = depth, nmask
        stack.append((nmask, nv, candidates(nmask) if nmask != full else []))

    stuck = [ops[j] for j in range(n) if not (best_mask >> j) & 1 and ops[j].status == "ok"]
    stuck.sort(key=lambda o: o.call)
    detail = f"could linearize at most {best_depth}/{bin(required).count('1')} ok ops; first unplaceable: {stuck[:3]}"
    return CheckResult(False, ops[0].key, n, explored, detail)


def _h(v: Any) -> Any:
    return v if isinstance(v, (int, str, float, type(None), bool)) else repr(v)


def check_history(ops: list[Op], initial: Any = None) -> CheckResult:
    """Check a multi-key history key by key. Returns the first failure, or ok."""
    by_key: dict[str, list[Op]] = defaultdict(list)
    for o in ops:
        by_key[o.key].append(o)
    total_ops = total_states = 0
    for key in sorted(by_key):
        r = check_register(by_key[key], initial)
        total_ops += r.ops_checked
        total_states += r.states_explored
        if not r.ok:
            r.key = key
            return r
    return CheckResult(True, None, total_ops, total_states)
