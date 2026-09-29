"""AST guided source mutations that edit the original text minimally.

Each mutant is a single (offset, old, new) replacement on the original
source, so reverting it is exact and the diff is one span. Four operators:

* flip_comparison  `<` to `<=`, `==` to `!=`, `in` to `not in`, and so on
* off_by_one       integer literal n to n + 1 (or n - 1)
* wrong_variable   a read of local name a becomes another local name b
* dropped_branch   an `if` statement with no else becomes `pass`

Only code inside function bodies is mutated, docstrings are never touched,
and every mutant must still parse.
"""

from __future__ import annotations

import ast
import io
import random
import tokenize
from dataclasses import dataclass

OPERATORS = ("flip_comparison", "off_by_one", "wrong_variable", "dropped_branch")

FLIP = {
    ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
    ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="),
}


@dataclass(frozen=True)
class Mutant:
    operator: str
    offset: int
    old: str
    new: str
    lineno: int
    function: str
    description: str

    def apply(self, src: str) -> str:
        assert src[self.offset:self.offset + len(self.old)] == self.old, "mutant does not match source"
        return src[:self.offset] + self.new + src[self.offset + len(self.old):]


class _Offsets:
    def __init__(self, src: str):
        self.starts = [0]
        for line in src.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))

    def __call__(self, lineno: int, col: int) -> int:
        return self.starts[lineno - 1] + col


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _docstring_nodes(fn: ast.AST) -> set[int]:
    body = getattr(fn, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return {id(body[0].value)}
    return set()


def _own_nodes(fn: ast.AST):
    """Nodes in fn's body, not descending into nested functions or classes."""
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(c)


def _local_names(fn: ast.FunctionDef) -> list[str]:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    for n in _own_nodes(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.append(n.id)
    out = []
    for x in names:
        if x not in out and x not in ("self", "cls", "_"):
            out.append(x)
    return out


def _op_token_offset(src: str, off: _Offsets, left: ast.expr, right: ast.expr, op_text: str) -> int | None:
    start = off(left.end_lineno, left.end_col_offset)
    end = off(right.lineno, right.col_offset)
    between = src[start:end]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(between).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    ops = [t for t in toks if t.type == tokenize.OP and t.string not in ("(", ")")]
    if len(ops) != 1 or ops[0].string != op_text:
        return None
    line, col = ops[0].start
    lines = between.splitlines(keepends=True)
    return start + sum(len(x) for x in lines[: line - 1]) + col


def enumerate_mutants(src: str, rng: random.Random | None = None) -> list[Mutant]:
    rng = rng or random.Random(0)
    tree = ast.parse(src)
    off = _Offsets(src)
    lines = src.splitlines(keepends=True)
    muts: list[Mutant] = []
    for fn in _functions(tree):
        doc = _docstring_nodes(fn)
        locals_ = _local_names(fn)
        for n in _own_nodes(fn):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in FLIP:
                old, new = FLIP[type(n.ops[0])]
                o = _op_token_offset(src, off, n.left, n.comparators[0], old)
                if o is not None:
                    muts.append(Mutant("flip_comparison", o, old, new, n.lineno, fn.name,
                                       f"{old} -> {new}"))
            elif isinstance(n, ast.Constant) and id(n) not in doc and type(n.value) is int \
                    and n.end_lineno == n.lineno:
                o = off(n.lineno, n.col_offset)
                old = src[o:off(n.end_lineno, n.end_col_offset)]
                if old != str(n.value):
                    continue  # hex, underscores, etc.
                new_v = n.value - 1 if (n.value > 0 and rng.random() < 0.5) else n.value + 1
                muts.append(Mutant("off_by_one", o, old, str(new_v), n.lineno, fn.name,
                                   f"{old} -> {new_v}"))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in locals_:
                others = [x for x in locals_ if x != n.id]
                if not others:
                    continue
                b = rng.choice(others)
                o = off(n.lineno, n.col_offset)
                muts.append(Mutant("wrong_variable", o, n.id, b, n.lineno, fn.name, f"{n.id} -> {b}"))
            elif isinstance(n, ast.If) and not n.orelse:
                start = off(n.lineno, 0)
                end = off(n.end_lineno, 0) + len(lines[n.end_lineno - 1])
                old = src[start:end]
                indent = old[: len(old) - len(old.lstrip())]
                if not old.lstrip().startswith("if "):
                    continue  # e.g. an if on the same line as something else
                new = indent + "pass\n" if old.endswith("\n") else indent + "pass"
                muts.append(Mutant("dropped_branch", start, old, new, n.lineno, fn.name,
                                   f"dropped if at line {n.lineno}"))
    valid = []
    for m in muts:
        try:
            ast.parse(m.apply(src))
        except SyntaxError:
            continue
        valid.append(m)
    return valid
