"""Verifiers decide pass or fail for an episode.

Two kinds, both declared in the task YAML:

* python: `module:function` called as fn(ctx, **args) -> VerifyResult. Used for
  answer checking, for example numeric answers with tolerance.
* command: a command run in a *fresh* verify directory that holds a copy of the
  agent's workspace plus the hidden files. Hidden files never exist in the
  agent's workspace. Pass means exit code 0.
"""

from __future__ import annotations

import importlib
import math
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .sandbox import Limits, run_sandboxed
from .task import Task, materialize


@dataclass
class VerifyResult:
    passed: bool
    detail: str = ""
    wall_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyContext:
    task: Task
    workspace: Path
    submission: Any
    submitted: bool


# ---- answer parsing ---------------------------------------------------

_NUM = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_number(x: Any) -> float | None:
    """Accept 12, "12.5", "$1,234.50", "about 3.2 minutes". Take the first number."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return parse_number(x[0])
    if isinstance(x, str):
        m = _NUM.search(x.replace("$", ""))
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None
    return None


def normalize_text(x: Any) -> str:
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    s = str(x).strip().strip("'\"").strip()
    return re.sub(r"\s+", " ", s).casefold()


def numeric_or_text_answer(ctx: VerifyContext, expected: Any, kind: str = "number",
                           rel_tol: float = 1e-4, abs_tol: float = 0.01) -> VerifyResult:
    if not ctx.submitted:
        return VerifyResult(False, "no submission", meta={"reason": "no_submit"})
    sub = ctx.submission
    if kind == "number":
        got = parse_number(sub)
        if got is None:
            return VerifyResult(False, f"could not parse a number from {sub!r}",
                                meta={"reason": "unparseable"})
        ok = math.isclose(got, float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
        return VerifyResult(ok, f"got {got} expected {expected}", meta={"got": got,
                                                                       "reason": "ok" if ok else "wrong"})
    got_s, exp_s = normalize_text(sub), normalize_text(expected)
    ok = got_s == exp_s
    return VerifyResult(ok, f"got {got_s!r} expected {exp_s!r}",
                        meta={"got": got_s, "reason": "ok" if ok else "wrong"})


# ---- command verifier ---------------------------------------------------

_PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|errors?)")


def run_command_verifier(task: Task, workspace: Path, limits: Limits | None = None) -> VerifyResult:
    v = task.verifier
    limits = limits or Limits(cpu_s=int(v.get("cpu_s", 60)), wall_s=float(v.get("timeout_s", 90)),
                              mem_mb=int(v.get("mem_mb", 2048)))
    vdir = Path(tempfile.mkdtemp(prefix="verify-"))
    try:
        shutil.copytree(workspace, vdir, dirs_exist_ok=True, symlinks=True,
                        ignore=shutil.ignore_patterns(".tmp", "__pycache__", "_hidden*"))
        materialize(v.get("hidden", []), vdir)
        argv = [sys.executable if a == "python" else a for a in v["command"]]
        r = run_sandboxed(argv, vdir, limits)
        counts: dict[str, int] = {}
        for n, kind in _PYTEST_SUMMARY.findall(r.stdout[-2000:]):
            counts[kind.rstrip("s")] = int(n)
        failing = re.findall(r"^FAILED (\S+)", r.stdout, flags=re.M)
        errors = re.findall(r"^ERROR (\S+)", r.stdout, flags=re.M)
        return VerifyResult(
            passed=r.ok,
            detail=(r.stdout[-1500:] + ("\n" + r.stderr[-500:] if r.stderr.strip() else "")),
            wall_s=r.wall_s,
            meta={"returncode": r.returncode, "timed_out": r.timed_out, "counts": counts,
                  "failing": failing[:20], "errors": errors[:20]},
        )
    finally:
        shutil.rmtree(vdir, ignore_errors=True)


def verify(task: Task, workspace: Path, submission: Any, submitted: bool) -> VerifyResult:
    t0 = time.monotonic()
    v = task.verifier
    if v["type"] == "command":
        if not submitted:
            return VerifyResult(False, "no submission", meta={"reason": "no_submit"})
        res = run_command_verifier(task, workspace)
    else:
        mod, fn = v["function"].split(":")
        func = getattr(importlib.import_module(mod), fn)
        res = func(VerifyContext(task, workspace, submission, submitted), **v.get("args", {}))
    res.wall_s = round(time.monotonic() - t0, 4)
    return res
