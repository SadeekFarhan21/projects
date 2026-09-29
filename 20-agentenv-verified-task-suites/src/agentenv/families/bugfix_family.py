"""Repo bug fixing tasks from mutation testing of three pure-Python packages.

For each package we copy the pristine source into the workspace, apply one
mutant (see mutations.py) and keep the task only if the hidden tests for the
mutated module fail. The hidden tests live only in the verify directory. The
reference solution restores the pristine file, which is exactly the inverse
of the mutation.
"""

from __future__ import annotations

import random
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .. import PROJECT_ROOT
from ..sandbox import Limits, run_sandboxed
from ..task import Task, materialize
from .mutations import OPERATORS, Mutant, enumerate_mutants

PYTEST = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
          "-rfE", "--tb=line", "-W", "ignore::DeprecationWarning", "_hidden"]
VERIFY_LIMITS = Limits(cpu_s=60, wall_s=120, mem_mb=2048)

TOOLZ_VERSION_STUB = (
    '"""Stub replacing versioneer output so imports never shell out to git."""\n'
    "def get_versions():\n"
    '    return {"version": "1.0.0", "full-revisionid": None, "dirty": False, "error": None, "date": None}\n'
)


@dataclass
class PackageSpec:
    name: str
    fixtures: list[dict]
    targets: dict[str, list[str]]  # workspace path of module -> hidden test files (project relative)
    support: list[dict] = field(default_factory=list)  # hidden helper files (conftest etc.)
    source_of: dict[str, str] = field(default_factory=dict)  # workspace path -> pristine project path


def package_specs() -> list[PackageSpec]:
    inf = "data/packages/inflection"
    tz = "data/packages/toolz"
    sv = "data/packages/semver"
    sv_tests = sorted(p.name for p in (PROJECT_ROOT / sv / "tests").glob("*.py"))
    specs = [
        PackageSpec(
            "inflection",
            fixtures=[{"copy": f"{inf}/inflection", "to": "inflection"}],
            targets={"inflection/__init__.py": [f"{inf}/test_inflection.py"]},
            source_of={"inflection/__init__.py": f"{inf}/inflection/__init__.py"},
        ),
        PackageSpec(
            "toolz",
            fixtures=[{"copy": f"{tz}/toolz", "to": "toolz", "exclude": ["tests"]},
                      {"copy": f"{tz}/tlz", "to": "tlz"},
                      {"path": "toolz/_version.py", "content": TOOLZ_VERSION_STUB}],
            targets={
                "toolz/itertoolz.py": [f"{tz}/toolz/tests/test_itertoolz.py"],
                "toolz/dicttoolz.py": [f"{tz}/toolz/tests/test_dicttoolz.py"],
                "toolz/functoolz.py": [f"{tz}/toolz/tests/test_functoolz.py",
                                       f"{tz}/toolz/tests/test_inspect_args.py"],
                "toolz/recipes.py": [f"{tz}/toolz/tests/test_recipes.py"],
            },
            source_of={f"toolz/{m}.py": f"{tz}/toolz/{m}.py"
                       for m in ("itertoolz", "dicttoolz", "functoolz", "recipes")},
        ),
        PackageSpec(
            "semver",
            fixtures=[{"copy": f"{sv}/src/semver", "to": "semver"}],
            targets={m: [f"{sv}/tests/{t}" for t in sv_tests if t.startswith("test_")]
                     for m in ("semver/version.py", "semver/_deprecated.py")},
            support=[{"copy": f"{sv}/tests/{t}", "to": f"_hidden/{t}"} for t in sv_tests
                     if not t.startswith("test_")],
            source_of={"semver/version.py": f"{sv}/src/semver/version.py",
                       "semver/_deprecated.py": f"{sv}/src/semver/_deprecated.py"},
        ),
    ]
    return specs


def hidden_files(spec: PackageSpec, module: str) -> list[dict]:
    return spec.support + [{"copy": t, "to": f"_hidden/{Path(t).name}"} for t in spec.targets[module]]


def run_hidden(spec: PackageSpec, module: str, mutant: Mutant | None) -> dict:
    """Materialize workspace (+ optional mutant) and hidden tests, run pytest."""
    d = Path(tempfile.mkdtemp(prefix=f"mut-{spec.name}-"))
    try:
        fx = list(spec.fixtures)
        if mutant is not None:
            fx.append({"patch": module, "offset": mutant.offset, "old": mutant.old, "new": mutant.new})
        materialize(fx + hidden_files(spec, module), d)
        argv = [sys.executable if a == "python" else a for a in PYTEST]
        r = run_sandboxed(argv, d, VERIFY_LIMITS)
        fails = re.findall(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$", r.stdout, flags=re.M)
        return {"returncode": r.returncode, "ok": r.ok, "timed_out": r.timed_out,
                "failing": [f[0] for f in fails], "messages": [f[1] for f in fails],
                "tail": r.stdout[-600:], "wall_s": r.wall_s}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _candidates(spec: PackageSpec, rng: random.Random) -> list[tuple[str, Mutant]]:
    """Interleave operators and modules so kept tasks stay balanced."""
    by_op: dict[str, list[tuple[str, Mutant]]] = {op: [] for op in OPERATORS}
    for module in spec.targets:
        src = (PROJECT_ROOT / spec.source_of[module]).read_text()
        for m in enumerate_mutants(src, random.Random(rng.random())):
            by_op[m.operator].append((module, m))
    for op in by_op:
        # round robin over modules inside each operator after shuffling
        per_mod: dict[str, list] = {}
        for mod, m in by_op[op]:
            per_mod.setdefault(mod, []).append((mod, m))
        for lst in per_mod.values():
            rng.shuffle(lst)
        mixed = []
        while any(per_mod.values()):
            for mod in sorted(per_mod):
                if per_mod[mod]:
                    mixed.append(per_mod[mod].pop())
        by_op[op] = mixed
    out = []
    while any(by_op.values()):
        for op in OPERATORS:
            if by_op[op]:
                out.append(by_op[op].pop(0))
    return out


def _hint(err: str) -> str:
    err = (err or "").strip()
    return err[:300] if err else "(no message)"


def make_task(spec: PackageSpec, module: str, m: Mutant, res: dict, idx: int, difficulty: str) -> Task:
    failing = res["failing"][:3]
    msgs = [x for x in res["messages"] if x][:2]
    lines = [f"The Python package `{spec.name}` in this workspace has a bug. A hidden test suite fails.",
             "Failing tests: " + ", ".join(failing) if failing else "The hidden tests fail.",
             ]
    if msgs:
        lines.append("Failure message: " + _hint(msgs[0]))
    if difficulty == "easy":
        lines.append(f"The bug is in the function `{m.function}` in {module}, near line {m.lineno}.")
    elif difficulty == "medium":
        lines.append(f"The bug is in {module}.")
    lines.append("Find and fix the bug with a minimal edit. The hidden tests are not in the workspace, "
                 "but you can run your own checks with run_python. Call submit when you are done.")
    tid = f"bugfix-{spec.name}-{idx:03d}"
    return Task(
        id=tid, family="bugfix", difficulty=difficulty,
        tags=[spec.name, m.operator, Path(module).stem],
        prompt="\n".join(lines),
        fixtures=spec.fixtures + [{"patch": module, "offset": m.offset, "old": m.old, "new": m.new}],
        allowed_tools=["run_python", "read_file", "write_file", "edit_file", "list_dir", "submit"],
        verifier={"type": "command", "command": PYTEST, "hidden": hidden_files(spec, module),
                  "timeout_s": VERIFY_LIMITS.wall_s, "cpu_s": VERIFY_LIMITS.cpu_s,
                  "mem_mb": VERIFY_LIMITS.mem_mb},
        timeout_s=600, limits={"step_limit": 15, "token_budget": 4000},
        reference={"restore": {"path": module, "from": spec.source_of[module]}},
        metadata={"package": spec.name, "module": module, "operator": m.operator,
                  "function": m.function, "lineno": m.lineno, "mutation": m.description,
                  "failing_tests": res["failing"][:20], "n_failing": len(res["failing"])},
    ).validate()


def generate(seed: int = 0, quotas: dict[str, int] | None = None, workers: int = 8,
             log=print) -> tuple[list[Task], dict]:
    quotas = quotas or {"inflection": 34, "toolz": 33, "semver": 33}
    tasks: list[Task] = []
    stats: dict = {}
    for spec in package_specs():
        quota = quotas[spec.name]
        for module in spec.targets:
            base = run_hidden(spec, module, None)
            if not base["ok"]:
                raise RuntimeError(f"pristine {spec.name}:{module} fails its hidden tests:\n{base['tail']}")
        rng = random.Random(f"{seed}:{spec.name}")
        cands = _candidates(spec, rng)
        kept: list[tuple[str, Mutant, dict]] = []
        tried = 0
        op_stats = {op: {"tried": 0, "kept": 0} for op in OPERATORS}
        seen_edits = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            i = 0
            while len(kept) < quota and i < len(cands):
                batch = []
                while len(batch) < workers and i < len(cands):
                    mod, m = cands[i]
                    i += 1
                    key = (mod, m.offset, m.new)
                    if key in seen_edits:
                        continue
                    seen_edits.add(key)
                    batch.append((mod, m))
                results = list(pool.map(lambda c: run_hidden(spec, c[0], c[1]), batch))
                for (mod, m), res in zip(batch, results):
                    tried += 1
                    op_stats[m.operator]["tried"] += 1
                    if res["timed_out"]:
                        continue  # a mutant that hangs is not a clean task
                    if res["returncode"] != 0 and res["failing"] and len(kept) < quota:
                        kept.append((mod, m, res))
                        op_stats[m.operator]["kept"] += 1
                log(f"{spec.name}: tried {tried}, kept {len(kept)}/{quota}")
        if len(kept) < quota:
            raise RuntimeError(f"{spec.name}: only {len(kept)} killing mutants")
        for j, (mod, m, res) in enumerate(kept):
            difficulty = ("easy", "medium", "hard")[j % 3]
            tasks.append(make_task(spec, mod, m, res, j, difficulty))
        stats[spec.name] = {"candidates": len(cands), "tried": tried, "kept": len(kept),
                            "operators": op_stats}
    return tasks, stats
