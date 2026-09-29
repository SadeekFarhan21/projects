"""YAML task format.

A task file looks like this (see tasks/ for generated examples):

    id: sql-chinook-0007
    family: sql
    difficulty: medium            # easy | medium | hard
    tags: [chinook, join, count]
    prompt: |
      How many tracks are in the genre 'Jazz'? ...
    fixtures:                     # materialized into the episode workspace, in order
      - copy: data/chinook.sqlite # path relative to PROJECT_ROOT (file or directory)
        to: db.sqlite
      - path: NOTES.md            # inline file
        content: "..."
      - patch: toolz/itertoolz.py # exact edit applied to an already materialized file
        offset: 1234
        old: "<"
        new: "<="
    allowed_tools: [run_sql, read_file, list_dir, submit]
    verifier:
      type: python                # python | command
      function: agentenv.verifiers:numeric_or_text_answer
      args: {expected: 18, rel_tol: 1.0e-4, abs_tol: 0.01}
    timeout_s: 300                # wall clock for the whole episode
    limits: {step_limit: 10, token_budget: 3000}
    reference: {sql: "SELECT ..."} # used by the reference agent only
    metadata: {...}

A command verifier instead looks like:

    verifier:
      type: command
      command: [python, -m, pytest, -q, _hidden]
      hidden: [{copy: data/packages/..., to: _hidden/test_x.py}]
      timeout_s: 60
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import PROJECT_ROOT

ALL_TOOLS = ("run_sql", "run_python", "read_file", "write_file", "edit_file", "list_dir", "submit")
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_IGNORE = ("__pycache__", "*.pyc", ".pytest_cache", ".git")


class _Dumper(yaml.SafeDumper):
    pass


def _str_repr(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data and not any(c in data for c in "\t\r") else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_repr)


class TaskError(ValueError):
    pass


@dataclass
class Task:
    id: str
    family: str
    prompt: str
    fixtures: list[dict[str, Any]]
    allowed_tools: list[str]
    verifier: dict[str, Any]
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    timeout_s: float = 300.0
    limits: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- validation -------------------------------------------------
    def validate(self) -> "Task":
        if not self.id or not self.family:
            raise TaskError("task needs id and family")
        if self.difficulty not in DIFFICULTIES:
            raise TaskError(f"{self.id}: difficulty {self.difficulty!r} not in {DIFFICULTIES}")
        bad = [t for t in self.allowed_tools if t not in ALL_TOOLS]
        if bad:
            raise TaskError(f"{self.id}: unknown tools {bad}")
        if "submit" not in self.allowed_tools:
            raise TaskError(f"{self.id}: submit must be allowed")
        vt = self.verifier.get("type")
        if vt == "python":
            if ":" not in self.verifier.get("function", ""):
                raise TaskError(f"{self.id}: python verifier needs module:function")
        elif vt == "command":
            if not self.verifier.get("command"):
                raise TaskError(f"{self.id}: command verifier needs command")
        else:
            raise TaskError(f"{self.id}: verifier type {vt!r} not python|command")
        for fx in self.fixtures:
            kinds = [k for k in ("copy", "path", "patch") if k in fx]
            if len(kinds) != 1:
                raise TaskError(f"{self.id}: fixture must have exactly one of copy/path/patch: {fx}")
        if self.timeout_s <= 0:
            raise TaskError(f"{self.id}: timeout must be positive")
        return self

    @property
    def step_limit(self) -> int:
        return int(self.limits.get("step_limit", 12))

    @property
    def token_budget(self) -> int:
        return int(self.limits.get("token_budget", 4000))

    # ---- (de)serialization -----------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise TaskError(f"unknown task keys {sorted(extra)}")
        return cls(**d).validate()

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml())


def load_tasks(root: str | Path, family: str | None = None) -> list[Task]:
    root = Path(root)
    paths = sorted(root.rglob("*.yaml"))
    tasks = [Task.load(p) for p in paths]
    if family:
        tasks = [t for t in tasks if t.family == family]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise TaskError("duplicate task ids")
    return tasks


# ---- fixture materialization ---------------------------------------------

def _resolve_src(p: str) -> Path:
    src = Path(p)
    return src if src.is_absolute() else PROJECT_ROOT / src


def _safe_dest(root: Path, rel: str) -> Path:
    dest = (root / rel).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise TaskError(f"fixture path escapes workspace: {rel}")
    return dest


def materialize(fixtures: list[dict[str, Any]], root: Path) -> None:
    """Write fixtures into root in order. Patches must match exactly or we fail loudly."""
    root.mkdir(parents=True, exist_ok=True)
    for fx in fixtures:
        if "copy" in fx:
            src = _resolve_src(fx["copy"])
            dest = _safe_dest(root, fx.get("to", src.name))
            ignore = tuple(DEFAULT_IGNORE) + tuple(fx.get("exclude", ()))
            if src.is_dir():
                shutil.copytree(
                    src, dest, dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names
                                             if any(fnmatch.fnmatch(n, g) for g in ignore)],
                )
            elif src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            else:
                raise TaskError(f"fixture source missing: {src} (run scripts/download_data.sh)")
        elif "path" in fx:
            dest = _safe_dest(root, fx["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(fx.get("content", ""))
        else:
            dest = _safe_dest(root, fx["patch"])
            text = dest.read_text()
            off, old, new = int(fx["offset"]), fx["old"], fx["new"]
            if text[off:off + len(old)] != old:
                raise TaskError(f"patch does not apply to {fx['patch']} at {off}")
            dest.write_text(text[:off] + new + text[off + len(old):])
