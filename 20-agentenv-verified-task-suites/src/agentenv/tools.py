"""Agent tools. Everything that executes code goes through the sandbox.

File tools (read_file, write_file, edit_file, list_dir) run in the parent
process but are confined to the workspace by resolving paths and refusing
anything that escapes it (including through symlinks).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .sandbox import Limits, run_sandboxed

MAX_OBS_CHARS = 2500
MAX_READ_LINES = 150

TOOL_SPECS: dict[str, str] = {
    "run_sql": '{"tool": "run_sql", "args": {"query": "<one SQLite SELECT statement>"}}  runs against db.sqlite (read only), returns up to 30 rows',
    "run_python": '{"tool": "run_python", "args": {"code": "<python source>"}}  runs with the workspace as cwd, returns stdout and stderr',
    "read_file": '{"tool": "read_file", "args": {"path": "<file>", "start_line": 1, "end_line": 80}}  line numbered, at most 150 lines per call',
    "write_file": '{"tool": "write_file", "args": {"path": "<file>", "content": "<full new content>"}}',
    "edit_file": '{"tool": "edit_file", "args": {"path": "<file>", "old": "<exact text to replace>", "new": "<replacement>"}}  old must occur exactly once',
    "list_dir": '{"tool": "list_dir", "args": {"path": "."}}',
    "submit": '{"tool": "submit", "args": {"answer": "<final answer, or empty when you fixed files>"}}  ends the episode',
}


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    output: str
    ok: bool = True
    done: bool = False
    submission: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


def _truncate(s: str, n: int = MAX_OBS_CHARS) -> str:
    if len(s) <= n:
        return s
    return s[: n - 60] + f"\n... [truncated {len(s) - n + 60} chars]"


_SQL_RUNNER = r"""
import json, sqlite3, sys
q = sys.stdin.read()
con = sqlite3.connect("file:db.sqlite?mode=ro", uri=True)
cur = con.execute(q)
cols = [d[0] for d in cur.description] if cur.description else []
rows = cur.fetchmany(31)
print(json.dumps({"columns": cols, "rows": rows[:30], "more": len(rows) > 30}, default=str))
"""


class Toolbox:
    def __init__(self, workspace: Path, allowed: list[str], limits: Limits = Limits()):
        self.ws = Path(workspace).resolve()
        self.allowed = list(allowed)
        self.limits = limits
        self.calls = 0
        self._impl: dict[str, Callable[..., ToolResult]] = {
            "run_sql": self.run_sql,
            "run_python": self.run_python,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_dir": self.list_dir,
            "submit": self.submit,
        }

    def describe(self) -> str:
        return "\n".join(f"- {TOOL_SPECS[t]}" for t in self.allowed)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls += 1
        if name not in self._impl:
            return ToolResult(f"error: unknown tool {name!r}. Allowed: {self.allowed}", ok=False,
                              meta={"error": "unknown_tool"})
        if name not in self.allowed:
            return ToolResult(f"error: tool {name!r} is not allowed here. Allowed: {self.allowed}",
                              ok=False, meta={"error": "disallowed_tool"})
        if not isinstance(args, dict):
            return ToolResult("error: args must be a JSON object", ok=False, meta={"error": "bad_args"})
        try:
            return self._impl[name](**args)
        except TypeError as e:
            return ToolResult(f"error: bad arguments for {name}: {e}", ok=False, meta={"error": "bad_args"})
        except ToolError as e:
            return ToolResult(f"error: {e}", ok=False, meta={"error": "tool_error"})

    # ---- helpers -----------------------------------------------------
    def _path(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel:
            raise ToolError("path must be a non-empty string")
        p = (self.ws / rel).resolve()
        if not p.is_relative_to(self.ws):
            raise ToolError(f"path {rel!r} is outside the workspace")
        return p

    # ---- tools ---------------------------------------------------------
    def run_sql(self, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        r = run_sandboxed([sys.executable, "-B", "-c", _SQL_RUNNER], self.ws, self.limits, stdin=query)
        if r.timed_out or r.mem_exceeded:
            return ToolResult("error: query exceeded sandbox limits", ok=False,
                              meta={"error": "limit", "wall_s": r.wall_s})
        if r.returncode != 0:
            last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
            return ToolResult(f"SQL error: {last}", ok=False, meta={"error": "sql_error", "wall_s": r.wall_s})
        data = json.loads(r.stdout)
        lines = [" | ".join(data["columns"])]
        lines += [" | ".join("NULL" if v is None else str(v) for v in row) for row in data["rows"]]
        if data["more"]:
            lines.append("... (more rows)")
        if not data["rows"]:
            lines.append("(no rows)")
        return ToolResult(_truncate("\n".join(lines)), meta={"wall_s": r.wall_s,
                                                             "rows": data["rows"][:5]})

    def run_python(self, code: str) -> ToolResult:
        if not isinstance(code, str):
            raise ToolError("code must be a string")
        r = run_sandboxed([sys.executable, "-B", "-"], self.ws, self.limits, stdin=code)
        status = f"exit code {r.returncode}"
        if r.timed_out:
            status = "killed, wall clock limit"
        elif r.mem_exceeded:
            status = "killed, memory limit"
        body = f"[{status}]\n{r.stdout}"
        if r.stderr.strip():
            body += f"\n[stderr]\n{r.stderr[-1500:]}"
        return ToolResult(_truncate(body), ok=r.ok, meta={"wall_s": r.wall_s, "returncode": r.returncode})

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file {path!r}")
        lines = p.read_text(errors="replace").splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else min(len(lines), int(end_line))
        end = min(end, start + MAX_READ_LINES - 1)
        out = [f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1)]
        footer = f"[lines {start}-{end} of {len(lines)}]"
        return ToolResult(_truncate("\n".join(out) + "\n" + footer, 8000))

    def write_file(self, path: str, content: str) -> ToolResult:
        p = self._path(path)
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(f"wrote {len(content)} chars to {path}", meta={"edited": path})

    def edit_file(self, path: str, old: str, new: str) -> ToolResult:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file {path!r}")
        text = p.read_text()
        n = text.count(old) if old else 0
        if n != 1:
            raise ToolError(f"old text must occur exactly once in {path}, found {n} times")
        p.write_text(text.replace(old, new, 1))
        return ToolResult(f"edited {path}", meta={"edited": path})

    def list_dir(self, path: str = ".") -> ToolResult:
        p = self._path(path)
        if not p.is_dir():
            raise ToolError(f"no such directory {path!r}")
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()
                       if not x.name.startswith("."))
        return ToolResult("\n".join(names) or "(empty)")

    def submit(self, answer: Any = None) -> ToolResult:
        return ToolResult("submitted", done=True, submission=answer)
