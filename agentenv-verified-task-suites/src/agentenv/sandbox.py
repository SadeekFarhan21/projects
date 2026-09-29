"""Per-episode process sandbox for macOS.

Every command an agent (or a verifier) runs goes through `run_sandboxed`:

* a fresh process group (start_new_session) so a timeout kills the whole tree
* RLIMIT_CPU, RLIMIT_FSIZE and RLIMIT_NOFILE set in the child before exec
* a memory watchdog: macOS refuses RLIMIT_AS and RLIMIT_DATA (setrlimit returns
  EINVAL), so the parent polls the resident set size of the process tree and
  kills it when it crosses the limit
* a wall clock limit enforced by the same polling loop
* `sandbox-exec` (Seatbelt) with a profile that denies all network access,
  denies writes outside the workspace, and denies reads under /Users except for
  the workspace and the Python installation, so hidden tests and task files in
  the project directory are not readable from inside an episode
* a scrubbed environment (HOME and TMPDIR point into the workspace)
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

SANDBOX_EXEC = shutil.which("sandbox-exec")
MAX_CAPTURE = 64_000  # bytes kept per stream


@dataclass(frozen=True)
class Limits:
    cpu_s: int = 20
    wall_s: float = 30.0
    mem_mb: int = 1024
    fsize_mb: int = 64
    nofile: int = 256
    network: bool = False


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool = False
    mem_exceeded: bool = False
    cpu_exceeded: bool = False
    peak_rss_mb: float = 0.0
    sandboxed: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not (self.timed_out or self.mem_exceeded)

    def to_dict(self) -> dict:
        return asdict(self)


def _python_read_roots() -> list[str]:
    """Directories the sandboxed Python needs to read (interpreter and venv)."""
    roots = {
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sysconfig.get_paths()["stdlib"],
        sysconfig.get_paths()["purelib"],
        str(Path(sys.executable).parent),
    }
    return sorted({os.path.realpath(r) for r in roots})


def _q(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(workspace: Path, network: bool = False) -> str:
    ws = os.path.realpath(workspace)
    reads = " ".join(f"(subpath {_q(r)})" for r in [ws, *_python_read_roots()])
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-read-data (subpath \"/Users\"))",
        f"(allow file-read-data {reads})",
        "(deny file-write*)",
        f"(allow file-write* (subpath {_q(ws)}) (literal \"/dev/null\") (literal \"/dev/tty\"))",
    ]
    if not network:
        lines.append("(deny network*)")
    return "\n".join(lines)


def _preexec(limits: Limits):
    def fn() -> None:
        cpu = max(1, int(limits.cpu_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        fs = limits.fsize_mb << 20
        resource.setrlimit(resource.RLIMIT_FSIZE, (fs, fs))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.nofile, limits.nofile))
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)

    return fn


def _reader(stream, sink: list[bytes]) -> None:
    total = 0
    for chunk in iter(lambda: stream.read(4096), b""):
        if total < MAX_CAPTURE:
            sink.append(chunk[: MAX_CAPTURE - total])
        total += len(chunk)
    stream.close()


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def sandbox_env(workspace: Path, extra: dict | None = None) -> dict:
    tmp = Path(workspace) / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(workspace),
        "TMPDIR": str(tmp),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
    }
    if extra:
        env.update(extra)
    return env


def run_sandboxed(
    argv: list[str],
    workspace: Path,
    limits: Limits = Limits(),
    stdin: str | None = None,
    env: dict | None = None,
    use_seatbelt: bool = True,
    poll_s: float = 0.02,
) -> RunResult:
    """Run argv with cwd=workspace under the limits. Never raises on child failure."""
    workspace = Path(workspace)
    notes: list[str] = []
    full = list(argv)
    sandboxed = bool(use_seatbelt and SANDBOX_EXEC)
    if sandboxed:
        full = [SANDBOX_EXEC, "-p", seatbelt_profile(workspace, limits.network), *full]
    elif use_seatbelt:
        notes.append("sandbox-exec not found, running without Seatbelt")

    t0 = time.monotonic()
    proc = subprocess.Popen(
        full,
        cwd=workspace,
        env=env if env is not None else sandbox_env(workspace),
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=_preexec(limits),
        start_new_session=True,
    )
    out: list[bytes] = []
    err: list[bytes] = []
    threads = [
        threading.Thread(target=_reader, args=(proc.stdout, out), daemon=True),
        threading.Thread(target=_reader, args=(proc.stderr, err), daemon=True),
    ]
    for t in threads:
        t.start()
    if stdin is not None:
        try:
            proc.stdin.write(stdin.encode())
            proc.stdin.close()
        except BrokenPipeError:
            pass

    ps = psutil.Process(proc.pid)
    timed_out = mem_exceeded = False
    peak = 0
    mem_cap = limits.mem_mb << 20
    next_mem_check = 0.0
    while proc.poll() is None:
        now = time.monotonic()
        if now - t0 > limits.wall_s:
            timed_out = True
            _kill_group(proc.pid)
            break
        if now >= next_mem_check:
            rss = _tree_rss(ps)
            peak = max(peak, rss)
            if rss > mem_cap:
                mem_exceeded = True
                _kill_group(proc.pid)
                break
            next_mem_check = now + 0.05
        time.sleep(poll_s)
    rc = proc.wait()
    _kill_group(proc.pid)  # reap stragglers that outlived the leader
    for t in threads:
        t.join(timeout=2)
    wall = time.monotonic() - t0
    cpu_exceeded = rc in (-signal.SIGXCPU, -signal.SIGKILL) and not (timed_out or mem_exceeded)
    if rc == -signal.SIGXCPU:
        notes.append("cpu limit")
    return RunResult(
        returncode=rc,
        stdout=b"".join(out).decode("utf-8", "replace"),
        stderr=b"".join(err).decode("utf-8", "replace"),
        wall_s=round(wall, 4),
        timed_out=timed_out,
        mem_exceeded=mem_exceeded,
        cpu_exceeded=cpu_exceeded,
        peak_rss_mb=round(peak / 2**20, 1),
        sandboxed=sandboxed,
        notes=notes,
    )
