from agentenv.sandbox import Limits, run_sandboxed, seatbelt_profile
from agentenv import PROJECT_ROOT

from conftest import PY

L = Limits(cpu_s=10, wall_s=20, mem_mb=400)


def run(ws, code, limits=L):
    return run_sandboxed([PY, "-c", code], ws, limits)


def test_runs_and_captures(ws):
    r = run(ws, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    assert r.returncode == 3 and r.stdout.strip() == "out" and "err" in r.stderr
    assert r.sandboxed


def test_network_denied(ws):
    r = run(ws, "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)")
    assert r.returncode != 0 and "Operation not permitted" in r.stderr


def test_writes_confined_to_workspace(ws):
    assert run(ws, "open('ok.txt', 'w').write('x')").returncode == 0
    assert (ws / "ok.txt").read_text() == "x"
    target = PROJECT_ROOT / "should_not_exist.txt"
    r = run(ws, f"open({str(target)!r}, 'w').write('x')")
    assert r.returncode != 0 and not target.exists()


def test_project_files_unreadable(ws):
    r = run(ws, f"print(open({str(PROJECT_ROOT / 'pyproject.toml')!r}).read())")
    assert r.returncode != 0 and "Operation not permitted" in r.stderr


def test_wall_clock_kill(ws):
    r = run(ws, "import time; time.sleep(60)", Limits(cpu_s=10, wall_s=1.5, mem_mb=400))
    assert r.timed_out and r.returncode != 0 and r.wall_s < 15


def test_memory_kill(ws):
    r = run(ws, "x = b'a' * (800 << 20); import time; time.sleep(5)", Limits(cpu_s=10, wall_s=30, mem_mb=200))
    assert r.mem_exceeded and r.returncode != 0


def test_child_processes_killed_with_group(ws):
    code = ("import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "open('child.pid', 'w').write(str(p.pid)); time.sleep(60)")
    r = run(ws, code, Limits(cpu_s=10, wall_s=3, mem_mb=400))
    assert r.timed_out
    import os, time
    pid = int((ws / "child.pid").read_text())
    time.sleep(0.5)
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive


def test_profile_mentions_workspace(ws):
    p = seatbelt_profile(ws)
    assert "(deny network*)" in p and str(ws.resolve()) in p
