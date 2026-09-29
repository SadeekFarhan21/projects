#!/usr/bin/env python3
"""Run the benchmark suite against kvd and save raw results under results/.

Experiments (each configuration is repeated --reps times; every run is kept):
  pipeline   50 connections, SET, pipeline depth 1..128, AOF off
  clients    pipeline 1, SET, 1..256 connections, AOF off
  workload   50 connections, pipeline 16: SET vs GET vs 50/50 mix, AOF off
  aof        50 connections, SET, AOF off / no / everysec / always at P=1 and P=16
  fsync1     1 connection, P=1, SET, same four AOF modes (fsync cost without group commit)
  expiry     200k keys with a 1 s TTL + 200k without; DBSIZE sampled every 50 ms
             with no client touching the keys (active expiry only)
  replay     startup time to replay AOF files of increasing size

Each kvbench run is a JSON line, extended with the 1-minute load average
before and after the run, because this machine is shared and load matters.

Usage: scripts/run_benchmarks.py [--build build/release] [--only pipeline,clients] [--reps 3]
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def wait_port(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError("server did not come up on %d" % port)


class Kvd:
    def __init__(self, build, port, aof=None, fsync="everysec"):
        cmd = [os.path.join(build, "kvd"), "--port", str(port)]
        if aof:
            cmd += ["--aof", aof, "--appendfsync", fsync]
        else:
            cmd += ["--appendonly", "no"]
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.stderr = []
        # Read stderr until the ready line so we can time startup (AOF replay).
        while True:
            line = self.proc.stderr.readline()
            if not line:
                raise RuntimeError("kvd exited: %s" % "".join(self.stderr))
            self.stderr.append(line)
            if "kvd ready" in line:
                break
        self.ready_s = time.time() - self.t0
        wait_port(port)
        global SERVER_PID
        SERVER_PID = self.proc.pid

    def stop(self):
        global SERVER_PID
        SERVER_PID = None
        self.proc.send_signal(signal.SIGTERM)
        self.stderr.append(self.proc.stderr.read())
        self.proc.wait(timeout=30)


def cpu_seconds(pid):
    """Total CPU time (user+sys) a process has used, from ps (10 ms resolution)."""
    out = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    parts = out.replace("-", ":").split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


THREADS = None  # --threads: cap kvbench client threads (keeps the load generator light)
SERVER_PID = None  # set by Kvd so kvbench() can attribute server CPU time to each run


def kvbench(build, port, **kw):
    args = [os.path.join(build, "kvbench"), "-p", str(port), "--json"]
    flags = {"clients": "-c", "pipeline": "-P", "requests": "-n", "workload": "-w",
             "value_size": "-d", "keyspace": "-r", "threads": "-t", "label": "--label"}
    if THREADS and "threads" not in kw:
        kw["threads"] = THREADS
    for k, v in kw.items():
        args += [flags[k], str(v)]
    load_before = os.getloadavg()[0]
    cpu0 = cpu_seconds(SERVER_PID) if SERVER_PID else None
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    row = json.loads(out.strip().splitlines()[-1])
    if cpu0 is not None:
        # Server CPU seconds spent during this run, and throughput per server
        # CPU second: a load-robust efficiency number (see DEVLOG).
        cpu = cpu_seconds(SERVER_PID) - cpu0
        row["server_cpu_s"] = round(cpu, 3)
        row["ops_per_server_cpu_s"] = round(row["requests"] / cpu) if cpu > 0 else None
    row["load1_before"] = round(load_before, 1)
    row["load1_after"] = round(os.getloadavg()[0], 1)
    return row


def emit(fh, row):
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    print("  %-18s c=%-3d P=%-3d %10.0f ops/s  p50 %8.1f  p99 %9.1f us  %9s ops/cpu-s  load %.0f" % (
        row.get("label", ""), row["clients"], row["pipeline"], row["ops_per_sec"],
        row["p50_us"], row["p99_us"], row.get("ops_per_server_cpu_s"), row["load1_before"]))


def requests_for(clients, pipeline):
    # Enough work for a stable number without taking forever on a loaded box.
    return max(50_000, min(1_000_000, 20_000 * clients * pipeline // 50 * 5))


def exp_pipeline(a, fh):
    with_srv = Kvd(a.build, a.port)
    try:
        kvbench(a.build, a.port, clients=50, pipeline=16, requests=200_000, workload="set", label="warmup")
        for rep in range(a.reps):
            for p in [1, 2, 4, 8, 16, 32, 64, 128]:
                row = kvbench(a.build, a.port, clients=50, pipeline=p, requests=requests_for(50, p),
                              workload="set", label="pipeline")
                row["rep"] = rep
                emit(fh, row)
    finally:
        with_srv.stop()


def exp_clients(a, fh):
    srv = Kvd(a.build, a.port)
    try:
        for rep in range(a.reps):
            for c in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                row = kvbench(a.build, a.port, clients=c, pipeline=1, requests=requests_for(c, 1),
                              workload="set", label="clients")
                row["rep"] = rep
                emit(fh, row)
    finally:
        srv.stop()


def exp_workload(a, fh):
    srv = Kvd(a.build, a.port)
    try:
        # Populate the keyspace first so GETs hit.
        kvbench(a.build, a.port, clients=50, pipeline=64, requests=1_000_000, workload="set",
                keyspace=100_000, label="populate")
        for rep in range(a.reps):
            for w in ["set", "get", "mix"]:
                row = kvbench(a.build, a.port, clients=50, pipeline=16, requests=1_000_000, workload=w,
                              keyspace=100_000, label="workload-" + w)
                row["rep"] = rep
                emit(fh, row)
    finally:
        srv.stop()


def exp_aof(a, fh):
    for rep in range(a.reps):
        for mode in ["off", "no", "everysec", "always"]:
            with tempfile.TemporaryDirectory() as d:
                aof = os.path.join(d, "bench.aof") if mode != "off" else None
                srv = Kvd(a.build, a.port, aof=aof, fsync=mode if aof else "everysec")
                try:
                    for p in [1, 16]:
                        row = kvbench(a.build, a.port, clients=50, pipeline=p, requests=200_000 if p == 1 else 1_000_000,
                                      workload="set", label="aof-" + mode)
                        row["rep"] = rep
                        row["aof"] = mode
                        emit(fh, row)
                finally:
                    srv.stop()
                if aof:
                    fh.write(json.dumps({"label": "aof-file", "aof": mode, "rep": rep,
                                         "bytes": os.path.getsize(aof)}) + "\n")


def exp_fsync1(a, fh):
    """One connection, pipeline 1: every SET waits for its own loop iteration,
    so with appendfsync always each write pays a full fsync (no group commit)."""
    for rep in range(a.reps):
        for mode in ["off", "no", "everysec", "always"]:
            with tempfile.TemporaryDirectory() as d:
                aof = os.path.join(d, "bench.aof") if mode != "off" else None
                srv = Kvd(a.build, a.port, aof=aof, fsync=mode if aof else "everysec")
                try:
                    row = kvbench(a.build, a.port, clients=1, pipeline=1, requests=100_000,
                                  workload="set", label="fsync1-" + mode)
                    row["rep"] = rep
                    row["aof"] = mode
                    emit(fh, row)
                finally:
                    srv.stop()


def resp(*args):
    out = [b"*%d\r\n" % len(args)]
    for x in args:
        x = x if isinstance(x, bytes) else str(x).encode()
        out.append(b"$%d\r\n%s\r\n" % (len(x), x))
    return b"".join(out)


def read_replies(sock, n):
    """Consume n simple replies (+OK / :int). Counts CRLF-terminated lines."""
    buf = b""
    got = 0
    while got < n:
        chunk = sock.recv(1 << 20)
        buf += chunk
        got += chunk.count(b"\r\n")
    return buf


def dbsize(sock):
    sock.sendall(resp("DBSIZE"))
    line = b""
    while not line.endswith(b"\r\n"):
        line += sock.recv(64)
    return int(line[1:-2])


def exp_expiry(a, fh):
    """200k persistent keys + 200k keys that all expire at the same absolute
    instant (PXAT). Nobody reads them afterwards, so only the active expiry
    cron can reclaim them. DBSIZE counts physically present keys."""
    srv = Kvd(a.build, a.port)
    try:
        s = socket.create_connection(("127.0.0.1", a.port))
        n, batch = 200_000, 10_000
        keep = [b"".join(resp("SET", "keep:%d" % i, "v") for i in range(b, b + batch)) for b in range(0, n, batch)]
        t_keep = time.time()
        for chunk in keep:
            s.sendall(chunk)
            read_replies(s, batch)
        keep_load_s = time.time() - t_keep
        # Encode first, then pick the deadline, so encoding time cannot eat the TTL.
        tmpl = [[("ttl:%d" % i) for i in range(b, b + batch)] for b in range(0, n, batch)]
        # The same amount of loading just took keep_load_s; leave twice that
        # plus 2 s of headroom so every TTL key is in before the deadline.
        deadline_ms = int(time.time() * 1000) + int(2000 * keep_load_s) + 2000
        ttl = [b"".join(resp("SET", k, "v", "PXAT", deadline_ms) for k in keys) for keys in tmpl]
        for chunk in ttl:
            s.sendall(chunk)
            read_replies(s, batch)
        loaded_at = time.time() * 1000
        fh.write(json.dumps({"label": "expiry-meta", "ttl_keys": n, "persistent_keys": n,
                             "deadline_ms": deadline_ms, "loaded_ms_before_deadline": round(deadline_ms - loaded_at),
                             "load1": round(os.getloadavg()[0], 1)}) + "\n")
        while time.time() * 1000 < deadline_ms + 4000:
            t_rel = time.time() * 1000 - deadline_ms
            fh.write(json.dumps({"label": "expiry", "t_ms_after_deadline": round(t_rel), "dbsize": dbsize(s)}) + "\n")
            time.sleep(0.05)
        s.close()
    finally:
        srv.stop()
    fh.write(json.dumps({"label": "expiry-server-log", "log": "".join(srv.stderr).strip()}) + "\n")


def exp_replay(a, fh):
    for n in [100_000, 400_000, 1_600_000]:
        with tempfile.TemporaryDirectory() as d:
            aof = os.path.join(d, "replay.aof")
            srv = Kvd(a.build, a.port, aof=aof, fsync="no")
            kvbench(a.build, a.port, clients=50, pipeline=64, requests=n, workload="set", keyspace=n,
                    label="replay-fill")
            srv.stop()
            size = os.path.getsize(aof)
            for rep in range(a.reps):
                srv = Kvd(a.build, a.port, aof=aof, fsync="no")
                ready = srv.ready_s
                log = "".join(srv.stderr)
                srv.stop()
                m = re.search(r"replayed (\d+) commands.*?(\d+) ms wall, (\d+) ms cpu", log)
                row = {"label": "replay", "commands": int(m.group(1)), "aof_bytes": size,
                       "replay_wall_ms": int(m.group(2)), "replay_cpu_ms": int(m.group(3)),
                       "startup_s": round(ready, 4), "rep": rep, "load1": round(os.getloadavg()[0], 1)}
                fh.write(json.dumps(row) + "\n")
                print("  replay %8d cmds %6.1f MB  wall %6d ms  cpu %6d ms" % (
                    row["commands"], size / 1e6, row["replay_wall_ms"], row["replay_cpu_ms"]))


EXPERIMENTS = {"pipeline": exp_pipeline, "clients": exp_clients, "workload": exp_workload,
               "aof": exp_aof, "fsync1": exp_fsync1, "expiry": exp_expiry, "replay": exp_replay}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "release"))
    ap.add_argument("--port", type=int, default=7379)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default=",".join(EXPERIMENTS))
    ap.add_argument("--threads", type=int, default=None, help="kvbench threads (default: kvbench's min(c, 8))")
    a = ap.parse_args()
    global THREADS
    THREADS = a.threads
    os.makedirs(RESULTS, exist_ok=True)
    env = {"host": os.uname().nodename.split(".")[0], "machine": os.uname().machine,
           "cpus": os.cpu_count(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "load1_at_start": round(os.getloadavg()[0], 1), "kvbench_threads": a.threads}
    for name in a.only.split(","):
        path = os.path.join(RESULTS, "bench_%s.jsonl" % name)
        print("== %s -> %s" % (name, os.path.relpath(path, ROOT)))
        with open(path, "w") as fh:
            fh.write(json.dumps({"label": "env", **env}) + "\n")
            EXPERIMENTS[name](a, fh)


if __name__ == "__main__":
    main()
