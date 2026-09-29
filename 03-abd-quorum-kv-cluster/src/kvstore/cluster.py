"""Start, kill and restart a cluster of node processes on localhost.

Used by the tests, the benchmark and the `kv-cluster` style demo. Each node is
a real OS process (`python -m kvstore.node`), so a kill is a real SIGKILL.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field


def free_ports(k: int) -> list[int]:
    socks, ports = [], []
    for _ in range(k):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


@dataclass
class LocalCluster:
    size: int = 5
    n: int = 3
    r: int = 2
    w: int = 2
    vnodes: int = 64
    data_root: str | None = None
    heartbeat_interval: float = 0.1
    suspect_after: float = 0.5
    rpc_timeout: float = 1.0
    writeback: bool = True
    jitter_ms: float = 0.0
    eager_writeback: bool = False
    procs: dict[str, subprocess.Popen] = field(default_factory=dict)
    ports: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ids = [f"n{i + 1}" for i in range(self.size)]
        self.ports = dict(zip(self.ids, free_ports(self.size)))
        if self.data_root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="kvstore-")
            self.data_root = self._tmp.name
        self.peer_spec = ",".join(f"{i}=127.0.0.1:{p}" for i, p in self.ports.items())

    @property
    def addrs(self) -> list[tuple[str, int]]:
        return [("127.0.0.1", self.ports[i]) for i in self.ids]

    def _spawn(self, node_id: str) -> subprocess.Popen:
        cmd = [
            sys.executable, "-m", "kvstore.node", "--id", node_id, "--peers", self.peer_spec,
            "--n", str(self.n), "--r", str(self.r), "--w", str(self.w),
            "--vnodes", str(self.vnodes),
            "--data-dir", os.path.join(self.data_root, node_id),
            "--heartbeat-interval", str(self.heartbeat_interval),
            "--suspect-after", str(self.suspect_after),
            "--rpc-timeout", str(self.rpc_timeout),
            "--jitter-ms", str(self.jitter_ms),
        ]
        if not self.writeback:
            cmd.append("--no-writeback")
        if self.eager_writeback:
            cmd.append("--eager-writeback")
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None)

    def _wait_ready(self, node_id: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.ports[node_id]), timeout=0.2):
                    return
            except OSError:
                if self.procs[node_id].poll() is not None:
                    raise RuntimeError(f"{node_id} exited with {self.procs[node_id].returncode}")
                time.sleep(0.02)
        raise TimeoutError(f"{node_id} did not start")

    def start(self) -> "LocalCluster":
        for i in self.ids:
            self.procs[i] = self._spawn(i)
        for i in self.ids:
            self._wait_ready(i)
        return self

    def kill(self, node_id: str) -> None:
        """SIGKILL: no shutdown hooks, the in-memory state is gone; only the WAL survives."""
        p = self.procs[node_id]
        p.send_signal(signal.SIGKILL)
        p.wait()

    def pause(self, node_id: str) -> None:
        """SIGSTOP: the process is alive but unresponsive, like a GC pause or a hung box."""
        self.procs[node_id].send_signal(signal.SIGSTOP)

    def resume(self, node_id: str) -> None:
        self.procs[node_id].send_signal(signal.SIGCONT)

    def restart(self, node_id: str) -> None:
        if self.procs[node_id].poll() is None:
            self.kill(node_id)
        self.procs[node_id] = self._spawn(node_id)
        self._wait_ready(node_id)

    def stop(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def __enter__(self) -> "LocalCluster":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
