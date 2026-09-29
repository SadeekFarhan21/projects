"""Jepsen-style workload: concurrent clients + a nemesis, recording a history.

Each client process loops: pick a key from a small set, then read it or write
a globally unique value. Every op is recorded with its invocation and
completion time from one monotonic clock, and its outcome (ok/fail/info).
The resulting history goes to checker.check_history.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from .checker import Op
from .client import KVClient
from .cluster import LocalCluster


@dataclass
class WorkloadConfig:
    clients: int = 5
    keys: int = 3
    duration: float = 5.0
    read_fraction: float = 0.5
    seed: int = 0
    think_time: float = 0.002  # small gap between ops keeps per-key histories checkable


async def _client_loop(
    pid: int, cluster: LocalCluster, cfg: WorkloadConfig, stop_at: float, history: list[Op]
) -> None:
    rng = random.Random(cfg.seed * 1000 + pid)
    client = KVClient(cluster.addrs, seed=cfg.seed * 1000 + pid)
    counter = 0
    try:
        while time.monotonic() < stop_at:
            key = f"k{rng.randrange(cfg.keys)}"
            if rng.random() < cfg.read_fraction:
                t0 = time.monotonic()
                r = await client.get(key)
                t1 = time.monotonic()
                status = "ok" if r["ok"] else ("fail" if r.get("definite") else "info")
                history.append(Op(pid, "read", key, r.get("value"), t0, t1, status))
            else:
                counter += 1
                val = pid * 1_000_000 + counter
                t0 = time.monotonic()
                r = await client.put(key, val)
                t1 = time.monotonic()
                status = "ok" if r["ok"] else ("fail" if r.get("definite") else "info")
                history.append(Op(pid, "write", key, val, t0, t1 if status != "info" else math.inf, status))
            await asyncio.sleep(rng.random() * cfg.think_time)
    finally:
        await client.close()


Nemesis = Callable[[LocalCluster, float], Awaitable[list[str]]]


async def kill_one_nemesis(cluster: LocalCluster, duration: float, victim: str = "n1") -> list[str]:
    """Kill one node a third of the way in, restart it at two thirds."""
    events = []
    await asyncio.sleep(duration / 3)
    await asyncio.to_thread(cluster.kill, victim)
    events.append(f"{time.monotonic():.3f} kill {victim}")
    await asyncio.sleep(duration / 3)
    await asyncio.to_thread(cluster.restart, victim)
    events.append(f"{time.monotonic():.3f} restart {victim}")
    return events


async def pause_one_nemesis(cluster: LocalCluster, duration: float, victim: str = "n2") -> list[str]:
    """SIGSTOP one node (it holds connections open but never answers), then SIGCONT."""
    events = []
    await asyncio.sleep(duration / 3)
    cluster.pause(victim)
    events.append(f"{time.monotonic():.3f} pause {victim}")
    await asyncio.sleep(duration / 3)
    cluster.resume(victim)
    events.append(f"{time.monotonic():.3f} resume {victim}")
    return events


async def run_workload(
    cluster: LocalCluster, cfg: WorkloadConfig, nemesis: Nemesis | None = None
) -> tuple[list[Op], list[str]]:
    history: list[Op] = []
    stop_at = time.monotonic() + cfg.duration
    tasks = [asyncio.create_task(_client_loop(p, cluster, cfg, stop_at, history)) for p in range(cfg.clients)]
    events: list[str] = []
    if nemesis is not None:
        events = await nemesis(cluster, cfg.duration)
    await asyncio.gather(*tasks)
    return history, events


def save_history(path: str, history: list[Op]) -> None:
    with open(path, "w") as f:
        for o in history:
            d = asdict(o)
            if d["ret"] == math.inf:
                d["ret"] = None
            f.write(json.dumps(d) + "\n")
