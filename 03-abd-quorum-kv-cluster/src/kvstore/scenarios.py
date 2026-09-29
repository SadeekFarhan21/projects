"""Deterministic failure scenarios, shared by the tests and scripts/inversion_demo.py."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .checker import Op, check_register
from .cluster import free_ports
from .node import Node, NodeConfig


def slow_rput(node: Node, delay: float) -> None:
    """Make `node` apply replica writes `delay` seconds late (a slow disk or a
    congested link) while still answering reads promptly."""
    orig = node._replica_op

    async def patched(msg: dict[str, Any]) -> dict[str, Any]:
        if msg.get("op") == "rput":
            await asyncio.sleep(delay)
        return await orig(msg)

    node._replica_op = patched  # type: ignore[method-assign]


async def run_inversion_scenario(data_dir: str, writeback: bool) -> tuple[Any, Any, bool]:
    """The scenario that motivates read write-back.

    3 nodes, N=3, R=2, W=2. A write of "new" reaches n1 only (n2 and n3 are
    slow to apply writes). A read coordinated by n1 sees "new" and returns.
    Then n1 crashes, and a read coordinated by n2 asks n2+n3, neither of which
    has applied "new" yet. Without write-back the second read returns "old"
    after the first returned "new": a new-old inversion, which the checker
    must reject. Returns (read1, read2, history_is_linearizable).
    """
    ids = ["n1", "n2", "n3"]
    peers = {i: ("127.0.0.1", p) for i, p in zip(ids, free_ports(3))}
    nodes = [
        Node(NodeConfig(node_id=i, host="127.0.0.1", port=peers[i][1], peers=peers, n=3, r=2, w=2,
                        data_dir=os.path.join(data_dir, i), heartbeat_interval=0.05,
                        suspect_after=0.25, rpc_timeout=2.5, writeback=writeback))
        for i in ids
    ]
    for nd in nodes:
        await nd.start()
    clock = asyncio.get_running_loop().time
    h: list[Op] = []
    try:
        t = clock()
        assert (await nodes[0].coord_put("k", "old"))["ok"]
        h.append(Op(0, "write", "k", "old", t, clock()))
        # Long enough that scheduler hiccups on a loaded machine cannot close the
        # race window, short enough (< rpc_timeout) that a write-back succeeds.
        slow_rput(nodes[1], 1.5)
        slow_rput(nodes[2], 1.5)
        t_put = clock()
        put = asyncio.create_task(nodes[0].coord_put("k", "new"))
        await asyncio.sleep(0.05)
        t = clock()
        r1 = await nodes[0].coord_get("k")
        h.append(Op(1, "read", "k", r1.get("value"), t, clock()))
        await nodes[0].stop()  # crash n1
        put.cancel()
        h.append(Op(0, "write", "k", "new", t_put, float("inf"), "info"))  # never acked
        t = clock()
        r2 = await nodes[1].coord_get("k")
        h.append(Op(2, "read", "k", r2.get("value"), t, clock()))
        return r1.get("value"), r2.get("value"), check_register(h).ok
    finally:
        for nd in nodes[1:]:
            await nd.stop()
