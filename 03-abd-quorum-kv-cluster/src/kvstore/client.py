"""Client library. Talks to any node, which then coordinates the request.

Retry policy: if we could not even connect to a coordinator, the request
definitely had no effect, so it is safe to try the next node. If the request
was sent and then the connection died or timed out, the outcome is unknown and
we surface that to the caller as definite=False instead of retrying (a blind
retry of a put could apply it twice, interleaved with other writers).
"""

from __future__ import annotations

import random
from typing import Any

from .rpc import Connection, RpcError, Unreachable


class KVClient:
    def __init__(self, addrs: list[tuple[str, int]], timeout: float = 3.0, seed: int | None = None):
        self.conns = [Connection(h, p) for h, p in addrs]
        self.timeout = timeout
        self._rng = random.Random(seed)

    async def _call(self, msg: dict[str, Any]) -> dict[str, Any]:
        order = list(range(len(self.conns)))
        self._rng.shuffle(order)
        last = ""
        for i in order:
            try:
                return await self.conns[i].call(msg, timeout=self.timeout)
            except Unreachable as e:
                last = str(e)
                continue  # safe: nothing was sent
            except RpcError as e:
                return {"ok": False, "error": str(e), "definite": False}
        return {"ok": False, "error": f"no coordinator reachable: {last}", "definite": True}

    async def get(self, key: str) -> dict[str, Any]:
        return await self._call({"op": "get", "key": key})

    async def put(self, key: str, value: Any) -> dict[str, Any]:
        return await self._call({"op": "put", "key": key, "value": value})

    async def status(self, idx: int) -> dict[str, Any]:
        return await self.conns[idx].call({"op": "status"}, timeout=self.timeout)

    async def dump(self, idx: int) -> dict[str, Any]:
        return await self.conns[idx].call({"op": "dump"}, timeout=self.timeout)

    async def close(self) -> None:
        for c in self.conns:
            await c.close()
