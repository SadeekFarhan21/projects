"""Consistent hash ring with virtual nodes.

Each physical node owns `vnodes` points (tokens) on a 64-bit ring. A key is
hashed onto the ring and its preference list is the first N *distinct*
physical nodes found walking clockwise from that position.
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Iterable


def hash64(data: str) -> int:
    """Stable 64-bit hash. Python's hash() is salted per process, so it is useless
    for placement that several processes must agree on."""
    return int.from_bytes(hashlib.blake2b(data.encode(), digest_size=8).digest(), "big")


class HashRing:
    def __init__(self, nodes: Iterable[str] = (), vnodes: int = 64) -> None:
        self.vnodes = vnodes
        self._tokens: list[int] = []  # sorted token positions
        self._owner: dict[int, str] = {}  # token -> physical node id
        self._nodes: set[str] = set()
        for n in nodes:
            self.add_node(n)

    @property
    def nodes(self) -> list[str]:
        return sorted(self._nodes)

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self.vnodes):
            t = hash64(f"{node}#{i}")
            # A 64-bit collision is astronomically unlikely; skip rather than overwrite.
            if t in self._owner:
                continue
            self._owner[t] = node
            bisect.insort(self._tokens, t)

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        dead = {t for t, n in self._owner.items() if n == node}
        self._tokens = [t for t in self._tokens if t not in dead]
        for t in dead:
            del self._owner[t]

    def preference_list(self, key: str, n: int) -> list[str]:
        """First n distinct physical nodes clockwise from hash(key)."""
        if not self._tokens:
            return []
        n = min(n, len(self._nodes))
        start = bisect.bisect_right(self._tokens, hash64(key))
        out: list[str] = []
        for i in range(len(self._tokens)):
            owner = self._owner[self._tokens[(start + i) % len(self._tokens)]]
            if owner not in out:
                out.append(owner)
                if len(out) == n:
                    break
        return out

    def extended_list(self, key: str) -> list[str]:
        """All physical nodes in ring order from the key. The first N are the
        preference list; the rest are fallback candidates."""
        return self.preference_list(key, len(self._nodes))
