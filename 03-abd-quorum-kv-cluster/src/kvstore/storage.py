"""Per-node storage: an in-memory map backed by an append-only write-ahead log.

Every record is versioned. A version is a tuple (counter, writer_node, seq):
compared lexicographically, so the counter orders writes and the
(writer_node, seq) suffix makes every version globally unique, which means
two different writes can never tie.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

Version = tuple[int, str, int]
ZERO: Version = (0, "", 0)


def as_version(v: Any) -> Version:
    """JSON turns tuples into lists; normalize back."""
    if v is None:
        return ZERO
    return (int(v[0]), str(v[1]), int(v[2]))


class Store:
    def __init__(self, data_dir: str | None = None, fsync: bool = False) -> None:
        self.data: dict[str, tuple[Version, Any]] = {}
        self.fsync = fsync
        self._wal = None
        if data_dir is not None:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            path = Path(data_dir) / "wal.jsonl"
            if path.exists():
                self._replay(path)
            self._wal = open(path, "a", buffering=1 << 16)

    def _replay(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn tail write from a crash; everything before it is intact
                self._apply_mem(rec["k"], as_version(rec["ver"]), rec["v"])

    def _apply_mem(self, key: str, ver: Version, value: Any) -> bool:
        cur = self.data.get(key)
        if cur is None or ver > cur[0]:
            self.data[key] = (ver, value)
            return True
        return False

    def get(self, key: str) -> tuple[Version, Any]:
        return self.data.get(key, (ZERO, None))

    def apply(self, key: str, ver: Version, value: Any) -> bool:
        """Install (ver, value) if it is newer than what we have. Durable before
        returning: the record is written to the OS (and fsynced if configured),
        so a process crash after an ack cannot lose it."""
        changed = self._apply_mem(key, ver, value)
        if changed and self._wal is not None:
            self._wal.write(json.dumps({"k": key, "ver": list(ver), "v": value}) + "\n")
            self._wal.flush()
            if self.fsync:
                os.fsync(self._wal.fileno())
        return changed

    def close(self) -> None:
        if self._wal is not None:
            self._wal.close()
            self._wal = None
