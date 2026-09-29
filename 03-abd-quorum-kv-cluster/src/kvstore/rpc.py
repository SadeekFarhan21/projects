"""Minimal multiplexed RPC over TCP.

Frame format: 4-byte big-endian length, then a UTF-8 JSON object.
Requests carry an integer "id"; the response echoes it, so many requests can
be in flight on one connection and responses may come back out of order.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import struct
from collections.abc import Awaitable, Callable
from typing import Any

_HDR = struct.Struct(">I")
MAX_FRAME = 16 * 1024 * 1024

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class RpcError(Exception):
    """Transport-level failure: connection refused/reset or timeout."""


class Unreachable(RpcError):
    """We never managed to send the request, so it definitely had no effect."""


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    hdr = await reader.readexactly(_HDR.size)
    (length,) = _HDR.unpack(hdr)
    if length > MAX_FRAME:
        raise ValueError(f"frame too large: {length}")
    return json.loads(await reader.readexactly(length))


def encode_frame(msg: dict[str, Any]) -> bytes:
    body = json.dumps(msg, separators=(",", ":")).encode()
    return _HDR.pack(len(body)) + body


class Connection:
    """Client side of one TCP connection with request multiplexing."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def _ensure(self, timeout: float) -> None:
        if self.connected:
            return
        async with self._connect_lock:
            if self.connected:
                return
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), timeout
                )
            except (OSError, asyncio.TimeoutError) as e:
                raise Unreachable(f"connect {self.host}:{self.port}: {e!r}") from e
            self._reader_task = asyncio.create_task(self._read_loop(self._reader))

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                msg = await read_frame(reader)
                fut = self._pending.pop(msg.get("id", -1), None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except (asyncio.IncompleteReadError, OSError, ValueError):
            pass
        except asyncio.CancelledError:
            pass
        finally:
            self._fail_all(RpcError(f"connection to {self.host}:{self.port} lost"))
            if self._writer is not None:
                self._writer.close()
            self._writer = None

    def _fail_all(self, exc: Exception) -> None:
        pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(exc)

    async def call(self, msg: dict[str, Any], timeout: float = 1.0) -> dict[str, Any]:
        await self._ensure(timeout)
        rid = next(self._ids)
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        assert self._writer is not None
        try:
            self._writer.write(encode_frame({**msg, "id": rid}))
        except OSError as e:  # pragma: no cover - rare
            self._pending.pop(rid, None)
            raise RpcError(repr(e)) from e
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as e:
            self._pending.pop(rid, None)
            raise RpcError(f"timeout after {timeout}s calling {self.host}:{self.port}") from e

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except BaseException:
                pass
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class Server:
    """Accepts connections and runs `handler` for each request concurrently."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self._server: asyncio.base_events.Server | None = None
        self._conns: set[asyncio.StreamWriter] = set()

    async def start(self, host: str, port: int) -> None:
        self._server = await asyncio.start_server(self._on_conn, host, port)

    async def _on_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._conns.add(writer)
        tasks: set[asyncio.Task[None]] = set()
        try:
            while True:
                msg = await read_frame(reader)
                t = asyncio.create_task(self._serve_one(msg, writer))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
        except (asyncio.IncompleteReadError, OSError, ValueError):
            pass
        finally:
            self._conns.discard(writer)
            writer.close()

    async def _serve_one(self, msg: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        try:
            resp = await self.handler(msg)
        except Exception as e:  # report handler bugs to the caller instead of hanging it
            resp = {"ok": False, "error": f"internal: {e!r}"}
        resp["id"] = msg.get("id")
        if not writer.is_closing():
            writer.write(encode_frame(resp))

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for w in list(self._conns):
            w.close()
