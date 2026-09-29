"""A storage node. Every node is both a replica and a coordinator.

Client ops ("get"/"put") can go to any node; that node coordinates quorum
reads and writes against the key's preference list on the hash ring.
Replica ops ("rget"/"rput") touch only the local store.

Consistency protocol (multi-writer ABD over a Dynamo-style ring):
  put: phase 1 reads versions from R replicas and picks counter = max + 1;
       phase 2 writes (version, value) to the replicas and waits for W acks.
  get: reads from R replicas and takes the highest version. If fewer than W
       replicas are known to hold it, the coordinator writes it back until W
       do, and only then returns. That write-back is what stops a later read
       from seeing an older value ("new-old inversion").
With R + W > N every read quorum intersects every write quorum, which gives a
linearizable register per key under crash-stop failures.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import random
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from .ring import HashRing
from .rpc import Connection, RpcError, Server
from .storage import ZERO, Store, Version, as_version

log = logging.getLogger("kvstore.node")


class QuorumFailed(Exception):
    pass


@dataclass
class NodeConfig:
    node_id: str
    host: str
    port: int
    peers: dict[str, tuple[str, int]]  # every node in the cluster, including self
    n: int = 3
    r: int = 2
    w: int = 2
    vnodes: int = 64
    data_dir: str | None = None
    fsync: bool = False
    rpc_timeout: float = 1.0
    heartbeat_interval: float = 0.1
    suspect_after: float = 0.5  # no successful heartbeat for this long => suspected down
    hint_interval: float = 0.2
    # Experiment knobs. Defaults are the correct protocol.
    writeback: bool = True  # False = skip read write-back (Dynamo-style "just return max")
    jitter_ms: float = 0.0  # uniform random delay before serving each replica op
    use_straggler_reads: bool = True  # count late rget replies before writing back

    def validate(self) -> None:
        if not (1 <= self.r <= self.n and 1 <= self.w <= self.n):
            raise ValueError("need 1 <= R, W <= N")
        if self.n > len(self.peers):
            raise ValueError(f"N={self.n} exceeds cluster size {len(self.peers)}")


@dataclass
class Stats:
    gets: int = 0
    puts: int = 0
    read_writebacks: int = 0
    hints_stored: int = 0
    hints_delivered: int = 0
    failed_ops: int = 0
    replica_rpcs_sent: int = 0  # coordinator -> remote replica requests (not heartbeats/hints)
    extra: dict[str, int] = field(default_factory=dict)


class Node:
    def __init__(self, cfg: NodeConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.id = cfg.node_id
        self.ring = HashRing(cfg.peers.keys(), vnodes=cfg.vnodes)
        self.store = Store(cfg.data_dir, fsync=cfg.fsync)
        self.conns = {p: Connection(h, port) for p, (h, port) in cfg.peers.items() if p != self.id}
        now = time.monotonic()
        # Optimistic start: assume peers are alive until heartbeats say otherwise.
        self.last_seen: dict[str, float] = {p: now for p in self.conns}
        # hints[target][key] = (version, value): writes `target` missed while down.
        self.hints: dict[str, dict[str, tuple[Version, Any]]] = {p: {} for p in self.conns}
        # Version suffix. Seeded from the wall clock so that a restarted node never
        # reuses a (node, seq) pair it handed out before the crash.
        self._seq = itertools.count(time.time_ns() // 1000)
        self.stats = Stats()
        self.server = Server(self.handle)
        self._bg: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.server.start(self.cfg.host, self.cfg.port)
        self._bg = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._hint_loop()),
        ]

    async def stop(self) -> None:
        for t in self._bg:
            t.cancel()
        await self.server.close()
        for c in self.conns.values():
            await c.close()
        self.store.close()

    # ------------------------------------------------------------ failure detector
    def alive(self, peer: str) -> bool:
        if peer == self.id:
            return True
        return time.monotonic() - self.last_seen[peer] < self.cfg.suspect_after

    async def _ping(self, peer: str) -> None:
        try:
            await self.conns[peer].call({"op": "ping"}, timeout=self.cfg.suspect_after)
            self.last_seen[peer] = time.monotonic()
        except RpcError:
            pass

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.gather(*(self._ping(p) for p in self.conns))
            await asyncio.sleep(self.cfg.heartbeat_interval)

    # -------------------------------------------------------------- hinted handoff
    def _add_hint(self, peer: str, key: str, ver: Version, value: Any) -> None:
        cur = self.hints[peer].get(key)
        if cur is None or ver > cur[0]:
            self.hints[peer][key] = (ver, value)
            self.stats.hints_stored += 1

    async def _deliver_hints(self, peer: str) -> None:
        for key, (ver, value) in list(self.hints[peer].items()):
            try:
                await self.conns[peer].call(
                    {"op": "rput", "key": key, "ver": list(ver), "value": value},
                    timeout=self.cfg.rpc_timeout,
                )
            except RpcError:
                return  # peer went away again; retry next round
            # Only drop the hint if no newer one replaced it while we were sending.
            if self.hints[peer].get(key, (None,))[0] == ver:
                del self.hints[peer][key]
                self.stats.hints_delivered += 1

    async def _hint_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.hint_interval)
            peers = [p for p, h in self.hints.items() if h and self.alive(p)]
            if peers:
                await asyncio.gather(*(self._deliver_hints(p) for p in peers))

    # ---------------------------------------------------------------- RPC plumbing
    async def _rpc(self, peer: str, msg: dict[str, Any]) -> dict[str, Any]:
        if peer == self.id:  # short-circuit: no TCP round trip to ourselves
            return await self._replica_op(msg)
        self.stats.replica_rpcs_sent += 1
        resp = await self.conns[peer].call(msg, timeout=self.cfg.rpc_timeout)
        self.last_seen[peer] = time.monotonic()  # any reply is proof of life
        return resp

    async def _quorum(
        self, targets: list[str], msg: dict[str, Any], need: int, hint: bool = False,
        stragglers: set[asyncio.Task[Any]] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Send msg to targets and return as soon as `need` of them succeed.

        Nodes the failure detector suspects are skipped (a write to them becomes
        a hint straight away) unless skipping them would make the quorum
        impossible, in which case we try everyone: the detector can be wrong.
        Requests still in flight when we return keep running in the background,
        so slow replicas still converge; failed writes turn into hints. If the
        caller passes a `stragglers` set, those in-flight tasks are added to it.
        """
        live = [p for p in targets if self.alive(p)]
        if len(live) < need:
            live = list(targets)
        if hint:
            for p in targets:
                if p not in live:
                    self._add_hint(p, msg["key"], as_version(msg["ver"]), msg["value"])

        async def one(p: str) -> tuple[str, dict[str, Any]]:
            try:
                resp = await self._rpc(p, msg)
            except RpcError:
                if hint:
                    self._add_hint(p, msg["key"], as_version(msg["ver"]), msg["value"])
                raise
            return p, resp

        pending = {asyncio.create_task(one(p)) for p in live}
        ok: list[tuple[str, dict[str, Any]]] = []
        failed = 0
        try:
            while pending and len(ok) < need:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None and t.result()[1].get("ok"):
                        ok.append(t.result())
                    else:
                        failed += 1
                if len(live) - failed < need:
                    break
        finally:
            # Stragglers keep running in the background (even if our caller was
            # cancelled); mark their exceptions as retrieved so asyncio does not
            # log "Task exception was never retrieved".
            for t in pending:
                t.add_done_callback(lambda t: t.cancelled() or t.exception())
            if stragglers is not None:
                stragglers.update(pending)
        if len(ok) < need:
            raise QuorumFailed(f"{len(ok)}/{need} acks for {msg['op']}")
        return ok

    # ------------------------------------------------------------------- handlers
    async def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        op = msg.get("op")
        if op == "get":
            return await self.coord_get(msg["key"])
        if op == "put":
            return await self.coord_put(msg["key"], msg["value"])
        if op == "status":
            return {
                "ok": True,
                "node": self.id,
                "alive": {p: self.alive(p) for p in self.conns},
                "hints": sum(len(h) for h in self.hints.values()),
                "keys": len(self.store.data),
                "stats": self.stats.__dict__,
                "cpu_s": time.process_time(),  # lets benchmarks compute CPU cost per op
            }
        if op == "dump":
            return {"ok": True, "data": {k: [list(v), x] for k, (v, x) in self.store.data.items()}}
        return await self._replica_op(msg)

    async def _replica_op(self, msg: dict[str, Any]) -> dict[str, Any]:
        op = msg.get("op")
        if self.cfg.jitter_ms and op in ("rget", "rput"):
            # Simulated network/queueing delay, to widen race windows in experiments.
            await asyncio.sleep(random.random() * self.cfg.jitter_ms / 1000)
        if op == "ping":
            return {"ok": True, "node": self.id}
        if op == "rget":
            ver, value = self.store.get(msg["key"])
            return {"ok": True, "ver": list(ver), "value": value}
        if op == "rput":
            self.store.apply(msg["key"], as_version(msg["ver"]), msg["value"])
            return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}

    # --------------------------------------------------------------- coordinator
    async def coord_put(self, key: str, value: Any) -> dict[str, Any]:
        self.stats.puts += 1
        pref = self.ring.preference_list(key, self.cfg.n)
        try:
            reads = await self._quorum(pref, {"op": "rget", "key": key}, self.cfg.r)
        except QuorumFailed as e:
            self.stats.failed_ops += 1
            # Nothing was written yet, so the client knows this put had no effect.
            return {"ok": False, "error": f"unavailable: {e}", "definite": True}
        counter = max(as_version(r["ver"])[0] for _, r in reads) + 1
        ver: Version = (counter, self.id, next(self._seq))
        msg = {"op": "rput", "key": key, "ver": list(ver), "value": value}
        try:
            await self._quorum(pref, msg, self.cfg.w, hint=True)
        except QuorumFailed as e:
            self.stats.failed_ops += 1
            # Some replicas may have applied it: outcome is unknown to the client.
            return {"ok": False, "error": f"write quorum not reached: {e}", "definite": False}
        return {"ok": True, "ver": list(ver)}

    async def coord_get(self, key: str) -> dict[str, Any]:
        self.stats.gets += 1
        pref = self.ring.preference_list(key, self.cfg.n)
        try:
            stragglers: set[asyncio.Task[Any]] = set()
            reads = await self._quorum(pref, {"op": "rget", "key": key}, self.cfg.r,
                                       stragglers=stragglers)
        except QuorumFailed as e:
            self.stats.failed_ops += 1
            return {"ok": False, "error": f"unavailable: {e}", "definite": True}
        best = max(as_version(r["ver"]) for _, r in reads)
        value = next(r["value"] for _, r in reads if as_version(r["ver"]) == best)
        holders = {p for p, r in reads if as_version(r["ver"]) == best}
        if (self.cfg.writeback and self.cfg.use_straggler_reads and best != ZERO
                and len(holders) < self.cfg.w):
            # We sent rget to every replica but only waited for R. Before paying
            # for a write-back, count the replies already in flight: a replica
            # holding `best` or anything newer counts toward W. Without this,
            # every read with R < W writes back even when the value is fully
            # replicated.
            for fut in asyncio.as_completed(stragglers):
                try:
                    p, r = await fut
                except RpcError:
                    continue
                if r.get("ok") and as_version(r["ver"]) >= best:
                    holders.add(p)
                    if len(holders) >= self.cfg.w:
                        break
        if self.cfg.writeback and best != ZERO and len(holders) < self.cfg.w:
            # Write-back: make sure a full write quorum has `best` before we
            # expose it, or a later read could miss it.
            self.stats.read_writebacks += 1
            others = [p for p in pref if p not in holders]
            msg = {"op": "rput", "key": key, "ver": list(best), "value": value}
            try:
                await self._quorum(others, msg, self.cfg.w - len(holders), hint=True)
            except QuorumFailed as e:
                self.stats.failed_ops += 1
                return {"ok": False, "error": f"write-back failed: {e}", "definite": True}
        return {"ok": True, "value": value, "ver": list(best)}


# ---------------------------------------------------------------------- CLI entry
def parse_peers(spec: str) -> dict[str, tuple[str, int]]:
    """'n1=127.0.0.1:7001,n2=127.0.0.1:7002' -> {'n1': ('127.0.0.1', 7001), ...}"""
    out: dict[str, tuple[str, int]] = {}
    for item in spec.split(","):
        name, addr = item.split("=")
        host, port = addr.rsplit(":", 1)
        out[name.strip()] = (host, int(port))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run one kvstore node")
    ap.add_argument("--id", required=True)
    ap.add_argument("--peers", required=True, help="n1=host:port,n2=host:port,...")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--vnodes", type=int, default=64)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--fsync", action="store_true")
    ap.add_argument("--rpc-timeout", type=float, default=1.0)
    ap.add_argument("--heartbeat-interval", type=float, default=0.1)
    ap.add_argument("--suspect-after", type=float, default=0.5)
    ap.add_argument("--no-writeback", action="store_true", help="unsafe: skip read write-back")
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--eager-writeback", action="store_true",
                    help="write back as soon as the first R replies lack W holders")
    ap.add_argument("--log-level", default="WARNING")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format=f"%(asctime)s {a.id} %(levelname)s %(message)s")
    peers = parse_peers(a.peers)
    host, port = peers[a.id]
    cfg = NodeConfig(
        node_id=a.id, host=host, port=port, peers=peers, n=a.n, r=a.r, w=a.w,
        vnodes=a.vnodes, data_dir=a.data_dir, fsync=a.fsync, rpc_timeout=a.rpc_timeout,
        heartbeat_interval=a.heartbeat_interval, suspect_after=a.suspect_after,
        writeback=not a.no_writeback, jitter_ms=a.jitter_ms,
        use_straggler_reads=not a.eager_writeback,
    )

    async def run() -> None:
        node = Node(cfg)
        await node.start()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, stop.set)
        print(f"READY {a.id} {host}:{port}", flush=True)
        await stop.wait()
        await node.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
