"""Command line entry point.

  kv-cli cluster --size 5 [--r 2 --w 2]    run a local cluster until Ctrl-C
  kv-cli put KEY VALUE --nodes 127.0.0.1:7001,127.0.0.1:7002
  kv-cli get KEY --nodes ...
  kv-cli status --nodes ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time

from .client import KVClient
from .cluster import LocalCluster


def _addrs(spec: str) -> list[tuple[str, int]]:
    out = []
    for item in spec.split(","):
        host, port = item.rsplit(":", 1)
        out.append((host, int(port)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kv-cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster", help="run a local cluster in the foreground")
    c.add_argument("--size", type=int, default=5)
    c.add_argument("--r", type=int, default=2)
    c.add_argument("--w", type=int, default=2)
    for name in ("put", "get", "status"):
        p = sub.add_parser(name)
        p.add_argument("--nodes", required=True, help="host:port,host:port,...")
        if name in ("put", "get"):
            p.add_argument("key")
        if name == "put":
            p.add_argument("value")
    a = ap.parse_args(argv)

    if a.cmd == "cluster":
        # Background shells ignore SIGINT, so also treat SIGTERM as "shut down cleanly".
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        cl = LocalCluster(size=a.size, n=3, r=a.r, w=a.w).start()
        nodes = ",".join(f"{h}:{p}" for h, p in cl.addrs)
        print(f"cluster up: {a.size} nodes, N=3 R={a.r} W={a.w}", flush=True)
        print(f"  --nodes {nodes}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            # `uv run` forwards the signal it received, so a second SIGTERM can
            # arrive while we are stopping. Ignore it, or the KeyboardInterrupt
            # it raises would abort stop() and orphan the node processes.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            cl.stop()
        return

    async def run() -> None:
        client = KVClient(_addrs(a.nodes))
        try:
            if a.cmd == "put":
                try:
                    value = json.loads(a.value)
                except json.JSONDecodeError:
                    value = a.value
                print(json.dumps(await client.put(a.key, value)))
            elif a.cmd == "get":
                print(json.dumps(await client.get(a.key)))
            else:
                for i in range(len(client.conns)):
                    try:
                        print(json.dumps(await client.status(i)))
                    except Exception as e:  # a dead node should not hide the others
                        print(json.dumps({"ok": False, "error": repr(e)}))
        finally:
            await client.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
