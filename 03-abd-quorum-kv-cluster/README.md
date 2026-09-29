# kvstore: a sharded, quorum-replicated key-value store

A small Dynamo-style key-value cluster that runs as several local processes. Keys are partitioned with consistent hashing (64 virtual nodes per server) and each key lives on N=3 replicas. Reads and writes use configurable quorums (R + W > N is enforced for the safe mode, weaker settings are allowed for experiments). A write that misses a replica leaves a hint that is replayed when the replica comes back, and every node runs a heartbeat failure detector.

The consistency protocol is multi-writer ABD on top of the ring: a put reads versions from R replicas and writes `max + 1` to W replicas; a get reads R replicas and, if the newest value is not yet on W replicas, writes it back before returning. That makes every key a linearizable register under crash failures, and the repo contains the tooling to check that claim: a Jepsen-style workload runner with a kill/pause nemesis, and a Knossos-style (Wing and Gong plus Lowe memoization) linearizability checker.

v0 is Python 3.12 asyncio with zero runtime dependencies. It is built to get the protocol right and measurable, not to be fast. A C++ port is a later milestone.

See [DESIGN.md](DESIGN.md) for the architecture, invariants and trade-offs, and [DEVLOG.md](DEVLOG.md) for how it was built, what broke, and the measurements.

### Layout

```
src/kvstore/
  ring.py        consistent hash ring with virtual nodes (blake2b)
  rpc.py         length-prefixed JSON over TCP, multiplexed request ids
  storage.py     versioned in-memory map + append-only WAL, replay on start
  node.py        the node: replica ops, quorum coordinator, heartbeats, hinted handoff
  client.py      client library (retries only when the request provably was not sent)
  cluster.py     start / SIGKILL / SIGSTOP / restart a cluster of real processes
  checker.py     linearizability checker for a register (WGL search with memoization)
  jepsen.py      concurrent workload + nemesis, records a history
  scenarios.py   deterministic failure scenario (new-old inversion)
  cli.py         kv-cli: run a local cluster, put, get, status
tests/           unit, in-process multi-node, and multi-process cluster tests
scripts/         benchmark, linearizability experiment, plots, inversion demo
results/         raw outputs of every experiment quoted in DEVLOG.md
```

### Build

Requires [uv](https://docs.astral.sh/uv/). uv picks up Python 3.12 from `.python-version`.

```
cd projects/03-abd-quorum-kv-cluster
uv sync
```

### Run

```
uv run kv-cli cluster --size 5              # 5 node processes, N=3 R=2 W=2; prints --nodes
uv run kv-cli put greeting '"hello"' --nodes 127.0.0.1:PORT1,127.0.0.1:PORT2
uv run kv-cli get greeting --nodes 127.0.0.1:PORT1,127.0.0.1:PORT2
uv run kv-cli status --nodes ...            # liveness view, hint count, keys, counters per node
```

A single node can also be started by hand: `uv run kv-node --id n1 --peers n1=127.0.0.1:7001,n2=127.0.0.1:7002,n3=127.0.0.1:7003 --data-dir data/n1`.

### Test

```
uv run pytest -q
```

28 tests, about 25 s. They cover the ring (balance, about 1/n keys move on a join), the checker (known linearizable and non-linearizable histories), in-process nodes (quorum rules, write-back, hints, WAL replay, the new-old inversion with and without write-back) and a real 5-process cluster: sharding, no acknowledged write lost when a node is SIGKILLed and later restarted, failure detection plus hint drain, and a Jepsen-style run with a kill nemesis and a pause nemesis checked by the linearizability checker.

### Benchmark and experiments

```
uv run python scripts/bench.py --trials 3 --duration 5        # nodes sweep + quorum sweep
uv run python scripts/plot_bench.py --tag bench               # results/bench_nodes.png, bench_quorum.png
uv run python scripts/linearizability.py --seeds 5            # checker vs protocol variants
uv run python scripts/inversion_demo.py                       # deterministic new-old inversion
bash scripts/run_all_experiments.sh                           # everything, about 20 minutes
```

All numbers in DEVLOG.md come from files in `results/`. The benchmarks were run on an M4 Pro while 14 other heavy jobs shared the machine (1 minute load average 200 to 300 on 14 cores, recorded per row in the CSVs), so absolute throughput is pessimistic and noisy. The per-op CPU and replica-RPC counts are measured inside the nodes and are much less sensitive to that.

### Status

| Milestone | Status |
| --- | --- |
| v0: consistent hashing with vnodes, N=3 replication | done |
| v0: configurable quorums, ABD versions, read write-back | done |
| v0: hinted handoff and retries, heartbeat failure detector | done |
| v0: WAL durability and crash recovery | done |
| v0: 5-node process harness with kill and pause nemesis | done |
| v0: Jepsen-style history recording and linearizability checker | done |
| v0: throughput and latency vs cluster size and vs quorum | done |
| v1: Raft replication (leader election, log replication, snapshots) | planned |
| v1: rebalancing and data transfer on membership change | planned |
| v1: anti-entropy with Merkle trees (repair keys no one reads) | planned |
| v2: C++ rewrite with a binary protocol | planned |

### What was cut from v0

Nothing from the v0 list was dropped, but some pieces are deliberately minimal:

* Network partitions between live nodes are not simulated. The nemesis only kills (SIGKILL) and pauses (SIGSTOP) processes; there is no per-link packet dropping.
* Hinted handoff uses strict quorums: hints repair a replica after it returns but never count toward W. A sloppy quorum was rejected for safety (see DESIGN.md).
* Membership is static. Ring math for joins is tested, but no data is moved.

### Known issues

* Throughput is low (hundreds to a few thousand ops/s) because every hop is JSON over asyncio in CPython. That is the motivation for the C++ port, not a protocol limit.
* No backpressure on the RPC writers and no request batching; under overload latency grows instead of requests being shed.
* WAL writes are flushed to the OS but not fsynced unless `--fsync` is passed, so a machine crash (not a process crash) can lose the most recent acknowledged writes.
* Hints live in the coordinator's memory. If the coordinator crashes before delivering a hint, the replica that missed the write stays stale until a later write to that key (a read only writes back when fewer than W replicas hold the newest value). This does not break safety, since W replicas still hold the write, but it reduces redundancy; anti-entropy is planned.
* The checker is exponential in the worst case. Histories with a single very hot key and many clients take several seconds per key (see DEVLOG).
