# Design: kvstore v0

A sharded, replicated key-value store. Every node runs the same process: it is a storage replica for some keys and a request coordinator for any key. There is no leader and no central metadata service. The v0 goal is a correct, testable protocol, not raw speed, so it is written in Python asyncio with one OS process per node.

### Components and data flow

```
                          client (KVClient)
                 picks a random node as coordinator;
            retries on another node only if connect failed
                               |
                               |  put(k, v) / get(k)      TCP, 4-byte length + JSON
                               v
 +--------------------------- node n2 (coordinator for this request) ---------------------------+
 |                                                                                               |
 |   HashRing (64 vnodes per node, blake2b 64-bit)                                               |
 |      preference_list(k, N=3) -> [n4, n1, n2]                                                  |
 |                                                                                               |
 |   Coordinator                                                                                 |
 |      put: phase 1  rget -> wait for R replies -> counter = max + 1                            |
 |           phase 2  rput(version, v) -> wait for W acks -> reply ok                            |
 |      get: rget -> wait for R replies -> pick max version                                      |
 |           if fewer than W replicas hold it: rput it (write-back) until W do -> reply          |
 |                                                                                               |
 |   FailureDetector            HintStore                         Store                          |
 |    ping every peer 100 ms     hints[peer][key] = (ver, v)       dict key -> (version, value)  |
 |    suspect after 500 ms       filled when an rput fails or       + append-only WAL             |
 |    silence                    the peer is suspected;            (wal.jsonl, flushed per write)|
 |                               replayed every 200 ms once                                      |
 |                               the peer is alive again                                         |
 +---------------+-------------------------------+-------------------------------+---------------+
                 | rget/rput                     | rget/rput                     | local call
                 v                               v                               v (no TCP)
          +-------------+                 +-------------+                 +-------------+
          |  node n4    |                 |  node n1    |                 |  node n2    |
          |  replica    |                 |  replica    |                 |  replica    |
          |  Store+WAL  |                 |  Store+WAL  |                 |  Store+WAL  |
          +-------------+                 +-------------+                 +-------------+

 Background, on every node:  heartbeat loop (ping all peers)  |  hint loop (deliver hints to live peers)
```

The source files map onto the boxes: `ring.py` (HashRing), `rpc.py` (framing, multiplexed connections, server), `storage.py` (Store + WAL), `node.py` (coordinator, replica handlers, failure detector, hints), `client.py`, `cluster.py` (process harness), `checker.py` (linearizability checker), `jepsen.py` (workload + nemesis), `scenarios.py` (deterministic failure scenarios).

### Key data structures

#### Hash ring

A sorted Python list of 64-bit tokens plus a `token -> node` dict. Each physical node contributes `vnodes` tokens (default 64) derived from `blake2b(f"{node}#{i}")`. Lookup is a `bisect` on the key's hash followed by a clockwise walk that collects the first N *distinct* physical nodes. I use blake2b rather than Python's `hash()` because `hash()` is salted per process and every node must compute the same placement.

#### Versions

A version is the tuple `(counter, writer_node_id, seq)`, compared lexicographically. The counter carries the ordering: a new write takes `max(counter seen in a read quorum) + 1`. The `(writer_node_id, seq)` suffix only exists to make versions unique, so two concurrent writers that chose the same counter still produce distinct, totally ordered versions. `seq` starts from the wall clock in microseconds when the node boots, so a restarted node never reissues a version it issued before the crash.

#### Store and WAL

`dict[key] -> (version, value)`. `apply(key, ver, value)` installs the record only if `ver` is strictly newer (last-writer-wins by version, which makes replica writes idempotent and commutative, so retries and hint replays are harmless). Every accepted record is appended to `wal.jsonl` and flushed to the OS before the replica acks. On start the WAL is replayed, and a torn last line is ignored.

#### Hints

`hints[target_peer][key] = (version, value)`, keeping only the newest version per key. A hint is created when an `rput` to a replica fails or times out, or when the failure detector already suspects that replica so the coordinator did not even try. A background loop delivers hints to peers the detector sees as alive and deletes a hint only if it was not replaced while in flight.

#### Failure detector

`last_seen[peer]`, updated by successful heartbeats (a ping to every peer every 100 ms) and by any successful RPC reply. A peer is suspected when `now - last_seen > suspect_after` (500 ms). The detector is advisory: coordinators skip suspected replicas, but if skipping them would make the quorum impossible they try everyone anyway, because the detector can be wrong.

### Invariants

1. Quorum intersection. With R + W > N (checked at startup: 1 <= R, W <= N), every read quorum shares at least one replica with every write quorum of the same key.
2. Monotonic replicas. A replica's version for a key never decreases; `apply` ignores anything not strictly newer.
3. Acknowledged means durable on W replicas. A put returns ok only after W replicas have applied the version and written it to their WAL.
4. Reads do not expose unreplicated data. A get returns a version only after at least W replicas hold it (it either saw W holders in its read quorum or wrote the value back until W acked). Together with 1 this means any later read's quorum contains that version or a newer one. This is the invariant that makes each key a linearizable register (multi-writer ABD).
5. Unique versions. No two distinct writes share a version, so "max version" is well defined and replicas can never disagree about which value a version names.
6. Definite versus unknown failures. The client reports `definite: true` only when the operation provably had no effect (could not connect, or phase 1 of a put failed before anything was written). Everything else that is not ok is reported as unknown, and the checker treats it that way.

### Failure model and what is guaranteed

Crash-stop and crash-recover nodes (SIGKILL, then restart from the WAL), paused nodes (SIGSTOP/SIGCONT), and slow replicas. Messages between live nodes are delivered or the connection errors out (TCP on localhost). Under that model, with R + W > N:

* each key behaves as a linearizable register, as long as at most N - max(R, W) replicas of a key are unavailable (with N=3, R=W=2: one replica);
* no acknowledged write is lost when a single node fails, including when it later comes back, because the write lives on W >= 2 replicas and each of them keeps it in its WAL;
* operations fail cleanly (not silently) when a quorum is unreachable.

Not guaranteed in v0: network partitions between live nodes are not simulated, WAL writes are not fsynced by default (a process crash is survived, a machine crash may lose the last writes unless `--fsync`), and membership is static.

### Trade-offs: chosen and rejected

#### Chosen: leaderless quorum replication with ABD-style reads and writes

Leaderless replication lets any node coordinate, so there is no failover step and a single node failure costs nothing but a slower quorum. The price is two round trips per write (version query, then write) and a possible second round trip for reads that land on a partially replicated value. I accepted that because the extra round trip is what buys linearizability without clocks or a leader.

#### Rejected: timestamps from the coordinator's clock (Cassandra-style LWW)

One round trip per write instead of two, but correctness then depends on clock synchronization. Two writes a few microseconds apart from different coordinators can be ordered backwards, and a later write can silently lose. Correctness first, so I paid for the extra round trip.

#### Rejected: vector clocks and siblings (Dynamo-style)

Vector clocks detect concurrent writes and return all siblings to the client for resolution. That gives availability under partitions but pushes merge logic to the application and is not a register. The goal here was a checkable register semantics, which the version-counter scheme gives directly.

#### Chosen: strict quorums with coordinator-held hints, rejected: sloppy quorums

In Dynamo, if a preference-list node is down, the write goes to the next healthy node on the ring and counts toward W (a sloppy quorum). That keeps writes available when more than N - W replicas of a key are down, but a sloppy write quorum need not intersect a later read quorum, so reads can miss acknowledged writes. I kept quorums strict (only the N preference-list replicas count) and used hinted handoff only as a repair mechanism: the coordinator remembers what a down replica missed and replays it on recovery. Availability is lower in a multi-failure scenario, safety is preserved.

#### Chosen: read write-back always waits for W

Plain Dynamo-style read repair fixes stale replicas in the background after replying. That allows the new-old inversion that `tests/test_node.py::test_without_writeback_new_old_inversion_is_caught` demonstrates. Waiting for W acks on the write-back costs latency only when a read hits a partially replicated value.

#### Chosen: one TCP connection per peer with request multiplexing

Each request has an id; responses can arrive out of order. That avoids head-of-line blocking of a request-response connection and avoids a connection pool. The server runs each request in its own task. I did not implement write batching or backpressure (`drain`) yet; README lists this under known issues.

#### Chosen: JSON framing

Readable in a packet capture and zero dependencies. It costs CPU; the C++ port will use a binary format.

#### Chosen: static membership

Rebalancing on membership change is a later milestone. With vnodes, adding a node moves about 1/(n+1) of the keys, which `tests/test_ring.py` checks, but there is no data transfer protocol yet.

#### Rejected for v0: phi-accrual failure detection and gossip

A fixed timeout on direct all-to-all pings is simpler and, at 5 to 9 nodes, the O(n^2) ping traffic is small (measured as idle CPU in the benchmark). Gossip-based membership would be needed for large clusters.
