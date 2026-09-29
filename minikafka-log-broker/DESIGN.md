# minikafka design (v0)

This document describes the single-broker v0: what the pieces are, how bytes move between them, which invariants the code relies on, and which trade-offs I took or rejected.

## Architecture

```
   producer app                                             consumer app
  +-------------------+                                  +---------------------+
  | Producer          |                                  | Consumer            |
  |  per-partition    |                                  |  join / heartbeat   |
  |  batch buffers    |                                  |  positions map      |
  +---------+---------+                                  +----------+----------+
            | Client (1 TCP conn, 1 request in flight)              | Client
            | Produce(topic, p, records)          Fetch(p, offset, max_bytes, max_wait)
            v                                                       v
  +---------------------------------------------------------------------------------+
  |                                   Broker                                        |
  |                                                                                 |
  |  accept thread (poll + self-pipe) --> one thread per connection                |
  |                                          |                                      |
  |                                    handle(frame)                                |
  |          +--------------------+----------+-----------+-----------------------+  |
  |          |                    |                      |                       |  |
  |   topics_ map           GroupCoordinator       OffsetStore             long-poll |
  |   (shared_mutex)        members, generation,   map<group/topic/p,      condvar + |
  |          |              range assignment       offset>                 epoch     |
  |          v                                          |                            |
  |   Topic -> PartitionLog[p] (mutex)                  v                            |
  |          |                               PartitionLog "__consumer_offsets-0"     |
  |          v                                                                       |
  |   Segment 0 | Segment 1 | ... | active Segment                                   |
  +---------------------------------------------------------------------------------+
            |
            v  on disk:  data/<topic>-<p>/00000000000000000000.log
                                         00000000000000000000.index
                                         00000000000000004817.log ...
```

### Produce path

1. The producer appends encoded records to a per-partition buffer (`append_record`) and sends the buffer when it reaches `batch_records` or `batch_bytes`, or on `flush()`.
2. The broker reads the whole frame, then `validate_batch` walks it: every length is bounds-checked and every CRC-32C is verified. Nothing touches the log until the whole batch is known good.
3. `PartitionLog::append` takes the partition mutex, rolls to a new segment if needed, writes the offsets into the record headers in place (the records arrived with offset 0), and issues one `pwrite` for the whole batch. While walking the batch, the segment adds an index entry at every record boundary where 4 KiB or more were written since the previous entry, so large batches are indexed inside, not just at their start. With a flush policy other than `none`, the segment is fsynced before the lock is released.
4. The broker bumps the long-poll epoch (only if any fetch is waiting) and replies with the base offset.

### Fetch path

1. For each requested partition, `PartitionLog::read` takes the mutex just long enough to check the offset range, pick the segment (binary search on base offsets), look up the floor index entry (binary search), and snapshot the segment size. It holds a `shared_ptr` to the segment.
2. Outside the lock it `pread`s one chunk starting at the index position, skips records below the target offset, and cuts at the last whole record that fits in `max_bytes`. The first record is always returned whole even if it exceeds `max_bytes`, so an oversized record cannot wedge a consumer.
3. If no partition had data and `max_wait_ms > 0`, the fetch sleeps on the broker condvar until a produce bumps the epoch or the deadline passes, then reads again.
4. The response carries the raw record bytes exactly as they sit in the file. The consumer parses and CRC-checks them.

## Formats

### Record (identical on the wire and on disk)

| field | type | notes |
|---|---|---|
| offset | u64 | 0 from producers; assigned by the broker |
| payload_len | u32 | bytes after the crc field |
| crc | u32 | CRC-32C of the payload |
| timestamp_ms | i64 | producer wall clock |
| key_len | u32 | |
| key | bytes | |
| value | bytes | payload_len minus 12 minus key_len |

Header is 16 bytes, fixed payload overhead 12 bytes, so a record costs 28 bytes plus key and value. All integers are little-endian (`static_assert` on host endianness).

### Sparse index entry

`{u32 offset - base_offset, u32 file position}`, 8 bytes, appended to `<base>.index`. One entry per `index_interval_bytes` (4 KiB default) of log data. For a 64 MiB segment that is at most 16384 entries, 128 KiB in memory.

### Frames

```
request:  u32 len | u16 api_key | u32 correlation_id | body
response: u32 len | u32 correlation_id | u16 error_code | body (empty on error)
```

APIs: CreateTopic, Metadata, Produce, Fetch (multi-partition, long poll), ListOffsets, OffsetCommit, OffsetFetch, JoinGroup, Heartbeat, LeaveGroup. Exact bodies are listed in `include/minikafka/protocol.h`.

## Key data structures

`Segment` owns two fds, the in-memory index vector, `size_` and `next_offset_`. Only `read_from` runs without the partition lock, and it only uses the fd plus arguments captured under the lock.

`PartitionLog` owns `vector<shared_ptr<Segment>>` sorted by base offset, the running byte total for retention, and one `std::mutex`. The mutex is held for the pwrite (and the fsync, if configured) but not for reads of file data.

`Broker` owns `map<string, Topic>` under a `shared_mutex` (topics are only ever added), the `OffsetStore`, the `GroupCoordinator`, and the connection threads.

`OffsetStore` is a `map<"group\0topic\0partition", offset>` plus a `PartitionLog` in `__consumer_offsets-0`. Commits append a record then update the map under one mutex; startup replays the log.

`GroupCoordinator` keeps per group a generation number, an ordered `map<member_id, Member>` and the current assignment. Any membership change bumps the generation and recomputes a range assignment on the broker.

## Invariants

1. Offsets within a partition are dense and strictly increasing: record i of the log has offset `log_start + i`. Appends are serialised by the partition mutex and offsets are assigned under it.
2. Bytes of a segment below its current `size_` never change. This is what lets fetches read without the lock.
3. Every record in a segment file passed `validate_batch` before it was written, so on-disk corruption can only come from the disk or a torn write, and recovery can treat any bad record as the end of the log.
4. Only the last segment can have a torn tail. Before a roll the outgoing segment is flushed with the configured policy. On startup only the last segment is scanned: each record must have the expected next offset, a sane length and a matching CRC; the file is truncated at the first failure and its index is rebuilt.
5. Index entries are strictly increasing in both offset and position and always point at a record boundary. A sealed segment whose index breaks this is rebuilt by scanning.
6. After retention, `size_bytes >= retention_bytes` whenever anything was deleted, and `size_bytes <= retention_bytes + segment_bytes`. The active segment is never deleted.
7. A committed offset means "next offset to read". The commit is durable once it is in the internal log, under the same flush policy as data.
8. A commit from a group member must carry the current generation. A member that was rebalanced away cannot move the new owner's offset backwards or forwards.

## Durability model

`FlushPolicy::None` acknowledges after `pwrite`. The data is in the OS page cache, so it survives a broker crash or kill (this is what the restart tests check) but not a kernel panic or power loss. `Fsync` calls `fsync(2)` before acknowledging; on macOS this pushes data to the drive but does not flush the drive's write cache. `FullFsync` uses `fcntl(F_FULLFSYNC)`, which does. Kafka itself defaults to the first option and relies on replication for durability; v0 has no replication, so the choice is exposed and measured in `results/flush_policy.csv`.

## Trade-offs

### Chosen

- Thread per connection with blocking sockets. It is the simplest correct server and keeps each request handler straight-line code. A benchmark has a handful of connections, so it is not the bottleneck at this scale. Rejected for v0: a kqueue event loop, which is the right design for thousands of connections but adds a state machine per request and would have eaten the time for correctness tests.
- Same byte format on the wire and on disk. The broker never re-encodes: produce patches offsets in place and does one write; fetch copies a byte range. This is Kafka's main trick and it is also what makes zero-copy `sendfile` a small step later.
- Validate the whole batch before appending. Costs one CRC pass on the broker, which the ARMv8 CRC32C instructions make cheap, and in exchange invariant 3 holds and recovery stays simple.
- Sparse index with a 4 KiB interval, loaded fully in memory. A lookup is a binary search plus at most about 4 KiB of scanning, which the fetch does inside a single `pread` chunk anyway. Rejected: a dense index (one entry per record), which is 8 bytes per record for no fetch benefit, and mmap of the index file, which Kafka uses but which complicates growth and truncation for little gain at this size.
- Reads outside the lock via a size snapshot and `shared_ptr<Segment>`. Fetches never block produces, and retention can unlink a segment while a fetch is still reading it (the fd stays valid until the last reference drops).
- Recovery scans only the active segment. Startup cost is bounded by one segment, not by the size of the log. The cost is that a corrupted sealed segment is not detected at startup; a consumer would see a CRC error when reading it.
- Broker-side range assignment, no SyncGroup. One round trip to join, and assignment logic lives in one place. The cost is that there is no join barrier: during a rebalance, two members can fetch the same partition until the old owner's next heartbeat. Generation fencing on commits turns that into duplicate delivery rather than lost or rewound offsets (at-least-once).
- Committed offsets in an internal log, replayed on start. Same durability path as data, and no second storage engine. Without compaction the log only grows, so retention is disabled for it; compaction is the roadmap fix.
- Little-endian integers. Kafka uses big-endian. Every machine this runs on is little-endian, so encode and decode are `memcpy`, and the code asserts the host order.

### Rejected or deferred

- Pipelined producer requests. One request in flight per connection makes per-partition ordering trivially correct but caps small-batch throughput at one round trip per batch (visible in `results/produce_throughput.csv` at batch size 1). Kafka allows several in-flight requests plus idempotence to keep ordering; that belongs with exactly-once.
- Group-commit fsync. With `Fsync`, each append pays its own fsync while holding the partition lock. Batching fsyncs across concurrent producers would help but requires acknowledging asynchronously.
- Time-based retention, compaction, topic deletion, replication. On the roadmap.
