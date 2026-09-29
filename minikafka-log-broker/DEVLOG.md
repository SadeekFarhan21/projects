# DEVLOG: building a Kafka-like log from scratch

### What I wanted to build

I wanted to understand why Kafka is fast and what it actually promises, by building the core of it myself: a single broker that stores topics as partitioned, append-only logs on disk, speaks a binary protocol over TCP, and lets consumer groups commit their position and pick up where they left off after a restart. The v0 goal was a system small enough to finish and verify in one session but real enough to measure: batched produce, fetch from an offset, segmented logs with a sparse index, size retention, crash recovery, consumer groups with offsets stored in an internal log, client libraries, tests for ordering, durability and resume, and throughput and latency benchmarks.

Replication, a controller, zero-copy `sendfile`, compaction and exactly-once are explicitly out of scope for v0 and are on the roadmap in README.md.

### Theory

A Kafka partition is a log: an ordered sequence of records where each record gets the next integer offset. Everything else follows from that.

- Appends only touch the end of a file, so writes are sequential and the OS page cache absorbs them. Reads by consumers that are caught up are served from the same cache.
- A consumer's entire state is one number per partition, the next offset to read. That makes committing progress cheap, and it moves "which messages have I seen" out of the broker.
- Ordering is only promised within a partition. Keys are hashed to partitions so that all events for one key stay ordered, and partitions are the unit of parallelism for consumer groups.
- Finding offset N without scanning the log needs an index. Kafka's index is sparse: one entry every few KiB of log mapping a relative offset to a file position. A lookup is a binary search to the nearest earlier entry followed by a short forward scan. Offsets are dense, so the floor entry is always close.
- Batching amortises per-request costs (syscalls, round trips, lock acquisitions, fsyncs) over many records. The experiments below are mostly about how much batching buys.
- Durability is a policy choice. Acknowledging after `write` survives a process crash (the data is in the kernel) but not power loss. `fsync` on macOS pushes data to the drive but not through its write cache; `fcntl(F_FULLFSYNC)` does. Kafka acknowledges after the write and relies on replication; a single broker has to choose.

### Architecture

Clients hold one TCP connection each with one request in flight. The broker accepts connections on one thread and serves each connection on its own thread. A request frame is decoded and dispatched to one of ten APIs (CreateTopic, Metadata, Produce, Fetch, ListOffsets, OffsetCommit, OffsetFetch, JoinGroup, Heartbeat, LeaveGroup).

```
Producer --Produce--> Broker --> Topic --> PartitionLog[p] --> Segment files (.log + .index)
Consumer --Fetch----> Broker --> PartitionLog.read (pread outside the lock) --> raw record bytes
Consumer --Join/Heartbeat--> GroupCoordinator (generation, range assignment)
Consumer --OffsetCommit----> OffsetStore --> PartitionLog "__consumer_offsets-0"
```

The record format is identical on the wire and on disk (16 byte header with offset, length and CRC-32C, then timestamp, key and value), so the broker never re-encodes: produce validates the batch, patches offsets into the headers in place and does one `pwrite`; fetch copies a byte range out of the file. Committed offsets are themselves records in an internal partition log, replayed into a map on startup. DESIGN.md has the full diagram, formats, invariants and trade-offs.

### Implementation

About 3,000 lines of C++20 across `src/`, `include/minikafka/`, `tools/` and `bench/`, plus about 830 lines of tests.

- `record.cpp`: encode, validate and decode records. `validate_batch` bounds-checks every length and verifies every CRC before anything is appended, so a bad batch is rejected whole.
- `crc32c.cpp`: CRC-32C using the ARMv8 CRC instructions, with a table-driven software fallback that the tests check against the hardware path.
- `segment.cpp`: one `.log` and one `.index` file per segment. `append` assigns offsets, writes the batch with a single `pwrite`, and adds an index entry at each record boundary once 4 KiB has been written since the last entry. `recover` scans the active segment on startup, truncates at the first record with a wrong offset, bad length or bad CRC, and rebuilds the index. A sealed segment whose index is not strictly increasing gets its index rebuilt.
- `partition_log.cpp`: the list of segments under one mutex, rolling at `segment_bytes`, and size retention that deletes whole segments from the front but never the active one. Reads take the lock only to pick the segment and snapshot its size, then `pread` without it; `shared_ptr<Segment>` keeps the fd alive if retention unlinks the file mid-read.
- `broker.cpp`: accept loop with a self-pipe for shutdown, thread per connection, the request handlers, and long-poll fetch using a condition variable plus an epoch counter that produce bumps only when someone is waiting.
- `group_coordinator.cpp` and `offset_store.cpp`: membership with session expiry, a generation number bumped on every change, broker-side range assignment, and commits that are rejected if they carry a stale generation.
- `producer.cpp` and `consumer.cpp`: the producer batches per partition (by record count or bytes) and hashes keys to partitions; the consumer joins, heartbeats, fetches its assigned partitions with long polling, and commits.
- `mk-broker`, `mk-cli` and `mk-bench` are the server, a command-line client and the benchmark harness.

Testing: 31 GoogleTest cases, run in a release build, an ASan plus UBSan build and a TSan build, plus `scripts/smoke_test.sh`, which runs the real broker binary, restarts it, and checks that data and committed offsets survive. All pass (`results/test_summary.txt`, `results/smoke_test.txt`).

### Problems

- The sparse index was only sparse between batches. My first `Segment::append` added at most one index entry per append, at the start of the batch. With 1000 record batches of 10 KB each, that meant one index entry per 10 MB of log, so a fetch starting in the middle of a batch had to read and skip megabytes to find its offset. The first benchmark run, kept in `results/run1_batch_start_index/consume_throughput.csv`, shows 10,000 B messages consumed at 14.5 MB/s with 4 KiB fetches and never above 83 MB/s. The fix walks the batch while appending and adds an entry at every record boundary past the 4 KiB interval; the regression test `Segment.IndexHasEntriesInsideLargeBatches` checks that every floor lookup lands within one interval of the target. The current run gets 225 MB/s for the same point and 2337 MB/s with 4 MiB fetches (`results/consume_throughput.csv`). That before and after is confounded by machine load (the first run was at a load average of 275 to 330), so I only claim the direction, not the ratio.
- Apple clang's sanitizer runtime hangs. On macOS 26.5 with Apple clang 17, an ASan build deadlocked during runtime initialisation, even for an empty program. I switched the sanitizer builds to Homebrew LLVM (`-DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++`), and CMake now prints a warning if you try the Apple toolchain. Upstream clang emits DWARF 5, which Apple's linker warns about, so non-Apple builds use `-gdwarf-4`.
- The machine was shared with many other heavy jobs. The first full benchmark run happened at a load average of 180 to 330 on 14 cores, and the numbers were both low and unstable. For example, 1000 B produces at batch 1000 went from 294 MB/s (`results/run2_loaded/produce_throughput.csv`) to 2528 MB/s once load dropped to about 8 (`results/produce_throughput.csv`). I added the 1 minute load average to every CSV row, report medians of 3 repetitions with min and max, and re-ran everything once load was around 8 to 11. The loaded runs are kept in `results/run2_loaded/` and `results/run1_batch_start_index/`.
- One noisy point even at low load. In the first low-load produce run, 100 B messages at batch 100 measured 14.6 MB/s, below batch 10, with a min to max spread of 11.7 to 38.6 (`results/produce_throughput_noisy.csv`). TCP_NODELAY was already set, so it was not Nagle. Running the same mode again gave 233.4 MB/s with a spread of 229.6 to 240.2 (`results/produce_throughput.csv`), so it was interference from other processes. I kept both files and use the second.
- Lower send rates had worse tail latency. At 1,000 msgs/s with no fsync, p99 was 3018 us, while at 10,000 msgs/s it was 142 us (`results/latency.csv`). At the low rate the producer sleeps between sends and the broker and consumer threads park, so every message pays for waking threads (and probably cores leaving low power states) up. I have not confirmed the cause by profiling; the pattern is consistent across the loaded and unloaded runs.

### Experiments

All experiments use `mk-bench`, which starts a broker in the same process with a fresh data directory on the internal SSD and talks to it over loopback TCP through the real client code. Throughput points are the median of 3 repetitions. Hardware: Apple M4 Pro, 14 cores, 48 GB, macOS 26.5, Apple clang 17, `-O2` (`results/bench_env.txt`). The machine was running other agents' builds and benchmarks at the same time; the 1 minute load average was 8 to 11 during the reported runs and is recorded per row. These are loaded-machine numbers.

1. Producer throughput vs batch size (1, 10, 100, 1000 records) and message size (100 B, 1 KB, 10 KB), one partition, no fsync. The batch is pre-encoded, so this measures transport plus broker. Output: `results/produce_throughput.csv`, plot `results/produce_throughput.png`.
2. Consumer throughput vs fetch `max_bytes` (4 KiB to 4 MiB) and message size, reading a pre-filled partition from offset 0 with CRC verification of every record. Output: `results/consume_throughput.csv`, plot `results/consume_throughput.png`.
3. End-to-end latency: a producer paced to a fixed schedule stamps each message with `steady_clock`, and a consumer long-polls and records receive time minus send time. Five configurations covering rate, batch and flush policy. Output: `results/latency.csv` (percentiles) and `results/latency_raw.csv` (every sample, gitignored because it is 5.5 MB), plot `results/latency_cdf.png`.
4. Cost of durability: 1 KB messages at each batch size under the three flush policies. Output: `results/flush_policy.csv`, plot `results/flush_policy.png`.
5. Producer scaling: 1, 2, 4 and 8 concurrent producers, each to its own partition, 1 KB messages, batch 100. Output: `results/producer_scaling.csv`, plot `results/producer_scaling.png`.

### Results

Batching dominates producer throughput. With 100 B messages, going from batch 1 to batch 1000 raised throughput from 32,886 to 12.7 M msgs/s (3.3 to 1274 MB/s), a factor of about 390 (`results/produce_throughput.csv`). At batch 1 every message size lands near 25,000 to 33,000 msgs/s, which is the round trip rate of one blocking request per connection, so the protocol and not the disk is the limit there. Large messages saturate earlier: 10 KB messages reach 1311 MB/s at batch 10 and only 2021 MB/s at batch 1000. The best point was 1 KB messages at batch 1000, 2528 MB/s.

Fetch size plays the same role for consumers. 100 B messages went from 92.8 MB/s with 4 KiB fetches to 2150 MB/s (21.5 M msgs/s) with 1 MiB fetches, and 4 MiB did not help further (2135 MB/s) (`results/consume_throughput.csv`). 10 KB messages are flat at about 220 MB/s for 4 and 16 KiB fetches because each fetch returns just one record (the first record is always returned whole even if it exceeds `max_bytes`), and reach 2337 MB/s at 4 MiB.

End-to-end latency without fsync is tens of microseconds at the median: p50 41.5 us, p99 142 us, p99.9 1.2 ms at 10,000 msgs/s, and p50 45.5 us, p99 185 us at 50,000 msgs/s in batches of 10 (`results/latency.csv`). With `fsync` at 1,000 msgs/s, p50 is 115 us and p99 586 us. With `F_FULLFSYNC` at 200 msgs/s, p50 is 4.0 ms and p99 7.8 ms, which is the cost of actually getting data through the drive's cache.

Durability costs depend heavily on batching (`results/flush_policy.csv`). At batch 100, 1 KB messages: 1188 MB/s with no flush, 595 MB/s with `fsync`, 23 MB/s with `F_FULLFSYNC`. At batch 1000 the gap narrows (1520, 1324 and 198 MB/s) because each flush is amortised over 1 MB. At batch 1, `F_FULLFSYNC` managed 231 msgs/s, about 4.3 ms per message.

Producer scaling across partitions is sublinear: 1340 MB/s with 1 producer, 1491 with 2, 1813 with 4 and 2061 with 8 (`results/producer_scaling.csv`), 1.54 times the single producer rate with 8 times the producers. Each producer has its own partition and lock, so it is not lock contention on the log. The likely limits are memory bandwidth for copying through loopback TCP and the page cache, and the other load on the machine, but I have not profiled this.

For contrast, the same benchmarks at a load average of 180 to 230 gave 294 MB/s for 1 KB messages at batch 1000 and 376 MB/s at best for consumers (`results/run2_loaded/`), and p99 latency of 172 ms at 1,000 msgs/s (`results/run1_batch_start_index/latency.csv`). Throughput numbers from a shared machine should be read as lower bounds.

### What I would change

- Pipeline producer requests. One request in flight caps unbatched throughput at about 30,000 msgs/s per connection. Allowing several in flight needs per-partition sequence numbers to keep ordering on retry, which is also the first step toward idempotent producers and exactly-once.
- Add a SyncGroup style barrier so that a rebalance revokes partitions before reassigning them. Today two consumers can briefly read the same partition; generation fencing makes that duplicate delivery rather than lost progress, but it is still a weaker guarantee than Kafka's.
- Replace thread per connection with a kqueue event loop before trying many clients, and serve fetches with `sendfile` since the on-disk format already equals the wire format.
- Group-commit fsyncs across concurrent producers so the `fsync` policy does not serialise one flush per append under the partition lock.
- Compact `__consumer_offsets`. It only grows today, and startup replays all of it.
- Profile before explaining. Producer scaling and the low-rate latency tail both have plausible explanations above that I did not verify, and I would run them on an idle machine with `perf`-style sampling (Instruments on macOS) before trusting them.

### Verification

On 2026-09-26 an independent clean release rebuild passed all 31 tests. The sanitizer (ASan plus UBSan) and TSan builds and the benchmarks were not rerun independently; the sanitizer, TSan and benchmark results above are from the original runs. The benchmarks were measured on a shared machine with the 1 minute load average recorded per row, so they should be treated as indicative.
