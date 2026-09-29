---
layout: post
title: "What Durability Costs a Kafka-Style Log Broker"
tags:
  - distributed-systems
  - storage
  - cpp
description: >-
  A single-broker event log in C++ with segmented partitions, crash recovery and
  consumer groups, and a measured look at what each flush policy costs.
date: 2026-09-29 02:17:11
---


I wrote minikafka, a single-broker event log in C++20 that follows Kafka's storage design closely<sup>[[1]](#ref-1)</sup>. Topics are split into partitions, and each partition is an append-only log of segment files with a sparse offset index. Producers send batches over a small binary TCP protocol, consumers fetch from an offset with long polling, and consumer groups get partitions assigned by the broker and commit their positions into an internal log, so a group resumes where it left off after a restart. Recovery after a crash truncates a torn tail, and size-based retention deletes old segments.

The code is about 2,700 lines of C++ across the library, the broker, a CLI and a benchmark harness, plus about 830 lines of GoogleTest. There are **31 tests**, and on 2026-09-26 an independent clean release rebuild passed all 31. The ASan plus UBSan and ThreadSanitizer builds also passed in my own runs, but those, and the benchmarks, were not rerun independently.

The headline measurement is how much batching matters. With 100 B messages, going from one record per produce request to 1,000 moved throughput from **32,886 to 12,744,103 messages per second**, a factor of about 390. End to end, without an fsync, a message took 41.5 µs at the median from send to receive at 10,000 messages per second. Pushing data all the way through the drive's write cache with `F_FULLFSYNC` raised that to **4.0 ms**. All benchmark numbers were measured on a shared Apple M4 Pro that was also running other people's builds and benchmarks, with the 1 minute load average recorded in every row (8 to 11 for the reported runs). They are indicative, not a clean benchmark.

v0 is one broker, with no replication.

The argument of this post is that Kafka's speed comes from a few simple decisions about bytes, and that the hard part of reproducing it was not the protocol but keeping the index, the locks and the benchmarks honest. Skip to [Problems](#problems) for the bugs.

## What I Wanted to Build

I wanted to understand why Kafka is fast and what it actually promises, by building the core of it myself. The v0 scope was chosen to be small enough to finish and verify but real enough to measure.

- Topics with partitions, each an append-only log of segment files on disk.
- A sparse offset index per segment, so that fetching from offset $N$ does not scan the log.
- Crash recovery that finds and removes a torn write at the end of the log.
- A binary TCP protocol with batched produce and long-poll fetch.
- Consumer groups with broker-side partition assignment, and committed offsets stored in an internal log so they survive a restart.
- Size-based retention.
- Producer and consumer libraries, a CLI, tests for ordering, durability and resume, and throughput and latency benchmarks.

Replication, a controller, zero-copy `sendfile`, compaction and exactly-once delivery are out of scope for v0. They are on the roadmap, and a few of the measurements below show exactly where their absence costs something.

## Theory

### A Partition Is a Log

A Kafka partition is an ordered sequence of records where each record gets the next integer offset. Almost everything else follows from that one decision.

Appends only touch the end of a file, so writes are sequential, and the operating system's page cache absorbs them. A consumer that is caught up reads data that was written moments ago, which is still in the same cache, so a well-behaved consumer rarely touches the disk at all.

A consumer's entire state is one number per partition, the next offset to read. That makes committing progress cheap and moves the question of which messages have been seen out of the broker.

### Ordering and Keys

Ordering is promised only within a partition. A producer that wants all events for one customer to stay in order gives them the same key, and the key is hashed to pick the partition. Partitions are also the unit of parallelism, since each partition in a group is read by exactly one consumer at a time.

### Finding Offset N

A consumer asks for "offset 1,048,576 onwards". Records have variable length, so offset $N$ is not at a computable byte position. The broker needs an index.

Kafka's index is sparse<sup>[[2]](#ref-2)</sup>. Every few KiB of log, it writes an entry mapping a relative offset to a file position. A lookup is a binary search for the last entry at or below the target, then a short forward scan through record headers. Because offsets are dense, the floor entry is never more than one interval of bytes behind the target, so the scan is bounded.

<figure class="excal" data-diagram="kafka-sparse-index"><a href="/img/diagrams/kafka-sparse-index.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/kafka-sparse-index.webp" alt="A sparse index lookup: index entries point to byte positions in the log, and fetching offset 6 binary searches to the floor entry at record 4, skips the headers of records 4 and 5, and returns records from byte 6000." width="2400" height="2161" loading="lazy" decoding="async"></a></figure>

A dense index would cost 8 bytes per record for no real gain, since the fetch reads a chunk of the file anyway. At a 4 KiB interval, a 64 MiB segment needs at most 16,384 entries, or 128 KiB, small enough to keep in memory.

### Batching

Every produce request pays fixed costs, a system call on each side, a network round trip, a lock acquisition and, if configured, an fsync. Batching amortises them over many records, and most of the experiments below ask how much that buys.

### Durability Is a Policy

Acknowledging a write after `write(2)` means the data is in the kernel. It survives the broker process crashing or being killed, but not a kernel panic or power loss. `fsync(2)` asks the kernel to push the data to the drive. On macOS, `fsync` hands data to the drive but does not flush the drive's own write cache, and `fcntl(F_FULLFSYNC)` does<sup>[[3]](#ref-3)</sup>. Kafka itself acknowledges after the write by default and relies on replication for durability<sup>[[4]](#ref-4)</sup>. A single broker has no replicas, so it has to choose, and v0 exposes the choice as a flag and measures it.

## Architecture

Clients hold one TCP connection each, with one request in flight. The broker accepts connections on one thread and serves each connection on its own thread. A request frame is decoded and dispatched to one of ten APIs, which are CreateTopic, Metadata, Produce, Fetch, ListOffsets, OffsetCommit, OffsetFetch, JoinGroup, Heartbeat and LeaveGroup.

<figure class="excal" data-diagram="kafka-broker"><a href="/img/diagrams/kafka-broker.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/kafka-broker.webp" alt="Kafka-style broker architecture: producer and consumer connections each get a broker thread that dispatches to the topics map, group coordinator, offset store and long-poll wakeups; data and committed offsets end up as segment .log and .index files on disk." width="2400" height="2969" loading="lazy" decoding="async"></a></figure>

### One Byte Format for the Wire and the Disk

The most important decision is that a record looks the same on the wire and on disk.

```
record = header (16 B) + payload
  u64 offset        0 from producers, assigned by the broker
  u32 payload_len   bytes after the crc field
  u32 crc           CRC-32C of the payload
  payload:
  i64 timestamp_ms
  u32 key_len
  key bytes
  value bytes       payload_len - 12 - key_len
```

A record costs 28 bytes plus its key and value. Because the formats match, the broker never re-encodes anything. On produce it validates the batch, writes the assigned offsets into the headers in place and issues a single `pwrite`. On fetch it copies a byte range out of the file into the response. The consumer does the parsing and CRC checking. This is Kafka's main trick<sup>[[4]](#ref-4)</sup>, and it is also what would make zero-copy `sendfile` a small change later.

Integers are little-endian, unlike Kafka's big-endian<sup>[[5]](#ref-5)</sup>. Every machine this runs on is little-endian, so encoding is a `memcpy`, and the code asserts the host byte order at compile time.

### The Produce Path

1. The producer appends encoded records to a per-partition buffer and sends it when it reaches a record count or byte limit, or when the caller asks for an explicit flush.
2. The broker reads the whole frame and validates the batch, walking it to bounds-check every length and verify every CRC. Nothing touches the log until the whole batch is known good.
3. The partition's append takes the partition mutex, rolls to a new segment if needed, patches offsets into the headers, and writes the batch. With a flush policy other than none, the segment is flushed before the lock is released.
4. If any fetch is waiting, the broker bumps a long-poll epoch and wakes it. It replies with the base offset.

### The Fetch Path

1. The partition's read takes the mutex only long enough to check the offset range, pick the segment by binary search over base offsets, look up the floor index entry, and snapshot the segment's current size.
2. Outside the lock, it `pread`s from the index position, skips records below the target, and cuts at the last whole record that fits in the fetch's byte limit. The first record is always returned whole, so an oversized record cannot wedge a consumer.
3. If no partition had data and the request allows waiting, the fetch sleeps on a condition variable until a produce bumps the epoch or the deadline passes, and then reads again.

### Invariants the Code Relies On

The design rests on eight invariants. The two that carry the most weight are that offsets in a partition are dense and assigned under the partition mutex, and that bytes of a segment below its current size never change, which is what lets fetches read without the lock. A third, that every record on disk passed batch validation, is what lets recovery treat any bad record as the end of the log.

## Implementation

### The Segment

A segment is one `.log` file and one `.index` file, named by the base offset padded to 20 digits so that a directory listing sorts in offset order. The index lives in memory as a vector of `{u32 relative offset, u32 file position}` pairs, and new entries are appended to the index file after the data they point at has been written.

The index loop is the part I got wrong first (see [Problems](#problems)). The current version walks the batch record by record while appending and drops an entry at every record boundary where 4 KiB or more has been written since the previous entry.

```cpp
void Segment::append(const uint8_t* data, size_t n, uint64_t first_offset, uint32_t count) {
  const size_t old_entries = index_.size();
  size_t pos = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (bytes_since_index_ >= index_interval_) {
      index_.push_back({static_cast<uint32_t>(first_offset + i - base_),
                        static_cast<uint32_t>(size_ + pos)});
      bytes_since_index_ = 0;
    }
    const size_t rec = kRecordHeaderSize + load_u32(data + pos + 8);
    bytes_since_index_ += rec;
    pos += rec;
  }
  pwrite_all(log_fd_, data, n, size_);
  // new index entries go out in one write, after the data they point at
  ...
}
```

Lookups use `std::upper_bound` to find the first entry past the target and step back one. Reads go through a small buffered reader that fetches a window of the file with one `pread` and serves header lookups from it, so a fetch is usually one or two system calls rather than one per record.

### Recovery

Only the last segment of a partition can have a torn tail, because before rolling to a new segment the outgoing one is flushed with the configured policy. So on startup only the active segment is scanned. Each record must have the expected next offset, a sane length and a matching CRC, and the file is truncated at the first record that fails.

```cpp
while (limit - pos >= kRecordHeaderSize) {
  const uint8_t* h = r.at(pos, kRecordHeaderSize);
  const uint64_t off = load_u64(h);
  const uint32_t len = load_u32(h + 8);
  const uint32_t crc = load_u32(h + 12);
  if (off != expect || len < kPayloadFixedSize || len > kMaxPayloadSize ||
      len > limit - pos - kRecordHeaderSize)
    break;
  const uint8_t* body = r.at(pos + kRecordHeaderSize, len);
  if (crc32c(body, len) != crc) break;
  ...  // rebuild index entries as we go
  pos += kRecordHeaderSize + len;
  ++expect;
}
if (pos != size_) ftruncate(log_fd_, pos);
```

The expected-offset check matters as much as the CRC. A torn write can leave a header whose length field happens to be plausible, and the offset is a second, independent check that the bytes belong where they are. Sealed segments trust their index, but if an index file is missing, has a partial entry, or is not strictly increasing in both offset and position, the segment rebuilds it by the same scan. The cost of scanning only the active segment is that corruption inside a sealed segment is not detected at startup. A consumer would see a CRC error when it reached it.

CRC-32C<sup>[[6]](#ref-6)</sup> uses the ARMv8 CRC instructions, eight bytes per instruction, with a table-driven software fallback. A test checks the two against each other and against known vectors.

### Reads Without the Lock

The partition mutex serialises appends. I did not want a slow consumer's disk read to block a producer, so a read holds the lock only to decide what to read.

```cpp
{
  std::lock_guard lk(mu_);
  ...  // range checks, pick segment by binary search
  seg = *std::prev(it);                   // shared_ptr<Segment>
  start_pos = seg->floor_position(offset);
  limit = seg->size();                    // snapshot, bytes below this are immutable
}
res.data = seg->read_from(start_pos, offset, max_bytes, limit);
```

Two things make this safe. Bytes below the snapshotted size never change, so a concurrent append cannot tear what the read sees. And the read holds a `shared_ptr` to the segment, so if retention deletes the segment in the middle of the read, the file is unlinked but the descriptor stays open until the last reference drops.

### Retention

Size retention deletes whole segments from the front, but only if what remains is still at least the retention target, and it never deletes the active segment. After retention runs, the log's size is between the target and the target plus one segment. The internal offsets log has retention forced off, because without compaction retention would silently forget committed offsets.

### Long Polling Without a Lost Wakeup

A consumer that is caught up sends a fetch with a maximum wait. The broker should park that request and wake it when data arrives, but it must not miss a produce that lands between the fetch's read and its wait. The usual bug is a wakeup sent before the waiter is waiting.

The fetch registers itself as a waiter before its first read, snapshots an epoch counter, reads, and then waits for the epoch to change. A produce increments the epoch after its append, and only if the waiter count is above zero, so producers pay nothing when no one is polling.

```cpp
if (guard.c) guard.c->fetch_add(1, std::memory_order_acq_rel);   // register first
for (;;) {
  uint64_t epoch;
  { std::lock_guard lk(data_mu_); epoch = data_epoch_; }         // snapshot before reading
  ...  // read every requested partition
  if (any || max_wait_ms == 0 || stopping_ || now >= deadline) break;
  std::unique_lock lk(data_mu_);
  data_cv_.wait_until(lk, deadline, [&] { return data_epoch_ != epoch || stopping_.load(); });
}
```

The partition mutex orders the fetch's read against the producer's append. Either the read sees the new data, or the append happened after it, in which case the producer sees the registered waiter and bumps the epoch, which the wait predicate then observes.

### Consumer Groups and Fencing

A consumer joins a group with the topics it wants. The coordinator keeps, per group, a generation number, an ordered map of members, and the current assignment. Any membership change, whether a join, a leave or a session expiry, bumps the generation and recomputes a range assignment on the broker. Each subscribed member gets a contiguous block of partitions, and the first few get one extra when the count does not divide evenly.

Kafka splits this into JoinGroup and SyncGroup<sup>[[5]](#ref-5)</sup>, with a barrier in between so that every member has given up its old partitions before anyone gets new ones. I left SyncGroup out. The cost is that during a rebalance two consumers can briefly fetch the same partition, until the old owner's next heartbeat tells it the generation changed. What keeps that from corrupting progress is fencing on commit.

```cpp
ErrorCode GroupCoordinator::check_commit(const std::string& group, const std::string& member_id,
                                         uint32_t generation) {
  if (member_id.empty()) return ErrorCode::None;  // standalone consumer, no fencing
  ...
  if (!g.members.count(member_id)) return ErrorCode::UnknownMember;
  if (generation != g.generation) return ErrorCode::IllegalGeneration;
  return ErrorCode::None;
}
```

A consumer that was rebalanced away cannot commit, so it cannot move the new owner's offset backwards or forwards. The overlap turns into duplicate delivery, never into lost or rewound progress, which is at-least-once.

Committed offsets are records in an internal partition, `__consumer_offsets-0`, keyed by group, topic and partition, with the 8-byte offset as the value. The offset store holds one mutex across the append and the map update, so the log order and the in-memory order agree, and startup replays the log into the map. That gives commits exactly the same durability path as data with no second storage engine.

### The Clients

The producer keeps one buffer per partition and hashes keys with FNV-1a<sup>[[7]](#ref-7)</sup> to choose one, or goes round robin for records without a key. The consumer fetches all of its assigned partitions in one long-poll request. Both sit on a shared client layer with one blocking connection and one request in flight, which keeps ordering trivial and caps unbatched throughput.

### Testing

31 GoogleTest cases cover the codec, the log (including torn and corrupt tails, a broken sealed index and retention across a restart), the group coordinator, and the broker over real TCP, where they check ordering, key stickiness, durability across a restart, long-poll wakeup, commit and resume, and fenced stale commits.

The concurrent ordering test is the one I trust most. Six producers, each with a different batch size, write 3,000 messages each to one partition. The test then reads the partition back and asserts that offsets are exactly 0, 1, 2 and so on with no gaps, and that each producer's messages appear in the order it sent them.

In my own runs the suite passed in a release build in 0.85 s, under ASan plus UBSan in 4.08 s, and under TSan. A smoke test runs the real broker binary and CLI, consumes with a group, restarts the broker, produces one more message, and checks that the group sees only the new one while a new group sees everything. On 2026-09-26 an independent clean release rebuild passed all 31 tests. The sanitizer and TSan builds were not rerun independently.

## Problems

### 1. The Sparse Index Was Only Sparse Between Batches

My first segment append added at most one index entry per call, at the start of the batch. Tests passed, because every lookup was still correct. It was only slow.

It showed up in the first consumer benchmark. Reading 10,000 B messages with 4 KiB fetches ran at **14.5 MB/s**, and no fetch size got the median above **82.5 MB/s**. A consumer reading 10 KB records should be the easy case, so the shape was the clue rather than the level. Larger messages were doing worse than small ones.

The benchmark fills the partition with batches of about 1 MiB, which for 10 KB messages is 104 records per produce. With one index entry per batch, a fetch for a record in the middle of a batch had to start from the batch's first record and skip forward through up to 1 MiB of headers. With 4 KiB fetches, each fetch returns one 10 KB record, so reading one batch meant 104 fetches each skipping on average half a megabyte, about 50 times more bytes read than delivered by my arithmetic. The index was sparse between batches but not within them, and with big batches that is not sparse at all. (My development notes describe this as one entry per 10 MB, but the benchmark fills 10 KB messages in batches of 104, so about 1 MiB is the right figure.)

The fix is the loop shown above, which indexes every record boundary past the 4 KiB interval, including inside a batch. A regression test appends a 1 MiB batch in one call, checks there are at least $\text{size}/4096 - 1$ entries, and checks that for offsets 1, 500 and 999 the floor position is within one interval plus one record of the target and that reading from it returns exactly that record.

After the fix the same point measured **225.3 MB/s**, and 4 MiB fetches reached **2,336.5 MB/s**. I only claim the direction from that comparison, not the ratio. The first run was at a load average of 275 to 330 on 14 cores and the second at about 9, so machine load and the fix are confounded.

### 2. Apple clang's Sanitizer Runtime Hangs

On macOS 26.5 with Apple clang 17, the ASan build deadlocked during runtime initialisation, before `main`, even for an empty program. That rules out my code. I moved the sanitizer and TSan builds to Homebrew LLVM, and the build now prints a warning if you ask for a sanitizer with the Apple toolchain. Upstream clang emits DWARF 5, which Apple's linker warns about, so non-Apple builds ask for DWARF 4 instead.

### 3. The Machine Was Not Mine

The first two full benchmark runs happened at load averages between 138 and 330 on 14 cores, with other agents' builds and benchmarks running. The numbers were low and unstable. For example, 1000 B messages at batch 1000 measured **294.5 MB/s** in the loaded run and **2,528.0 MB/s** once load dropped to about 8, more than eight times faster for the same code.

I could not get an idle machine, so I changed the harness instead. Every result row now records the 1 minute load average at the time it was measured, every throughput point is the median of 3 repetitions with the min and max kept, and I reran everything once load was around 8 to 11. The loaded runs are kept rather than deleted. One detail to watch is that the recorded environment header still shows a "load average before run" of 204.79, which was taken before the loaded run. The per-row load average is what describes the reported numbers.

Even at low load, nominally identical configurations disagree. 1000 B messages at batch 100 to one partition with no flush measured **1,001.6 MB/s** in the producer sweep, 1,187.7 MB/s in the flush sweep, and **1,340.3 MB/s** as the single-producer point of the scaling sweep. That spread of about a third is a fair estimate of how much to trust any single number in this post.

### 4. One Noisy Point Even at Low Load

In the first low-load producer run, 100 B messages at batch 100 measured **14.6 MB/s**, below batch 10, with a min to max spread of 11.7 to 38.6. A curve that goes down when batching goes up usually means Nagle's algorithm<sup>[[8]](#ref-8)</sup> holding small writes, but `TCP_NODELAY` was already set. Running the same mode again gave **233.4 MB/s** with a spread of 229.6 to 240.2. A tight spread on the rerun against a threefold spread in the original points to interference from other processes, not to my code. I kept both runs and report the second.

### 5. Slower Send Rates Had Worse Tails

Without a flush, p99 latency at 1,000 messages per second was **3.02 ms**, while at 10,000 per second it was **142 µs**. Sending less made the tail about twenty times worse.

My explanation is that at the low rate the producer sleeps between sends, and the broker's connection thread and the consumer's long-poll thread park, so every message pays for waking threads up, and probably for cores leaving low-power states. At 10,000 per second the threads stay warm. The same pattern appears in the loaded and unloaded runs. I have not confirmed it by profiling, so it is a hypothesis.

## Experiments

All experiments use a benchmark harness that starts a broker in the same process with a fresh data directory on the internal SSD and talks to it over loopback TCP through the real client code. Throughput is counted as payload bytes (the message value, not the 28 bytes of record overhead) in decimal megabytes. Throughput points are the median of 3 repetitions.

The hardware is an Apple M4 Pro with 14 cores and 48 GiB, running macOS 26.5, built with Apple clang 17 at `-O2`. The machine was running other jobs throughout. The 1 minute load average was 8 to 11 during the reported runs and is recorded per row. None of the benchmarks were rerun independently, so these are single-session, loaded-machine numbers.

1. Producer throughput against batch size (1, 10, 100 and 1,000 records) and message size (100 B, 1 KB, 10 KB), one partition, no fsync. The batch is pre-encoded, so this measures transport plus broker.
2. Consumer throughput against the fetch size limit (4 KiB to 4 MiB) and message size, reading a pre-filled partition from offset 0 and verifying every record's CRC.
3. End-to-end latency. A producer paced to a fixed schedule stamps each message with a monotonic clock, and a consumer long-polls and records receive time minus send time. Five configurations cover send rate, batch size and flush policy, and every individual sample is kept.
4. Cost of durability, with 1 KB messages at each batch size under the three flush policies.
5. Producer scaling with 1, 2, 4 and 8 concurrent producers, each to its own partition, 1 KB messages, batch 100.

## Results

### Batching Dominates Producer Throughput

<figure data-figure="chart:projects/kafka-from-scratch/kafka-from-scratch-batching"></figure>

With 100 B messages, going from batch 1 to batch 1000 raised throughput from **32,886 to 12,744,103 messages per second** (3.3 to 1,274.4 MB/s), a factor of about 390. At batch 1, every message size lands between 25,457 and 32,886 messages per second. That is the rate of one blocking round trip per connection, so at batch 1 the protocol is the limit, not the disk or the broker.

Large messages saturate earlier. 10 KB messages reach 1,311.4 MB/s at batch 10 but only **2,020.6 MB/s** at batch 1000, because a 10-record batch of 10 KB is already 100 KB per request and the fixed costs are amortised. The best point was 1 KB messages at batch 1000, **2,528.0 MB/s**, or about 2.5 million messages per second into one partition.

### Fetch Size Does the Same for Consumers

<figure data-figure="chart:projects/kafka-from-scratch/kafka-from-scratch-fetch-size"></figure>

100 B messages went from **92.8 MB/s** with 4 KiB fetches to **2,150.1 MB/s** (21.5 million messages per second) with 1 MiB fetches, and 4 MiB did not help further (2,134.9 MB/s). These numbers include a CRC check of every record on the consumer.

10 KB messages are flat at **225.3 and 219.7 MB/s** for 4 KiB and 16 KiB fetches, because each fetch returns exactly one record. The first record is always returned whole even when it exceeds the fetch size, so the fetch size stops mattering below the record size. They reach **2,336.5 MB/s** at 4 MiB. 1 KB messages are the odd ones out, topping out at 1,645.8 MB/s, with a wide min to max spread at 1 MiB (725.7 to 1,785.0), which I read as noise rather than a real knee.

### Latency

<figure data-figure="chart:projects/kafka-from-scratch/kafka-from-scratch-latency"></figure>

Without fsync, end-to-end latency is tens of microseconds at the median. At 10,000 messages per second, p50 was **41.5 µs**, p99 **142 µs** and p99.9 1.21 ms. The produce acknowledgement alone had a p50 of 32.9 µs in the same run, which suggests that most of the median is the cost of one request crossing loopback TCP and the broker, not the long poll. At 50,000 per second in batches of 10, p50 was 45.5 µs and p99 185 µs.

With `fsync` at 1,000 per second, p50 was **115 µs** and p99 586 µs. With `F_FULLFSYNC` at 200 per second, p50 was **4.03 ms** and p99 7.79 ms. That median is about 35 times the plain `fsync` median, and it is the cost of actually getting data past the drive's write cache, and it is the honest price of single-node durability on this machine.

### Durability Costs Depend on Batching

<figure data-figure="chart:projects/kafka-from-scratch/kafka-from-scratch-flush-policy"></figure>

At batch 100 with 1 KB messages, throughput was **1,187.7 MB/s** with no flush, 594.9 MB/s with `fsync` and **23.2 MB/s** with `F_FULLFSYNC`. At batch 1000 the gap narrows to 1,519.5, 1,324.1 and 197.6 MB/s, because each flush is amortised over about 1 MB. At batch 1, `F_FULLFSYNC` managed 231 messages per second, about 4.3 ms per message.

Each flush happens per append while the partition lock is held, so concurrent producers to one partition queue behind each other's flushes. Group commit<sup>[[9]](#ref-9)</sup> would fix that but needs asynchronous acknowledgements, which v0 does not have.

### Producer Scaling Is Sublinear

Aggregate throughput was **1,340.3 MB/s** with one producer, 1,490.7 with two, 1,812.9 with four and **2,061.2 with eight**, 1.54 times the single-producer rate with eight times the producers. Each producer has its own partition and its own lock, so it is not contention on the log. Plausible limits are memory bandwidth for copying through loopback TCP and the page cache, and the other load on the machine. I have not profiled it, and given the one-third spread between nominally identical runs noted in [Problems](#problems), I would not read much into the shape beyond "sublinear".

### What Load Does to All of This

For contrast, the second loaded run, at per-row load averages of 138 to 231, gave **294.5 MB/s** for 1 KB messages at batch 1000 and 382.0 MB/s at best for consumers. The first run at 275 to 330 had a p99 of **172 ms** at 1,000 messages per second without fsync, against 3.02 ms at low load. Throughput from a shared machine is a lower bound, and tail latency from a shared machine mostly measures the other jobs.

## What I Would Change

### Pipeline Producer Requests

One request in flight caps unbatched throughput at about 25,000 to 33,000 messages per second per connection. Allowing several in flight is easy for throughput and hard for ordering, since a retry of an earlier batch could land after a later one. Kafka solves that with per-partition sequence numbers checked by the broker, which is also the first step toward idempotent producers and exactly-once<sup>[[10]](#ref-10)</sup>. I would build them together.

### Add a SyncGroup Barrier

Today two consumers can briefly read the same partition during a rebalance. Generation fencing turns that into duplicate delivery rather than lost progress, but it is still a weaker guarantee than Kafka's, where the old owner revokes before the new one starts.

### An Event Loop, sendfile and Group Commit

Thread per connection kept every handler straight-line code and was fine for a handful of benchmark connections, but it will not survive thousands of clients. A kqueue<sup>[[11]](#ref-11)</sup> event loop is the right design there, and since the on-disk format already equals the wire format, fetches could then use `sendfile` and skip the copy through user space. The same asynchronous acknowledgement path would allow group-commit fsyncs across concurrent producers.

### Compact the Offsets Log

`__consumer_offsets` only grows, and startup replays all of it. Compaction, keeping only the latest record per key, is what Kafka uses to bound it<sup>[[4]](#ref-4)</sup>, and it is on the roadmap along with time-based retention.

### Profile Before Explaining

Producer scaling and the low-rate latency tail both have plausible explanations in this post that I did not verify. I would rerun them on an idle machine with sampling (Instruments on macOS) before trusting either, and I would rerun every benchmark on an idle machine before quoting any number here as more than indicative.

## Reproducibility

Requirements are CMake 3.24 or newer, Ninja and a C++20 compiler. The project was developed with Apple clang 17 on macOS 26.5 on an Apple M4 Pro, and GoogleTest is fetched by CMake. The sanitizer and ThreadSanitizer builds need Homebrew LLVM, because Apple clang 17's sanitizer runtime hangs on macOS 26.5. A standard CMake release build followed by CTest runs the release test suite in under a second.

The smoke test drives the real broker and CLI through a restart. The benchmark harness runs one mode at a time with 3 repetitions per point and records the load average alongside every result, so the numbers in this post can be regenerated, though on a different or busier machine they will differ.

Find the code at [github.com/SadeekFarhan21/projects/minikafka-log-broker](https://github.com/SadeekFarhan21/projects/tree/main/minikafka-log-broker).

## References

1. <span id="ref-1"></span>Jay Kreps, Neha Narkhede and Jun Rao. *Kafka: a Distributed Messaging System for Log Processing*. NetDB Workshop, 2011. [link](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/09/Kafka.pdf)
2. <span id="ref-2"></span>Apache Software Foundation. *Apache Kafka Documentation: Topic Configs (index.interval.bytes)*. Apache Kafka 4.3 documentation, 2026. [link](https://kafka.apache.org/43/configuration/topic-configs/)
3. <span id="ref-3"></span>Apple. *Mac OS X Manual Page For fsync(2)*. Apple Developer Documentation Archive (iOS Manual Pages), 1993. [link](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html)
4. <span id="ref-4"></span>Apache Software Foundation. *Apache Kafka Documentation: Design*. Apache Kafka 4.3 documentation, 2026. [link](https://kafka.apache.org/43/design/design/)
5. <span id="ref-5"></span>Apache Software Foundation. *Kafka Protocol Guide*. Apache Kafka 4.3 documentation, 2026. [link](https://kafka.apache.org/43/design/protocol/)
6. <span id="ref-6"></span>Guy Castagnoli, Stefan Bräuer and Martin Herrmann. *Optimization of Cyclic Redundancy-Check Codes with 24 and 32 Parity Bits*. IEEE Transactions on Communications 41(6), 1993. [doi:10.1109/26.231911](https://doi.org/10.1109/26.231911)
7. <span id="ref-7"></span>Glenn Fowler, Landon Curt Noll and Phong Vo. *FNV Hash*. Landon Curt Noll (isthe.com), 1991. [link](http://www.isthe.com/chongo/tech/comp/fnv/index.html)
8. <span id="ref-8"></span>John Nagle. *Congestion Control in IP/TCP Internetworks*. RFC 896, 1984. [link](https://www.rfc-editor.org/rfc/rfc896.html)
9. <span id="ref-9"></span>David J. DeWitt, Randy H. Katz, Frank Olken et al. *Implementation Techniques for Main Memory Database Systems*. SIGMOD, 1984. [doi:10.1145/602259.602261](https://doi.org/10.1145/602259.602261)
10. <span id="ref-10"></span>Jason Gustafson, Flavio Paiva Junqueira, Apurva Mehta et al. *KIP-98: Exactly Once Delivery and Transactional Messaging*. Apache Kafka Improvement Proposal, 2017. [link](https://cwiki.apache.org/confluence/display/KAFKA/KIP-98+-+Exactly+Once+Delivery+and+Transactional+Messaging)
11. <span id="ref-11"></span>Jonathan Lemon. *Kqueue: A Generic and Scalable Event Notification Facility*. USENIX Annual Technical Conference, FREENIX Track, 2001. [link](https://www.usenix.org/legacy/publications/library/proceedings/usenix01/freenix01/lemon.html)
