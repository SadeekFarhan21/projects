---
layout: post
title: "A Crash-Safe B+ Tree That Survives kill -9"
tags:
  - databases
  - storage
  - cpp
description: >-
  A B+ tree key-value store in C++ with a write-ahead log and journaled
  checkpoints. It survived kills aimed at every checkpoint step without losing
  an acknowledged write.
date: 2026-09-29 02:17:08
---


I wrote kvdb, a persistent single-node key-value store in C++20 with no runtime dependencies. It keeps a B+ tree in 4 KiB pages in one file, caches pages in a buffer pool with LRU eviction and pin counts, logs every commit to a write-ahead log (WAL) with a configurable sync policy, makes checkpoints crash-atomic with a page journal, and recovers after a crash by replaying the log. A small REPL sits on top.

The code is about 1,700 lines of C++ in `src/` and `include/`, headers and comments included, plus about 930 lines of GoogleTest. There are **36 tests**, and on 2026-09-27 an independent clean release rebuild passed all 36, including the tests that fork a writer process and kill it. The AddressSanitizer plus UBSan build also passed all 36 in my own run, but it was not rerun independently, and neither were the benchmarks.

The headline result is the crash testing. A child process writes and acknowledges each commit over a pipe, and the parent kills it with SIGKILL. In the verified release run, **12 random kills covering 9,721 acknowledged operations lost none of them**. A second test aims kills into the middle of a large checkpoint, and **6 of its 10 killed rounds** landed in the window where the journal was complete but the pages were only partly written in place, and were repaired from the journal. A third test crashes the process at each of **5 named steps** of the checkpoint protocol and then crashes it again at the same step inside recovery. Every one of these reopens the database, walks the whole tree checking its invariants, and compares it to a model. These are process crashes. The OS page cache survives them, so they say nothing about power loss, which I did not test.

The second story is what durability costs on a Mac. With no sync a commit took **1.2 µs** at the median. With `fsync` it took **24 µs**, and with `F_FULLFSYNC`, the only call on macOS that makes the drive flush its own cache, it took **4.01 ms**, about 3,300 times the unsynced cost. Batching 256 puts into one `F_FULLFSYNC` commit brought throughput from 243 to **59,450 puts per second**. All benchmark numbers come from a shared Apple M4 Pro that was running other jobs, with the 1 minute load average recorded in every row (3.9 to 6.3 for the reported runs). They are indicative, not a clean benchmark. The 1M key database also fits in memory, so the throughput numbers measure CPU and system call cost, not the disk.

Code is in `projects/kvdb-btree-wal-engine`. The argument of this post is that a small storage engine becomes trustworthy through its crash tests rather than its design document, and that on macOS "durable" is a choice between three very different prices. Skip to [problems](#problems) for what went wrong.

## table of contents

- [what I wanted to build](#what-i-wanted-to-build)
- [theory](#theory)
- [architecture](#architecture)
- [implementation](#implementation)
- [problems](#problems)
- [experiments](#experiments)
- [results](#results)
- [what I would change](#what-i-would-change)
- [reproducibility](#reproducibility)

## what I wanted to build

I wanted a storage engine I understand all the way down, from bytes in a 4 KiB page, to a page in a buffer frame, to a frame pinned by a tree operation, to a commit in a log record, to a crash that loses nothing it promised to keep.

The v0 scope was fixed up front. A pager over one file. A buffer pool with LRU eviction and pin counts. A B+ tree with insert, point lookup, range scan and lazy delete. A WAL with fsync and redo recovery on startup. A REPL. A crash test that really kills a process. And benchmarks against `std::map` and SQLite at 1M keys.

The bar was correctness first. Every claim about durability had to be backed by a test that kills a process with SIGKILL and checks what survives against a model, and every performance number had to come from a run saved under `results/`. Transactions, concurrency, page checksums, real delete and an LSM variant are later milestones and are not in this post.

## theory

### B+ trees on disk

A disk-based B+ tree stores sorted keys in fixed-size pages. Internal pages hold separator keys and child pointers. Leaves hold the data and are linked left to right, so a range scan is one descent followed by a walk along the leaf chain. The point of the shape is fanout. In kvdb an internal cell for a 16-byte key is 24 bytes plus a 2-byte slot, so a 4 KiB page holds about 150 children, and 1M keys fit in a tree of height 4. A lookup touches four pages and does a binary search inside each.

Splits decide how full the leaves are. A textbook split cuts a full node in half, and under random inserts leaves settle at about 69% full (ln 2). Under ascending inserts the left half is never touched again and stays half empty, so the append pattern deserves a special case.

### the write-ahead rule

Durability rests on one rule. A change is committed once a description of it is on stable storage, and that description must get there before the data pages change on disk. After a crash, the committed state is rebuilt from the last consistent page file plus the log.

The general design is ARIES, which is STEAL (a dirty page may be written back before its transaction commits) and NO-FORCE (pages need not be written at commit). It is flexible and it needs both undo and redo logging, page LSNs to know which changes a page already contains, and a three-pass recovery.

I took a simpler corner of the design space. kvdb's buffer pool is NO-STEAL, meaning dirty pages never reach the page file except at a checkpoint. The page file on disk is therefore always exactly the state as of the last checkpoint, and the log only needs logical redo records ("put k v"). There is no undo, no page LSN, and recovery is "load the snapshot, redo the operations after it".

### making the checkpoint atomic

The remaining problem is the checkpoint itself. It overwrites many pages in place, and a crash halfway leaves a parent from the new generation pointing at a child from the old one, which is a corrupt tree. The fix is a double write. Copy every page the checkpoint will overwrite to a side file, sync it, and only then write in place. A crash before the journal is complete leaves the page file untouched. A crash after it leaves a complete journal that recovery replays. SQLite's rollback journal and InnoDB's doublewrite buffer do versions of the same thing.

The subtle part is deciding whether a journal is complete. Writing the header last is not enough, because without a barrier between them the header can reach the disk before the body. So kvdb accepts a journal only if the magic matches, the file size equals exactly the size the header's page count implies, and a CRC over the entire body matches. A journal cut short at any byte fails one of those checks.

### three durability levels on macOS

On macOS, `fsync()` pushes data to the drive but does not make the drive flush its volatile write cache. Only `fcntl(fd, F_FULLFSYNC)` does that. So there are three levels, and kvdb exposes all of them as `SyncMode`.

| Mode | What it survives | Call after each WAL append |
|---|---|---|
| `kNone` | a process crash | none |
| `kFsync` (default) | a process or OS crash | `fsync` |
| `kFullFsync` | power loss, if the drive honors the flush | `fcntl(F_FULLFSYNC)` |

`kNone` survives a process crash because a `pwrite` that has returned is in the kernel's page cache, and the kernel writes it out whether or not the process is still alive. That matters for reading the crash tests correctly, as I explain below.

## architecture

A database at path `P` is three files. `P` is the page file, where page 0 is a meta page and every other page is a B+ tree node. `P-wal` holds one CRC-framed record per committed batch since the last checkpoint. `P-journal` is empty except while a checkpoint runs.

<figure data-figure="diagram:kvdb-write-and-recovery"></figure>

The meta page holds the magic number, format version, root page id, page count, tree height and checkpoint LSN. It lives in the buffer pool like any node, so a root split or a page allocation is checkpointed atomically with the nodes it refers to. There is no separate "superblock write" to get wrong.

Nodes are slotted pages. A 16-byte header is followed by an array of 2-byte slot offsets sorted by key, growing toward the end of the page, while cells (`klen, vlen, key, value`) grow backward from the end.

<figure data-figure="diagram:kvdb-slotted-page"></figure>

A leaf's `link` is its right sibling. An internal node's `link` is its leftmost child, and each cell's value is a 4-byte child id. Keys are capped at 128 bytes and values at 512, which makes a worst-case cell plus slot 646 bytes, so any overflowing node holds at least 6 cells and a byte-balanced split always produces two halves that fit. That cap is what lets v0 skip overflow pages.

A WAL record is `u32 crc32 | u32 len | u64 lsn | payload`, and the payload is the `WriteBatch` encoding itself, so logging a batch is one copy and one `pwrite`. The journal is `u64 magic | u32 page count | u32 crc32(body)` followed by `(u32 page id, 4096 bytes)` per page.

Three invariants carry the design. Between checkpoints the page file is byte-identical to the last completed checkpoint. A batch is committed if and only if its WAL record is intact on disk. And replaying a batch over a page file that already contains some or all of it gives the same result, because putting or deleting a whole key is idempotent.

## implementation

### the write path

A write is four lines, and the order is the whole durability argument.

```cpp
void DB::write(const WriteBatch& batch) {
  if (batch.count() == 0) return;
  maybe_checkpoint(true);
  const uint64_t lsn = next_lsn_++;
  wal_->append(lsn, batch.rep(), opts_.sync);  // commit point
  apply(batch.rep());
  applied_lsn_ = lsn;
  ++stats_.commits;
}
```

`put` and `del` are one-op batches, and deleting a missing key is not logged at all. `Wal::append` frames the record, computes the CRC with the ARMv8 CRC32 instructions (8 bytes per instruction, with a table fallback that a test checks agrees), issues a single `pwrite`, and syncs according to the mode. The WAL file is grown in 4 MiB `ftruncate` steps, so most appends land inside the file and the file has a zero-filled tail that replay must treat like garbage.

### replay stops at the first thing it does not trust

```cpp
while (off + kHeaderSize <= data.size()) {
  const uint32_t crc = load<uint32_t>(data.data() + off);
  const uint32_t len = load<uint32_t>(data.data() + off + 4);
  if (len > data.size() - off - kHeaderSize) break;          // torn tail
  if (crc32(data.data() + off + 4, 12 + len) != crc) break;  // corrupt
  const uint64_t lsn = load<uint64_t>(data.data() + off + 8);
  if (lsn <= last_lsn) break;                                // stale bytes
  fn(lsn, std::string_view(data.data() + off + kHeaderSize, len));
  last_lsn = lsn;
  off += kHeaderSize + len;
}
```

A torn write, a flipped byte and the preallocated zero tail all end replay the same way, and the file is then truncated at the last good record so new records are never appended after garbage. The WAL tests cut the log at every possible byte offset of the last record and check that replay keeps exactly the records before it.

### the buffer pool

The pool is one page-aligned allocation split into frames, an `unordered_map` page table, a free list and a `std::list` LRU. The key design choice is that the LRU list contains only frames that are both unpinned and clean. Eviction is then `lru_.back()`, and it structurally cannot pick a pinned or dirty page. If every frame is pinned or dirty, `fetch` throws rather than corrupting anything. `PageGuard` is an RAII pin that carries the dirty bit to `unpin`, so a tree operation cannot forget to release a page.

Because the pool is NO-STEAL, dirty pages accumulate until a checkpoint. The DB layer checkpoints when the WAL passes 64 MiB, at clean close, at the end of recovery, and whenever the dirty count comes within 64 frames of capacity. That last trigger is where NO-STEAL shows its cost, as the small-pool benchmark shows.

### the append split

On overflow the tree materializes the node's entries plus the new one and splits by bytes rather than by count, since cells vary in size. One special case does a lot of work.

```cpp
if (idx == n - 1 && node::link(leaf.data()) == kNoPage) {
  // Appending past the end of the rightmost leaf (ascending inserts): leave
  // the old leaf full and start a new one. Gives ~100% fill instead of 50%.
  m = n - 1;
} else {
  m = std::clamp(byte_midpoint(ents), 1, n - 1);
}
```

The first key of the right half is pushed into the parent, and internal nodes split the same way, except the middle key moves up rather than being copied. `BTree::verify()` walks the whole tree checking each node's format, every key against the range its parent's separators allow, that all leaves are at the same depth, and that the sibling chain visits them in order. Every crash test calls it after reopening.

### the checkpoint

```cpp
tree_->set_checkpoint_lsn(applied_lsn_);
const auto pages = pool_->dirty_pages();        // sorted by page id, meta included
// stream (id, page) entries to the journal body in 1 MiB chunks, CRC as we go
failpoint::hit("ckpt.before_journal_header");
pwrite_all(journal_fd_, hdr, kJournalHeader, 0);
sync_fd(journal_fd_, mode);
failpoint::hit("ckpt.journal_synced");
for (size_t i = 0; i < pages.size(); ++i) {
  pager_->write_page(pages[i].first, pages[i].second);
  if (i == pages.size() / 2) failpoint::hit("ckpt.mid_data");
}
pager_->sync(mode);
failpoint::hit("ckpt.data_synced");
// then truncate the WAL (failpoint ckpt.wal_reset) and the journal
```

The `failpoint::hit` calls are no-ops unless a test has armed that name, in which case the process calls `_exit(77)` on the spot, with no destructors and no flushing.

Recovery is the mirror image. If the journal is valid, every page in it is written to its home location and synced, then the journal is truncated. If it is not valid, the crash happened before any in-place write, so the page file is still the previous checkpoint and the journal is simply discarded.

### batches bigger than the pool

One case needed care. A single batch can dirty more pages than the pool holds. `apply` then checkpoints in the middle of the batch, but without truncating the WAL and without moving the checkpoint LSN past the previous batch. After a crash, recovery redoes the whole batch on top of a page file that already contains part of it, which is safe because whole-key puts and deletes are idempotent.

### crash tests with real processes

The random-kill test forks a child that opens the database with a 256-page pool and a 64 KiB WAL threshold, so checkpoints happen constantly, and runs a deterministic stream of operations. Most are single puts, about one in five is a delete, and every 25th is a 20-op batch, to check that batches are atomic. After each `write()` returns, the child writes the op index to a pipe.

```cpp
while (seen < kill_after && ::read(fds[0], &ack, sizeof ack) == sizeof ack) {
  last = ack;
  ++seen;
}
::kill(pid, SIGKILL);
// Drain acks the child managed to send before dying: those were committed.
while (::read(fds[0], &ack, sizeof ack) == sizeof ack) last = ack;
```

The parent then reopens the database, runs `verify()`, and requires the contents to equal a `std::map` model after every acknowledged op, or after exactly one more. The extra one is the op that may have reached the WAL between `write()` returning and the ack being sent. Anything else fails, including a lost acknowledged op, a resurrected delete and a half-applied batch. The test runs 12 rounds on the same file, so each round also recovers on top of the previous recovery.

The `flock` that stops two processes opening the same database dies with the process, which is what lets the parent reopen immediately after a kill.

## problems

### 1. the random kills never hit a checkpoint

The random-kill test was supposed to be the checkpoint test too. It was not. Checkpoints are short compared with the time between them, so a kill at a random moment almost always lands in the write path. In the final release run, **0 of 12 rounds** found a journal to replay. The test was passing without ever exercising the journal.

I found this because the test prints how many rounds recovered from a journal, and the number was zero. I added two tests to cover the gap. `KillNineDuringLargeCheckpoint` builds a checkpoint big enough to aim at (6,000 mixed ops plus 50,000 new keys, with an 8,192-page pool), runs one uncounted calibration round that reports how long the journal and data phases take, and then kills 10 rounds at random delays between halfway through the journal write and the end of the checkpoint. In the release run the calibration checkpoint took 6.7 ms, 1.4 ms of it writing the journal and 5.4 ms writing pages in place and truncating. **6 of the 10 kills** landed after the journal was complete and were repaired from it. The rest landed before it was complete and were correctly discarded.

That hit rate depends on timing. Under ASan the journal phase took 39.4 ms instead of 1.4 ms, and only 2 of 10 kills landed in the repair window. So the aimed test is probabilistic, and I did not want the protocol's correctness to rest on luck. The deterministic `CheckpointCrash` tests arm a failpoint at each of five steps (before the journal header, after the journal sync, halfway through the in-place writes, after the data sync, after the WAL reset), crash there, then arm the same failpoint and open the database, so recovery's own checkpoint crashes at the same step. A third open must produce exactly the model. All five pass in both builds.

### 2. SQLite's full sync was faster than a full sync

With `synchronous=FULL` and `fullfsync=ON`, SQLite committed **1,006 times per second** against kvdb's **247** with `F_FULLFSYNC`. That looked like kvdb doing something wrong, until I compared latencies. A real `F_FULLFSYNC` on this machine takes about 4 ms, in both kvdb and a bare `pwrite` loop, while SQLite's median commit took **0.69 ms**. SQLite cannot be issuing a full flush on every commit.

My hypothesis is that Apple's system SQLite, which is the copy the benchmark links, uses the lighter `F_BARRIERFSYNC` for WAL commits, which orders writes without waiting for the cache flush. I have not verified it, with `fs_usage` or by building SQLite from source. Until I do, the `F_FULLFSYNC` column of the durability comparison is not like for like, and I mark it that way in the figure below.

### 3. a comment claimed a 14x speedup nobody had measured

A comment in `wal.h` said that on APFS an append which extends the file costs about 14 times more than a write inside a preallocated file. I had not measured that, so I wrote the `walgrow` experiment. It did not hold up. Preallocation made a 128-byte append **1.70 times** faster with no sync, **1.09 times** faster with `fsync`, and made no difference with `F_FULLFSYNC` (247 versus 248 appends per second). I kept preallocation, since it costs nothing and replay already handles a garbage tail, and corrected the comment to cite the file.

### 4. the machine was busy

The machine was shared with a dozen other builds. An earlier benchmark run, later interrupted, happened at a 1 minute load average of about 256 on 14 cores, and its wall-clock numbers were roughly an order of magnitude below what I measured later at a load of 4 to 6. I threw it away. Every benchmark row now records ops per CPU second and the load average, the build script uses `-j2`, and I ran one benchmark or test binary at a time. Even so, `results/machine.txt` shows a 15 minute load average of 42 when the reported run started, so the machine had been very busy shortly before.

### 5. Apple clang's sanitizer runtime hung

An early UBSan-only run under Apple clang 17 stalled inside the large-checkpoint crash test and never finished, and the sanitizer runtime hung at startup on this macOS version more generally. I moved the sanitizer preset to Homebrew LLVM 23, which runs ASan and UBSan together without trouble. Switching compilers then surfaced two smaller problems. Three files used `errno` without including `<cerrno>` and had only compiled because an Apple header pulled it in transitively. And Homebrew LLVM's DWARF 5 debug info made Apple's linker print a harmless warning for every object file, burying real warnings, so the sanitizer build now passes `-gdwarf-4` for upstream Clang.

## experiments

All runs were on an Apple M4 Pro (14 cores, 48 GB) running macOS 26 (Darwin 25.5.0), release build at `-O2`, with the 1 minute load average recorded next to each row. Keys are 16 bytes (`user%012d`, so lexicographic order is numeric order) and values are 100 bytes.

1. **Correctness.** The 36 tests in 8 suites, in a release build and in an ASan plus UBSan build. The release suite was rerun independently on 2026-09-27 and passed. The sanitizer suite was not rerun.
2. **Throughput and latency at 1M keys**, 3 repetitions, medians reported. For each engine, insert 1M keys in sequential or shuffled order, then read all 1M in sequential and shuffled order, then scan everything. Every operation is timed individually with `steady_clock`, so p50, p99 and p99.9 are exact. The engines are `std::map<std::string, std::string>`, kvdb with a 65,536-frame (256 MiB) pool, kvdb with a 4,096-frame (16 MiB) pool, and SQLite 3.51.0 with `journal_mode=WAL`, `synchronous=OFF`, a 256 MiB cache, prepared statements and a `WITHOUT ROWID` table. kvdb ran with `SyncMode::kNone`, so both disk engines write their log on every commit and neither syncs.
3. **Durability cost.** 2,000 random puts, one per commit, under each sync mode for kvdb and SQLite, then kvdb with `F_FULLFSYNC` and 1, 4, 16, 64 or 256 puts per `WriteBatch`. One run each.
4. **WAL preallocation.** 3,000 raw 128-byte `pwrite` appends, each followed by no sync, `fsync` or `F_FULLFSYNC`, into a file that either grows with each append or was sized in advance.

What the benchmarks do not measure matters as much. With the 256 MiB pool, the whole database is resident. The pool hit rate was 1.000 for both fills, and the page file was 122.1 MiB after the sequential fill and 170.9 MiB after the random one. Even the 16 MiB pool misses only into the OS page cache, since the machine has 48 GB and the file was just written. **No benchmark here reads from the SSD.** The throughput numbers are the cost of CPU work and system calls, and the only experiment where the storage device is on the critical path is the durability one.

## results

### crash safety

| Test | What it does | Release result |
|---|---|---|
| `KillNineDuringWritesLosesNothingAcknowledged` | 12 random SIGKILLs during writes, `kNone` | 9,721 acknowledged ops, none lost, 2,254 WAL records redone |
| `KillNineWithFsyncCommits` | 3 SIGKILLs after 300, 500 and 700 acks, `kFsync` | passed |
| `KillNineDuringLargeCheckpoint` | 10 SIGKILLs aimed into a 6.7 ms checkpoint | all 10 exact, 6 repaired from a complete journal |
| `CheckpointCrash`, 5 steps | crash at a step, then again at that step in recovery | all 5 exact |
| `CrashWithoutCloseThenTornWalTail` | exit without close, chop 7 bytes off the WAL, append junk | 1,999 of 2,000 records recovered, the torn one dropped |

All 36 tests passed in 2.5 s in the release build (`results/tests_release.txt`) and in 34.7 s under ASan plus UBSan (`results/tests_asan.txt`). The sanitizer run's random-kill test covered 9,693 acknowledged ops with 2,212 records redone, the different count coming from different kill timing, and it also lost none.

It is worth being exact about the scope. The random-kill test runs with `SyncMode::kNone`, and with `kNone` the checkpoint's journal "sync" is a no-op too. That is fine for a process crash, where every returned `pwrite` is already in the kernel's page cache and ordering between files is preserved by that cache. It would not be fine for power loss, where unsynced data can vanish and writes can reach the disk out of order. None of these tests pulls the plug, so the claim is that kvdb loses no acknowledged write to a process crash at any point I could aim at, including inside checkpoints and inside recovery. The protocol is designed for power loss when run with `kFullFsync`, but that is a design argument, not a test result.

### what durability costs

<figure data-figure="chart:projects/kvdb-btree-wal-engine/kvdb-btree-wal-engine-durability"></figure>

| Mode | kvdb commits/s | kvdb p50 | SQLite commits/s | SQLite p50 |
|---|---|---|---|---|
| no sync | 693,922 | 1.2 µs | 70,517 | 10.3 µs |
| fsync | 38,162 | 24.0 µs | 4,900 | 67.5 µs |
| F_FULLFSYNC | 247 | 4.01 ms | 1,006 (see problem 2) | 0.69 ms |

Each step costs one to two orders of magnitude. Going from no sync to `fsync` cut kvdb's commit rate by a factor of 18, and going from `fsync` to `F_FULLFSYNC` cut it by another 154. At the median, a truly durable single-put commit on this Mac costs about 4 ms, which is about 3,300 times the unsynced commit.

The `fsync` row is the uncomfortable one. It is the default in kvdb, as it is in SQLite on macOS, and it costs 24 µs, which feels like durability. It is not durability against power loss, because the data may still be sitting in the drive's cache. The honest price of that guarantee is the 4 ms row.

### batching is the only way to pay less

<figure data-figure="chart:projects/kvdb-btree-wal-engine/kvdb-btree-wal-engine-batching"></figure>

With `F_FULLFSYNC` on every commit, putting more operations in each `WriteBatch` scales almost perfectly. Throughput was 243, 980, 3,960, 15,485 and 59,450 puts per second at 1, 4, 16, 64 and 256 puts per batch, while the median commit stayed at 4.00 to 4.05 ms throughout. The flush is the cost and it is paid once per batch, so the 256-put batch is about 245 times faster than single puts. A `WriteBatch` is one WAL record, one `pwrite` and one sync, and it is atomic, which the random-kill test checks with its 20-op batches.

This is group commit done by hand, because v0 is single-threaded. Letting concurrent committers share one flush automatically is on the v2 list.

### throughput when everything is in memory

<figure data-figure="chart:projects/kvdb-btree-wal-engine/kvdb-btree-wal-engine-throughput"></figure>

| Workload | `std::map` | kvdb 256 MiB | kvdb 16 MiB | SQLite |
|---|---|---|---|---|
| sequential put, ops/s | 3,852,003 | 728,263 | 719,354 | 94,221 |
| random put, ops/s | 1,098,039 | 439,183 | 206,867 | 49,851 |
| random get after random put, ops/s | 840,719 | 1,251,868 | 639,082 | 170,354 |
| sequential get after sequential put, ops/s | 6,986,963 | 3,668,542 | 3,364,736 | 253,240 |
| random put p50 / p99, ns | 875 / 1,791 | 1,833 / 5,083 | 2,583 / 5,708 | 10,667 / 74,041 |
| random get p50 / p99, ns | 1,042 / 2,125 | 750 / 1,250 | 1,542 / 2,209 | 5,792 / 7,875 |
| full scan after random put, keys/s | 11,357,624 | 60,934,431 | 20,973,686 | 13,935,243 |

These are medians of 3 runs from `results/summary_main.csv`. Given that none of it touches the SSD, I read them as follows.

### kvdb beats a red-black tree on reads

Random gets ran at 1.25M per second in kvdb against 0.84M for `std::map`, with a p50 of 750 ns against 1,042 ns. A red-black tree filled in random order scatters 1M nodes across the heap and follows about 20 pointers per lookup, many of them likely cache misses. The B+ tree touches 4 pages and binary-searches inside each. I did not profile this, so the explanation is a plausible one rather than a measured one. The full scan shows the same effect more strongly, 61M keys per second for kvdb against 11M for `std::map` after a random fill, because a leaf chain is sequential memory and a red-black tree walk is not.

### kvdb loses to a red-black tree on writes

Sequential puts were 5.3 times slower than `std::map` and random puts 2.5 times slower. Each put encodes a batch, computes a CRC, issues one `pwrite` system call to the WAL and may split nodes. The system call per put is the obvious target, and it is the first thing on the change list below.

### the gap to SQLite is mostly generality

kvdb's random puts were 8.8 times faster than SQLite's and its random gets 7.3 times faster. I would not read that as kvdb being a better storage engine. SQLite here runs prepared statements through its bytecode VM, encodes records, and maintains a far more general B-tree, while kvdb has a specialized key-value path. It is a comparison of a narrow tool against a general one, both in memory.

### the append split works

Sequential fill produced 30,304 leaves at 99% fill. Random fill produced 43,324 leaves at 69% fill, which is the textbook ln 2 value for random inserts. Both trees have height 4, and the random-fill file is 40% larger for the same data.

### NO-STEAL turns memory pressure into checkpoints

With a 16 MiB pool, random fill needed **213 checkpoints** against 3 for the 256 MiB pool, and ran 2.1 times slower (207K versus 439K puts per second, pool hit rate 0.832). Random inserts dirty leaves all over the tree, and a NO-STEAL pool cannot write any of them back without a full checkpoint. Sequential fill barely noticed (9 checkpoints, 719K versus 728K per second, hit rate 0.988), because it only ever dirties the rightmost path.

### WAL preallocation is a small win

| Sync | growing file, appends/s | preallocated, appends/s | ratio |
|---|---|---|---|
| none | 601,253 | 1,021,581 | 1.70 |
| fsync | 34,394 | 37,525 | 1.09 |
| F_FULLFSYNC | 247 | 248 | 1.00 |

Whatever extending the file costs on APFS, it shows up when nothing else is expensive and disappears under a 4 ms flush.

## what I would change

### make the pool STEAL and the log physiological

NO-STEAL made recovery easy to reason about, and the crash tests are simple partly because of it. But the small-pool run shows that it converts memory pressure directly into checkpoints, 213 of them for 1M random puts, and it rules out transactions larger than memory. With concurrent transactions coming in v1 and v2, the right design is page LSNs and physiological log records, which let the pool evict dirty pages and let checkpoints be fuzzy.

### checksum every page

Today a torn or bit-flipped page in the data file would be read silently. The WAL and the journal carry CRCs, and pages should too, with torn-page detection at read time.

### buffer the log instead of a pwrite per put

The write path is where kvdb loses most to `std::map`, and one system call per put is the obvious cause. An in-memory log buffer flushed at commit boundaries would keep the same durability semantics for synced modes and remove most of the system calls for `kNone`.

### real delete

Lazy delete was the right cut for v0, since lookups and scans stay correct and deleting everything then reinserting works. But space is never reclaimed, so a delete-heavy workload wastes space without bound. Merge, redistribution and a free-page list are in v3.

### settle the SQLite question and test power loss

I would verify SQLite's sync behavior with `fs_usage` or a source build, so the durability comparison is like for like. And the crash tests should gain a mode that simulates power loss, for example by interposing on writes and dropping everything after the last completed sync, since the process-crash tests cannot say anything about that case.

### benchmark larger than memory, on a quiet machine

Every throughput number here is an in-memory number. A database several times larger than RAM, with the page cache dropped, would make the buffer pool go to the SSD and make the random-read results reflect I/O. It should run on a machine with nothing else on it.

## reproducibility

Build. This needs CMake 3.24 or newer, Ninja and a C++20 compiler, and GoogleTest is fetched at configure time. SQLite is optional, and the macOS SDK's copy is found automatically and enables the SQLite baseline. The `asan` preset uses Homebrew LLVM at `/opt/homebrew/opt/llvm/bin/clang++` because Apple clang 17's sanitizer runtime hangs on this macOS version.

```sh
cd projects/kvdb-btree-wal-engine
cmake --preset release            # -O2, build/release
cmake --build --preset release -j2
cmake --preset asan               # ASan + UBSan, build/asan
cmake --build --preset asan -j2
```

Tests.

```sh
./build/release/kvdb_tests        # 36 tests, a few seconds
./build/asan/kvdb_tests           # the same 36 under ASan + UBSan
ctest --preset release            # or through CTest
```

Benchmarks and plots. The script writes every raw CSV and log to `results/`, with the load average in each row, and takes about 5 minutes. The plotting script needs matplotlib.

```sh
scripts/run_bench.sh 1000000 3
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/plot_results.py      # results/summary_main.csv and PNGs
./build/release/kvdb_bench --exp=sync --n=2000 --out=/tmp/sync.csv   # one experiment
```

Code is in `projects/kvdb-btree-wal-engine`, with the public API in `include/kvdb/db.h`, the engine in `src/`, the crash tests in `tests/recovery_test.cc`, the benchmark harness in `bench/bench.cc`, the formats, invariants and trade-offs in `DESIGN.md`, and the build log in `DEVLOG.md`.
