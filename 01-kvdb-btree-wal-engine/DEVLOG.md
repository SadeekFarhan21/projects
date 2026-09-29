# DEVLOG: building a database from scratch (v0)

### What I wanted to build

I wanted a storage engine I understand all the way down: bytes in a 4 KiB page, a page in a buffer frame, a frame pinned by a tree operation, a commit in a log record, and a crash that loses nothing it promised to keep. The target for v0 was a persistent single-node key-value store with a pager over one file, a buffer pool with LRU eviction and pin counts, a B+ tree with insert, point lookup, range scan and lazy delete, a write-ahead log with fsync, redo recovery on startup, a REPL, a crash test that actually kills the process, and benchmarks against `std::map` and SQLite at 1M keys.

The bar I set was correctness first. Every claim about durability had to be backed by a test that kills a process with SIGKILL and checks the survivor against a model, and every performance number had to come from a run saved under `results/`.

### Theory

A disk-based B+ tree stores sorted keys in fixed-size pages. Internal pages hold separator keys and child pointers; leaves hold the data and are linked left to right so a range scan is a descent followed by a walk. With 16-byte keys a 4 KiB internal page has a fanout of about 150, so 1M keys fit in a tree of height 4.

Durability comes from the write-ahead rule: a change is committed once a description of it is on stable storage, before the data pages change on disk. After a crash you rebuild the committed state from the last consistent page file plus the log. The classic general design is ARIES: STEAL (dirty pages may be written before commit) and NO-FORCE (pages need not be written at commit), which requires both undo and redo logging and page LSNs.

I took a simpler corner of the design space. With NO-STEAL, dirty pages never reach the page file except at a checkpoint, so the page file is always the state as of the last checkpoint, and the log only needs logical redo records ("put k v"). The remaining problem is making the checkpoint itself atomic, since it overwrites many pages in place and a crash in the middle would leave parents and children from different generations. I solved that with a journal (a double write): copy every page to a side file, sync it, then write in place. This is close to what SQLite's rollback journal and InnoDB's doublewrite buffer do.

On macOS there is one more piece of theory that matters: `fsync()` pushes data to the drive but does not force the drive to flush its own cache. Only `fcntl(F_FULLFSYNC)` does that. So there are really three durability levels, and I exposed all three (`SyncMode::kNone`, `kFsync`, `kFullFsync`).

### Architecture

```
  put/del/write(batch) --> DB --> Wal::append + sync (commit point)
                            |
                            +--> BTree --> BufferPool (LRU, pins, NO-STEAL) --> Pager --> data file
                            |
                            +--> checkpoint: dirty pages --> journal (sync) --> data file (sync)
                                             --> truncate WAL and journal

  open: journal complete? replay it : discard it --> load meta --> redo WAL above checkpoint LSN
```

Three files per database: the page file, `-wal`, and `-journal`. Page 0 is a meta page (magic, root, page count, height, checkpoint LSN) that lives in the buffer pool like any node, so it is checkpointed atomically with the tree. DESIGN.md has the full diagram, formats and invariants.

### Implementation

The code is about 1,500 lines of C++20 in `src/` and `include/`, plus about 1,000 lines of tests.

The pager (`src/pager.cc`) is `pread`/`pwrite` of whole pages. It takes an `flock` on the file so a second process cannot open the same database; the lock dies with the process, which the kill tests rely on.

The buffer pool (`src/buffer_pool.cc`) is one page-aligned allocation split into frames, a page table (`unordered_map`), a free list and a `std::list` LRU. Only frames that are unpinned and clean are in the LRU, so eviction is `lru_.back()` and can never pick a pinned or dirty page. `PageGuard` is an RAII pin that carries the dirty bit to `unpin`. If every frame is pinned or dirty, `fetch` throws instead of silently corrupting something; the DB layer prevents that by checkpointing when dirty pages come within 64 frames of capacity.

Nodes (`src/node.cc`) are slotted pages: a 16-byte header, a sorted array of u16 slot offsets growing down, and cells (`klen, vlen, key, value`) growing up from the end. Erase leaves a hole and adds to a fragmentation counter; an insert that fits only after reclaiming holes compacts the page first.

The B+ tree (`src/btree.cc`) descends recording the path, then updates or inserts in the leaf. On overflow it materializes the entries, splits by bytes rather than by count (cells vary in size), and pushes the separator into the parent, splitting upward as needed. One special case matters a lot: when the new key lands at the end of the rightmost leaf, the old leaf stays full and the new right leaf starts nearly empty. Sequential loads then produce 99% full leaves instead of 50%. `BTree::verify()` walks the whole tree checking node invariants, key ranges against parent separators, leaf depth and the sibling chain; tests call it after every crash.

The WAL (`src/wal.cc`) frames each `WriteBatch` as `crc32 | len | lsn | payload`. The CRC uses the ARMv8 CRC32 instructions (8 bytes per instruction) with a table-driven fallback, and a test checks they agree. The file is preallocated in 4 MiB steps with `ftruncate`, so replay must, and does, treat a zero-filled tail like a torn one.

`DB` (`src/db.cc`) ties it together. A write assigns an LSN, appends and syncs the record, then applies it. A checkpoint stamps the meta page with the last fully applied LSN, streams the dirty pages to the journal body, writes the header (count and CRC of the body) last, syncs, writes the pages in place, syncs, then truncates the WAL and the journal. Recovery replays a complete journal, loads the tree, redoes WAL records above the checkpoint LSN and checkpoints.

Batches larger than the buffer pool needed care. If applying one batch dirties more pages than the pool can hold, `apply` checkpoints in the middle without truncating the WAL and without advancing the checkpoint LSN past the previous batch. After a crash the whole batch is redone on top of a page file that already contains part of it, which is safe because putting or deleting a whole key is idempotent.

Crash testing (`tests/recovery_test.cc`) uses real processes. A forked child writes a deterministic stream of ops (single puts, deletes, and every 25th op a 20-op batch) and writes each op index to a pipe after `write()` returns. The parent SIGKILLs it after a random number of acks, drains any acks still in the pipe, reopens the database and requires the contents to equal a `std::map` model after all acked ops, or after one more (the op in flight may have reached the WAL without being acked). A second test aims the kill inside a large checkpoint using a calibration round, and a parameterized test uses named failpoints (`_exit(77)` at a given step) to crash at each step of the checkpoint protocol and then again at the same step during recovery.

The benchmark (`bench/bench.cc`) times every operation individually with `steady_clock` and reports exact percentiles, wall-clock throughput, throughput per CPU-second (from `getrusage`) and the 1-minute load average for each row.

### Problems

1. The machine was shared with a dozen other builds. An earlier, interrupted benchmark run happened at a load average of about 256 on 14 cores, and its wall-clock numbers were roughly an order of magnitude below what I measured later at load 4 to 6. I discarded it. That is why every benchmark row now records ops per CPU-second and the load average, why `scripts/run_bench.sh` builds with `-j2`, and why I ran one benchmark and one test binary at a time.

2. Apple clang 17's sanitizer runtime hung at startup on this macOS version. An earlier UBSan-only run under Apple clang stalled inside the large-checkpoint crash test and never finished. I switched the sanitizer preset to Homebrew LLVM 23 (`/opt/homebrew/opt/llvm/bin/clang++`), which runs ASan and UBSan together without trouble.

3. The first Homebrew LLVM build failed with "use of undeclared identifier `errno`" and `EINTR` in `src/pager.cc`. Three files used `errno` without including `<cerrno>` and only compiled under Apple clang because some other system header pulled it in transitively. I added the include to `pager.cc`, `wal.cc` and `db.cc`.

4. Homebrew LLVM emits DWARF 5 debug info and Apple's linker printed "can't parse dwarf compilation unit info" for every object file. It was harmless but buried real warnings, so the sanitizer build now passes `-gdwarf-4` when the compiler is upstream Clang.

5. A comment in `wal.h` claimed that on APFS an append which extends the file costs about 14 times more than a write inside a preallocated file. I had not measured that in this session, so I wrote the `walgrow` experiment to check. The claim did not hold: preallocation made a 128-byte append about 1.7 times faster with no sync, about 1.09 times faster with `fsync`, and made no difference with `F_FULLFSYNC` (numbers under Experiments). I kept preallocation, since it is free and replay already handles a garbage tail, and corrected the comment.

6. SQLite with `synchronous=FULL` and `fullfsync=ON` committed about 4 times faster than kvdb with `F_FULLFSYNC` (1,006 versus 247 commits per second). A real `F_FULLFSYNC` on this machine takes about 4 ms, and SQLite's median commit took 0.69 ms, so SQLite cannot be issuing a full flush on every commit. My hypothesis is that Apple's system SQLite, which is the copy the benchmark links, uses the lighter `F_BARRIERFSYNC` for WAL commits. I have not verified this, so the comparison in that one column is not apples to apples.

7. The random-kill crash test almost never lands inside a checkpoint, because checkpoints are short compared with the time between them. In the final release run it recovered 0 of 12 rounds from the journal. The journal path is covered instead by the calibrated large-checkpoint test and by the deterministic failpoint tests. How many of the calibrated kills land in the "journal complete, data partly written" window depends on timing: 6 of 10 in the release run and 2 of 10 under ASan, which slows the checkpoint about sevenfold.

### Experiments

All runs were on an Apple M4 Pro (14 cores, 48 GB) running macOS 26 (Darwin 25.5.0), release build at `-O2`, with the load average recorded next to each row. Keys are 16 bytes (`user%012d`, so lexicographic order is numeric order), values are 100 bytes.

1. Correctness. `./build/release/kvdb_tests` ran 36 tests in 8 suites, all passing in 2.5 s (`results/tests_release.txt`). The ASan plus UBSan build ran the same 36 tests, all passing in 34.7 s (`results/tests_asan.txt`). In the release run the random-kill test did 12 SIGKILL rounds covering 9,721 acknowledged ops, with 2,254 WAL records redone across the reopens. The calibrated checkpoint test measured a checkpoint of 6.7 ms (1.4 ms journal, 5.4 ms in-place data and truncation) and repaired 6 of its 10 killed rounds from a complete journal (`results/tests_release.txt`).

2. Main throughput and latency (`scripts/run_bench.sh 1000000 3`, raw per run in `results/bench_main_rep1.csv` to `results/bench_main_rep3.csv` with matching `.txt` logs, medians in `results/summary_main.csv`). For each engine: insert 1M keys in sequential or shuffled order, then read all 1M in sequential and shuffled order, then scan everything. Engines: `std::map<std::string, std::string>` in memory; kvdb with a 65,536-frame (256 MiB) pool; kvdb with a 4,096-frame (16 MiB) pool; SQLite 3.51.0 with `journal_mode=WAL`, `synchronous=OFF`, a 256 MiB cache and a `WITHOUT ROWID` table. kvdb ran with `SyncMode::kNone`, so both disk engines write their log on every commit but neither syncs. Load average was 3.9 to 6.3 during these runs (`results/machine.txt` and the `load1` column).

3. Durability cost (`results/bench_sync.csv`, `results/bench_sync.txt`). 2,000 random puts, one per commit, under each sync mode for kvdb and SQLite, then kvdb with `F_FULLFSYNC` and 1, 4, 16, 64 or 256 puts per `WriteBatch`.

4. WAL preallocation (`results/bench_walgrow.csv`, `results/bench_walgrow.txt`). 3,000 raw 128-byte `pwrite` appends, each followed by no sync, `fsync` or `F_FULLFSYNC`, into a file that either grows with each append or was sized in advance with `ftruncate`.

### Results

Throughput and latency at 1M keys, median of 3 runs (`results/summary_main.csv`; plots in `results/throughput.png` and `results/latency.png`):

| Workload | std::map | kvdb 256 MiB | kvdb 16 MiB | SQLite |
|---|---|---|---|---|
| sequential put, ops/s | 3,852,003 | 728,263 | 719,354 | 94,221 |
| random put, ops/s | 1,098,039 | 439,183 | 206,867 | 49,851 |
| random get after random put, ops/s | 840,719 | 1,251,868 | 639,082 | 170,354 |
| sequential get after sequential put, ops/s | 6,986,963 | 3,668,542 | 3,364,736 | 253,240 |
| random put p50 / p99, ns | 875 / 1,791 | 1,833 / 5,083 | 2,583 / 5,708 | 10,667 / 74,041 |
| random get p50 / p99, ns | 1,042 / 2,125 | 750 / 1,250 | 1,542 / 2,209 | 5,792 / 7,875 |
| full scan after random put, keys/s | 11,357,624 | 60,934,431 | 20,973,686 | 13,935,243 |

What I take from this:

- kvdb random puts are 8.8 times faster than SQLite's (439 K versus 50 K per second) and random gets 7.3 times faster. Most of this gap is not storage engine quality: SQLite parses nothing per call here (prepared statements), but it still runs its bytecode VM, record encoding and a more general B-tree. This compares a specialized KV path against a general SQL engine.
- With the 256 MiB pool the whole database fits in memory: the pool hit rate was 1.000 and the file was 122 MB after the sequential fill (`note` column of `results/bench_main_rep1.csv`). So "kvdb 256 MiB" measures an in-memory B+ tree plus a non-synced WAL write per commit, not disk reads. Even the 16 MiB pool only misses into the OS page cache, not the SSD.
- kvdb random gets beat `std::map` (1.25 M versus 0.84 M per second, p50 750 ns versus 1,042 ns). A red-black tree after a random fill scatters 1M nodes across the heap and touches about 20 of them per lookup, while the B+ tree touches 4 pages and does binary search inside each. The full scan shows the same effect more strongly: 61 M keys per second for kvdb against 11 M for `std::map` after a random fill.
- Writes are where kvdb loses to `std::map`: 5.3 times slower for sequential puts and 2.5 times for random puts. Each put encodes a batch, computes a CRC, issues one `pwrite` system call to the WAL, and may split nodes.
- Leaf fill confirms the append-split special case: sequential fill produced 30,304 leaves at 99% fill, random fill produced 43,324 leaves at 69% fill, the textbook value for random B-tree inserts is about 69% (ln 2), and both trees have height 4 (`results/bench_main_rep1.csv`).
- The small pool shows the price of NO-STEAL. Random fill with 16 MiB of frames needed 213 checkpoints against 3 for the large pool, and ran 2.1 times slower (207 K versus 439 K puts per second, pool hit rate 0.832). Sequential fill barely noticed (9 checkpoints, 719 K versus 728 K) because it dirties only the rightmost path.

Durability (`results/bench_sync.csv`, plot in `results/sync.png`):

| Mode | kvdb commits/s | kvdb p50 | SQLite commits/s | SQLite p50 |
|---|---|---|---|---|
| no sync | 693,922 | 1.2 µs | 70,517 | 10.3 µs |
| fsync | 38,162 | 24.0 µs | 4,900 | 67.5 µs |
| F_FULLFSYNC | 247 | 4.01 ms | 1,006 | 0.69 ms |

With `F_FULLFSYNC`, group commit scales almost linearly: 243, 980, 3,960, 15,485 and 59,450 puts per second at 1, 4, 16, 64 and 256 puts per batch, with the commit latency staying at about 4 ms. The drive's cache flush is the cost and it is paid once per batch. A truly durable single-put commit on this Mac costs about 4 ms, which is 3,300 times the cost of a non-synced one. For SQLite's faster `F_FULLFSYNC` column, see Problems item 6.

WAL preallocation (`results/bench_walgrow.csv`):

| Sync | growing file, appends/s | preallocated, appends/s | ratio |
|---|---|---|---|
| none | 601,253 | 1,021,581 | 1.70 |
| fsync | 34,394 | 37,525 | 1.09 |
| F_FULLFSYNC | 247 | 248 | 1.00 |

### What I would change

- Make the buffer pool STEAL and the log physiological (page id, slot, before and after image) with page LSNs. NO-STEAL made recovery easy to reason about, but the small-pool run shows it turns memory pressure directly into checkpoints (213 of them for 1M random puts), and it rules out transactions larger than memory.
- Checksum every page. Today a torn or bit-flipped page in the data file would be read silently. The journal and WAL already carry CRCs; pages should too, with torn-page detection at read time.
- Implement real delete with merge and redistribution plus a free-page list. Lazy delete was the right cut for v0, but a delete-heavy workload would waste space without bound.
- Replace the per-put `WriteBatch` encode, CRC and `pwrite` with an in-memory log buffer flushed at commit boundaries. The write path is where kvdb loses most to `std::map`, and one system call per put is the obvious target.
- Use CLOCK instead of an exact LRU list once there are multiple threads, to avoid taking a lock to move a list node on every hit.
- Verify the SQLite sync behavior (Problems item 6) with `fs_usage` or a build of SQLite from source, so the durability comparison is like for like.
- Run the benchmarks on a quiet machine and on a database larger than RAM, so the buffer pool has to go to the SSD and the random-read numbers reflect I/O rather than memory.

### Verification

On 2026-09-27 an independent clean release rebuild passed all 36 tests, including the forked-process crash tests (`Recovery.KillNineDuringWritesLosesNothingAcknowledged`, `Recovery.KillNineDuringLargeCheckpoint` and the `Steps/CheckpointCrash.*` failpoint tests). The ASan plus UBSan build and the benchmarks were not rerun independently; the sanitizer result and every benchmark number above come from the original runs saved under `results/`.
