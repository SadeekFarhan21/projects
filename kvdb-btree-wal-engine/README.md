# kvdb: a database from scratch

kvdb is a persistent, single-node key-value store written in C++20 with no runtime dependencies. It stores a B+ tree in 4 KiB pages in a single file, caches pages in a buffer pool with LRU eviction and pin counts, logs every commit to a write-ahead log with a configurable fsync policy, and recovers committed data after a crash by replaying that log. A small REPL sits on top.

It is the first project in a systems series. The goal of v0 is a storage engine that is small enough to read in an afternoon and is tested hard enough (kill -9 during writes and during checkpoints) to trust.

```
$ ./build/release/kvdb_repl demo.kv
kvdb opened demo.kv (recovered 0 WAL records). type 'help'.
kvdb> put apple red fruit
OK
kvdb> put cherry dark red
OK
kvdb> scan - - 10
apple = red fruit
cherry = dark red
(2 rows)
```

## Status

| Milestone | Scope | Status |
|---|---|---|
| v0 | Pager, buffer pool (LRU + pins), B+ tree (insert, get, range scan, lazy delete), WAL with fsync and redo recovery, atomic write batches, crash-atomic checkpoints, REPL, crash tests, benchmarks vs std::map and SQLite | Done |
| v1 | Transactions: MVCC snapshots or strict 2PL, with isolation tests | Planned |
| v2 | Concurrency: latch crabbing in the B+ tree, a thread-safe buffer pool, group commit across threads | Planned |
| v3 | Checksums on every page, torn-page detection, a free-page list and real delete (merge and redistribute) | Planned |
| v4 | LSM-tree variant (memtable, SSTables, bloom filters) and a head-to-head comparison with the B+ tree | Planned |
| v5 | LSM compaction strategies (leveled vs tiered) and write amplification measurements | Planned |

### What v0 does not do

These were cut or deliberately left out, and are listed in DEVLOG.md as well:

1. Delete is lazy. It removes the cell but never merges nodes or frees pages, so space is not reclaimed.
2. Keys are limited to 128 bytes and values to 512 bytes (no overflow pages).
3. Single-threaded, single-process. A second open of the same file fails with a lock error.
4. No page checksums. The WAL and the checkpoint journal are CRC-protected; the page file is not.
5. The buffer pool is NO-STEAL: all pages dirtied since the last checkpoint must fit in memory, so a small pool forces frequent checkpoints.

## Layout

```
include/kvdb/     public headers (db.h is the API)
src/              pager, buffer pool, slotted node, B+ tree, WAL, DB (recovery, checkpoint)
tools/repl.cc     interactive shell
tests/            GoogleTest suites, including fork + SIGKILL crash tests
bench/bench.cc    throughput and latency harness (kvdb, std::map, SQLite)
scripts/          run_bench.sh (runs everything), plot_results.py (summary + PNGs)
results/          raw benchmark and test outputs from the runs cited in DEVLOG.md
DESIGN.md         architecture, on-disk formats, invariants, trade-offs
DEVLOG.md         build log and case study
```

## Build

Requirements: CMake 3.24+, Ninja, a C++20 compiler (tested with Apple clang 17 and Homebrew LLVM 23 on macOS 26, Apple M4 Pro). GoogleTest is fetched by CMake at configure time. SQLite is optional; the macOS SDK's copy is found automatically and enables the SQLite baseline in the benchmark.

```
cmake --preset release            # -O2, build/release
cmake --build --preset release -j2
```

Sanitizer build (AddressSanitizer + UndefinedBehaviorSanitizer, Debug, libc++ hardening). The preset uses Homebrew LLVM (`/opt/homebrew/opt/llvm/bin/clang++`) because Apple clang 17's sanitizer runtime hung at startup on this macOS version. Edit `CMakePresets.json` if your LLVM lives elsewhere.

```
cmake --preset asan               # build/asan
cmake --build --preset asan -j2
```

## Test

```
./build/release/kvdb_tests        # 36 tests, about 4 s
./build/asan/kvdb_tests           # same tests under ASan + UBSan, about 40 s
ctest --preset release            # or through CTest
```

The suites cover the slotted node format, the buffer pool (hits, LRU order, pinned and dirty pages never evicted), the B+ tree against a `std::map` model (random ops, splits with maximum-size cells, range bounds, delete everything and reinsert), the WAL (a torn tail at every possible cut point, corruption, preallocated zero tail), and crash recovery:

- `Recovery.KillNineDuringWritesLosesNothingAcknowledged`: a forked child writes and acknowledges each commit over a pipe; the parent SIGKILLs it at a random point, reopens, and checks that the database equals the model after every acknowledged op (or one more, the in-flight op). 12 rounds on the same file.
- `Recovery.KillNineDuringLargeCheckpoint`: SIGKILL aimed inside a large checkpoint (6,000 mixed ops plus 50,000 new keys, pool of 8,192 pages), so some kills land after the journal is durable but before the in-place writes finish.
- `Steps/CheckpointCrash.*`: deterministic crashes (`_exit`) at each step of the checkpoint protocol, then a second crash at the same step during recovery.

Latest outputs: `results/tests_release.txt`, `results/tests_asan.txt`.

## Benchmark

```
scripts/run_bench.sh 1000000 3                 # 1M keys, 3 repetitions, about 5 minutes
.venv/bin/python scripts/plot_results.py       # results/summary_main.csv and PNGs
```

The plotting script needs matplotlib (`python3 -m venv .venv && .venv/bin/pip install matplotlib`). Single experiments can be run directly, for example `./build/release/kvdb_bench --exp=main --n=100000 --out=/tmp/x.csv`; `--exp` is one of `main`, `sync`, `walgrow`.

Every op is timed individually with `steady_clock`, and p50/p99/p99.9 are exact over all ops. Each row also records ops per CPU-second and the 1-minute load average, because the benchmark machine is shared.

Headline results, 1M keys (16 B keys, 100 B values), median of 3 runs, no fsync per commit for either engine, load average 4 to 6 (`results/summary_main.csv`):

| Workload | std::map | kvdb (256 MiB pool) | kvdb (16 MiB pool) | SQLite 3.51 (WAL, sync off) |
|---|---|---|---|---|
| sequential put | 3.85 M/s | 728 K/s | 719 K/s | 94 K/s |
| random put | 1.10 M/s | 439 K/s | 207 K/s | 50 K/s |
| random get | 841 K/s | 1.25 M/s | 639 K/s | 170 K/s |
| random put p50 / p99 | 875 / 1,791 ns | 1,833 / 5,083 ns | 2,583 / 5,708 ns | 10,667 / 74,041 ns |
| random get p50 / p99 | 1,042 / 2,125 ns | 750 / 1,250 ns | 1,542 / 2,209 ns | 5,792 / 7,875 ns |

Durable commits (`results/bench_sync.csv`): one put per commit runs at 38 K/s with `fsync` and 247/s with `F_FULLFSYNC` (about 4 ms each); batching 256 puts per `F_FULLFSYNC` commit reaches 59 K puts/s.

Plots: `results/throughput.png`, `results/latency.png`, `results/sync.png`. DEVLOG.md discusses what these numbers do and do not show.

## Use as a library

```cpp
#include "kvdb/db.h"

kvdb::Options opts;
opts.sync = kvdb::SyncMode::kFullFsync;   // kNone, kFsync (default), kFullFsync
auto db = kvdb::DB::open("data.kv", opts);
db->put("user:1", "alice");
std::string v;
if (db->get("user:1", &v)) { /* ... */ }

kvdb::WriteBatch b;                        // atomic, one WAL record, one sync
b.put("a", "1");
b.del("user:1");
db->write(b);

db->scan("a", "z", [](std::string_view k, std::string_view v) { return true; });
```

A database at `data.kv` uses `data.kv`, `data.kv-wal` and `data.kv-journal`.
