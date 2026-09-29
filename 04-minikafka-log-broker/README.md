# minikafka: a Kafka-like event streaming broker in C++20

minikafka is a single-broker event log written from scratch. Topics are split into partitions; each partition is an append-only log of segment files on disk with a sparse offset index. Producers send batches over a small binary TCP protocol, consumers fetch from an offset (with long polling), and consumer groups get partitions assigned by the broker and commit their positions into an internal log, so they resume where they left off after a restart. Old segments are deleted by size-based retention.

It is a learning project that follows Kafka's storage design closely (same byte format on the wire and on disk, sparse index, segment rolling, `__consumer_offsets`) and is honest about what it leaves out. See [DESIGN.md](DESIGN.md) for the architecture and [DEVLOG.md](DEVLOG.md) for how it was built and measured.

## Status

| Milestone | Scope | Status |
|---|---|---|
| v0 | Single broker; topics and partitions; segmented log with sparse index; crash recovery (torn-tail truncation); binary TCP protocol with batched produce and long-poll fetch; consumer groups with generation-fenced commits stored in an internal log; size retention; producer and consumer libraries; CLI; tests; benchmarks | Done |
| v1 | Multi-broker replication: leader and followers per partition, ISR tracking, high watermark, a controller for leader election | Planned |
| v2 | Zero-copy fetch with `sendfile`, pipelined producer requests, kqueue/epoll event loop instead of thread per connection | Planned |
| v3 | Log compaction (also used to bound `__consumer_offsets`), time-based retention | Planned |
| v4 | Exactly-once semantics: idempotent producer (producer id + sequence numbers), transactions | Planned |

### What v0 does not do

Every item on the v0 list was built. The simplifications below are deliberate, and all of them are covered in DESIGN.md and DEVLOG.md.

- One broker only; no replication, so durability is only as good as the flush policy (see DESIGN.md, Durability model).
- No SyncGroup barrier: during a rebalance two consumers can briefly read the same partition. Commits are fenced by generation, so the effect is duplicate delivery, never lost offsets.
- No topic deletion, no time-based retention, no compaction, no authentication, no compression.
- The producer has one request in flight per connection.

## Layout

```
include/minikafka/   public headers (record format, segment, partition log, broker, clients)
src/                 implementation
tools/               mk-broker (server) and mk-cli (command-line client)
tests/               GoogleTest unit and integration tests
bench/               mk-bench throughput and latency harness
scripts/             run_benchmarks.sh, plot.py, smoke_test.sh
results/             raw benchmark outputs (CSV, TXT) and plots (PNG)
```

## Build

Requirements: CMake 3.24+, Ninja, a C++20 compiler (developed with Apple clang 17 on macOS 26.5, Apple M4 Pro). GoogleTest is fetched by CMake.

```sh
# Release (-O2)
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release

# Debug with AddressSanitizer + UBSan.
# Apple clang 17's sanitizer runtime hangs at startup on macOS 26.5 (see DEVLOG, Problems),
# so the sanitizer build uses Homebrew LLVM.
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMK_SANITIZE=ON \
      -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++"
cmake --build build/debug

# Optional: ThreadSanitizer
cmake -S . -B build/tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMK_TSAN=ON -DMK_BUILD_BENCH=OFF \
      -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++"
cmake --build build/tsan
```

## Test

```sh
ctest --test-dir build/release --output-on-failure
ctest --test-dir build/debug --output-on-failure     # ASan + UBSan
./build/tsan/mk-tests                                 # TSan
./scripts/smoke_test.sh                               # real broker process + CLI, including a restart
```

31 tests (GoogleTest) cover: CRC-32C vectors and hardware/software agreement; record encoding and corruption detection; offset assignment and ordering across many segments; random offset lookups through the sparse index; torn-tail and corrupt-tail recovery; rebuilding a broken sealed index; size retention (including across restart); offset store replay; index entries inside large batches; group assignment, rebalance, expiry and generation fencing; and over TCP: ordering with one and with six concurrent producers on one partition, key-to-partition stickiness, durability across a broker restart, long-poll wakeup, retention through the broker, offset commit and resume, committed offsets surviving a restart, and fenced stale commits.

## Benchmark

```sh
./scripts/run_benchmarks.sh 3          # writes results/*.csv and results/bench_env.txt
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/plot.py        # writes results/*.png
```

`scripts/run_benchmarks.sh` runs one mode at a time, and each mode runs its configurations one after another, so at most a broker plus a few client threads are busy at once. `results/latency_raw.csv` (about 5 MB) is gitignored and regenerated by the script.

### Headline numbers

Measured on a shared machine that had other heavy jobs running (the 1 minute load average is recorded in every CSV row and was 8 to 11 for these runs; an earlier run at load 180 to 230 is kept in `results/run2_loaded/` for comparison). Treat them as loaded-machine numbers, not a clean benchmark.

| What | Result | File |
|---|---|---|
| Produce, 1000 B messages, batch 1000 | 2528 MB/s, 2.53 M msgs/s | `results/produce_throughput.csv` |
| Produce, 100 B messages, batch 1 | 32.9 k msgs/s | `results/produce_throughput.csv` |
| Consume, 100 B messages, 1 MiB fetches | 2150 MB/s, 21.5 M msgs/s | `results/consume_throughput.csv` |
| End-to-end latency, 10k msgs/s, no fsync | p50 41.5 us, p99 142 us, p99.9 1.2 ms | `results/latency.csv` |
| End-to-end latency, 200 msgs/s, F_FULLFSYNC | p50 4.0 ms, p99 7.8 ms | `results/latency.csv` |
| 1000 B, batch 100: no fsync vs fsync vs F_FULLFSYNC | 1188 vs 595 vs 23 MB/s | `results/flush_policy.csv` |

Plots: `results/produce_throughput.png`, `consume_throughput.png`, `latency_cdf.png`, `flush_policy.png`, `producer_scaling.png`.

`mk-bench` starts a broker in the same process and talks to it over loopback TCP with real clients. Modes: `produce` (throughput vs batch size and message size), `consume` (throughput vs fetch size and message size), `latency` (end-to-end percentiles at a fixed send rate), `flush` (throughput vs flush policy), `scaling` (aggregate throughput vs number of producers). Headline results are in DEVLOG.md, Results.

## Run

```sh
./build/release/mk-broker --data-dir ./data --port 9092 &
./build/release/mk-cli create-topic orders 3
printf 'a\nb\nc\n' | ./build/release/mk-cli produce orders
./build/release/mk-cli consume orders my-group 10     # commits after each poll
./build/release/mk-cli offsets orders 0
```

Broker flags: `--data-dir`, `--host`, `--port`, `--segment-bytes` (default 64 MiB), `--index-interval` (default 4096), `--retention-bytes` (default unlimited), `--flush none|fsync|full`.

### Library use

```cpp
mk::Client client("127.0.0.1", 9092);
client.create_topic("orders", 3);

mk::Producer producer(client, {.batch_records = 500});
producer.send("orders", "customer-42", "order payload");   // keyed: same key, same partition
producer.flush();

mk::Consumer consumer(client, {.group = "billing"});
consumer.subscribe({"orders"});
for (const mk::Record& r : consumer.poll(1000)) { /* r.offset, r.key, r.value */ }
consumer.commit_sync();
```
