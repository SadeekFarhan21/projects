# kvd, a Redis-like in-memory store in C++20

kvd is a from-scratch, single-threaded, event-loop TCP server that speaks a subset of RESP2, the Redis wire protocol. Strings only, with TTLs, pipelining and append-only-file persistence. It is small enough to read in an afternoon and is built to be measured. `kvbench`, a pipelined C++ load generator, ships with it.

See [DESIGN.md](DESIGN.md) for the architecture and trade-offs and [DEVLOG.md](DEVLOG.md) for how it was built, what broke, and the measurements.

## Status

| Milestone | Scope | Status |
|---|---|---|
| v0 | Event loop (kqueue, epoll behind an interface), RESP2 parser with pipelining, PING GET SET(EX/PX/NX/XX/KEEPTTL) DEL EXISTS EXPIRE TTL INCR KEYS, lazy + active expiry, AOF with replay and torn-tail recovery, load generator, benchmarks | Done |
| v0 gaps | epoll backend compiled only on Linux and not yet run (no Linux box in this session), no real redis-cli or redis-benchmark run (not installed), no AOF rewrite | Open, see Known issues |
| v1 | Multithreaded I/O (Redis 6 style: I/O threads parse and write, one thread executes) | Planned |
| v2 | RDB snapshots via `fork()` and copy-on-write, AOF rewrite | Planned |
| v3 | Data structures: lists, hashes, sets, sorted sets on a skiplist | Planned |
| v4 | Pub/sub | Planned |
| v5 | Leader/follower replication (PSYNC-style backlog) | Planned |

## Supported commands

`PING [msg]`, `ECHO`, `GET`, `SET key value [NX|XX] [EX s|PX ms|EXAT s|PXAT ms|KEEPTTL]`, `DEL key...`, `EXISTS key...`, `EXPIRE`, `PEXPIRE`, `PEXPIREAT`, `PERSIST`, `TTL`, `PTTL`, `INCR`, `DECR`, `INCRBY`, `DECRBY`, `KEYS pattern`, `DBSIZE`, `FLUSHALL`/`FLUSHDB`, `SELECT 0`, `QUIT`, plus `COMMAND` and `CONFIG GET` stubs that return empty arrays so `redis-cli` and `redis-benchmark` handshakes succeed. Both multibulk and inline (telnet style) requests are accepted.

## Build

Requires CMake 3.24+, Ninja and a C++20 compiler. GoogleTest is fetched by CMake at configure time.

```sh
cmake --preset release && cmake --build --preset release   # -O2, build/release/
cmake --preset asan    && cmake --build --preset asan      # Debug + ASan/UBSan, build/asan/
```

The `asan` preset uses Homebrew LLVM (`/opt/homebrew/opt/llvm/bin/clang++`) because Apple clang 17's AddressSanitizer runtime hangs at startup on this macOS 26 machine, even for an empty `main` (details in DEVLOG). Release builds use Apple clang 17.

## Test

```sh
ctest --preset release     # 49 GoogleTest cases + the end-to-end Python client test
ctest --preset asan        # same suite under AddressSanitizer + UBSan
python3 scripts/client_test.py --server build/release/kvd   # e2e test alone
```

`scripts/client_test.py` is a stdlib-only RESP client written independently of the C++ code. It starts the real binary, runs every command, checks pipelining, real-time expiry, SIGTERM shutdown, `kill -9` recovery with `appendfsync always`, and torn-tail AOF recovery. It stands in for `redis-cli`, which is not installed here. If you have it, the server works with it directly:

```sh
build/release/kvd --port 6380 &
redis-cli -p 6380 set greeting hello ex 60
redis-cli -p 6380 ttl greeting
redis-benchmark -p 6380 -t set,get -n 100000 -P 16 -q
```

## Run

```sh
build/release/kvd --port 6379 --aof appendonly.aof --appendfsync everysec
build/release/kvd --appendonly no          # pure in-memory
printf 'SET a 1\r\nGET a\r\n' | nc 127.0.0.1 6379   # inline protocol works too
```

Flags: `--port N` (0 picks a free port), `--bind ADDR`, `--aof PATH`, `--appendonly yes|no` (default yes), `--appendfsync always|everysec|no` (default everysec), `--hz N` (cron frequency, default 10), `--verbose`.

## Benchmark

```sh
build/release/kvbench -p 6379 -c 50 -P 16 -n 1000000 -w set     # one run, human readable
python3 scripts/run_benchmarks.py --reps 3 --threads 4          # full suite -> results/bench_*.jsonl
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/plot_results.py                        # -> results/summary.csv + PNGs
```

`kvbench` options: `-c` connections, `-P` pipeline depth (requests in flight per connection), `-n` total requests, `-t` threads (default min(c, 8)), `-d` value size, `-r` keyspace size, `-w set|get|mix|ping`, `--json`.

`run_benchmarks.py` options: `--only pipeline,clients,workload,aof,fsync1,expiry,replay`, `--reps N`, `--threads N` (cap kvbench client threads; the reported numbers used `--threads 4` to keep the shared machine lightly loaded).

Every benchmark row records the machine's 1-minute load average and the CPU time the server process used during the run (`ops_per_server_cpu_s`). When the server is saturated the two throughput numbers agree, which is a quick check that the server, not the load generator, was the bottleneck.

### Headline numbers

Apple M4 Pro, 14 cores, macOS 26.5, release build, loopback TCP, 3 repetitions per configuration, medians, 1-minute load average 7 to 12 during the runs. All from `results/summary.csv`, built from the `results/bench_*.jsonl` files named below.

| Measurement | Result | Source |
|---|---|---|
| SET, 1 connection, no pipelining | 40,120 ops/s, p50 22 us, p99 65 us | `bench_fsync1.jsonl` (aof off) |
| SET, 50 connections, no pipelining | 107,160 ops/s, p50 438 us, p99 936 us | `bench_pipeline.jsonl` |
| SET, 50 connections, pipeline 16 | 999,599 ops/s, p99 1.7 ms | `bench_pipeline.jsonl` |
| SET, 50 connections, pipeline 128 | 2,942,811 ops/s, p99 4.4 ms | `bench_pipeline.jsonl` |
| GET / SET / 50:50 mix at c=50, P=16 | 1,198,850 / 1,072,646 / 1,046,355 ops/s | `bench_workload.jsonl` |
| SET c=50 P=16, AOF off vs `appendfsync always` | 1,217,265 vs 1,042,970 ops/s | `bench_aof.jsonl` |
| SET c=1 P=1, AOF off vs `appendfsync always` | p50 22 us vs 59 us | `bench_fsync1.jsonl` |
| Active expiry of 200,000 keys sharing one deadline | all reclaimed between 63 and 171 ms after the deadline | `bench_expiry.jsonl` |
| AOF replay, 1.6 M SETs (94.4 MB) | 854 ms | `bench_replay.jsonl` |

Charts: `results/pipeline.png`, `results/clients.png`, `results/aof.png`, `results/expiry.png`. An earlier full run made while the machine was at load average 200 to 360 is kept in `results/run1_loaded/` for comparison; its wall-clock throughput was about 10x lower (see DEVLOG, Problems).

## Layout

```
src/        server library + kvd main
  resp.*          RESP2 parser and reply encoders
  store.*         keyspace, TTL index, lazy and active expiry
  commands.*      command table and semantics, AOF propagation rewrite
  aof.*           append-only file, fsync policies, replay
  poller*.{h,cpp} kqueue and epoll backends behind one interface
  server.*        event loop, connections, pipelining, backpressure
bench/loadgen.cpp   kvbench load generator
tests/              GoogleTest unit and in-process TCP tests
scripts/            e2e client test, benchmark runner, plotting
results/            raw benchmark output (jsonl), summary.csv, charts
  run1_loaded/      first run, made at load average 200 to 360
```

## Known issues

- The epoll backend (`src/poller_epoll.cpp`) compiles only on Linux and has never been built or run; there was no Linux machine or container available in this session.
- No AOF rewrite. The file grows forever, including records for keys that later expired or were deleted.
- `std::unordered_map` rehashes all at once, so growing past a power of two stalls the loop for a moment. Redis avoids this with incremental rehashing.
- `everysec` fsync runs inline on the event loop, so a slow disk shows up as client latency. Redis does it on a background thread.
- On macOS, `fsync` does not flush the drive's write cache (`F_FULLFSYNC` would). `appendfsync always` therefore protects against process crashes, not power loss.
- `KEYS` is O(N) and blocks the loop, as in Redis.
- Not implemented from the Redis string API: `SET ... GET`, `GETSET`, `MGET`/`MSET`, `APPEND`, `EXPIRE` flags (`NX|XX|GT|LT`), quoted strings in inline commands.
