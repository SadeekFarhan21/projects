# DEVLOG: building kvd, a Redis-like store, from scratch

### What I wanted to build

I wanted to understand why Redis, a single-threaded server, is fast enough that most people never need anything else. The way to find out was to build the core of it and measure it: a TCP server with one event loop, the RESP2 wire protocol, string keys with TTLs, pipelining, and an append-only file (AOF) that survives a crash.

The v0 scope was fixed up front: PING, GET, SET with EX/PX, DEL, EXISTS, EXPIRE, TTL, INCR and KEYS; lazy and active expiry; AOF persistence with replay on startup; a kqueue event loop with an epoll path behind the same interface; and a C++ load generator that reports throughput and latency percentiles. Multithreaded I/O, RDB snapshots, richer data types, pub/sub and replication were explicitly left for later milestones.

The result is `kvd`, about 1,900 lines of C++20 in `src/`, plus `kvbench` (the load generator), 49 GoogleTest cases and an end-to-end Python client test.

### Theory

A request/response server spends most of its time waiting. With one thread per connection, the waiting is done by blocked threads, and the cost is context switches and memory for stacks. An event loop inverts that: one thread asks the kernel (kqueue on macOS, epoll on Linux) which sockets are ready, and only touches those. As long as every command is short, one core can serve thousands of connections, and because only one thread touches the data, every command is atomic without a single lock.

Two ideas decide how fast such a server can go.

The first is that the per-request cost is dominated by system calls and round trips, not by the hash table lookup. A GET that finds its key in a hash table costs well under a microsecond; the `read()`, `write()` and `kevent()` around it cost several. Pipelining attacks exactly this. If a client sends 16 commands before reading any replies, the server reads them with one `read()`, executes them back to back, and answers with one `write()`. The syscall cost is shared 16 ways.

The second is Little's law: requests in flight = throughput x latency. With 50 connections and one request in flight each, a server doing 100,000 requests per second must show a mean latency of 50 / 100,000 s = 500 us, however fast each individual command is. Latency in a saturated closed-loop benchmark is set by the queue, not the work. I used this to sanity check every result below.

For expiry, the problem is that a key with a TTL must disappear on time but nobody wants a timer per key. Redis combines two cheap mechanisms. Lazy expiry checks the deadline whenever a key is touched, so an expired key is never visible. Active expiry runs about 10 times per second, samples 20 random keys that have a TTL, deletes the expired ones, and repeats while more than 25% of the sample was expired. That bounds how much memory expired but untouched keys can hold, at a cost proportional to the garbage, not the keyspace.

For durability, the AOF is a log of write commands in the same RESP format clients send. Replaying it rebuilds the dataset. The durability knob is when to call `fsync`: after every write (`always`), once per second (`everysec`) or never (`no`, leave it to the OS).

### Architecture

Everything runs on one thread. The flow of a request is:

```
socket readable -> read() up to 16 KB into Client.in
                -> parse every complete RESP frame in the buffer
                -> execute each against the Store, reply bytes into Client.out
                -> write commands are rewritten and appended to the AOF buffer
before sleeping -> write() (and fsync if "always") the AOF buffer
                -> send() every client's pending replies
                -> kevent()/epoll_wait() until the next socket or cron tick
cron (10 Hz)    -> one active expiry cycle, everysec fsync check
```

The components, each in its own file under `src/`:

- `poller.h` with `poller_kqueue.cpp` and `poller_epoll.cpp`: a four-method interface (add, remove, set read/write interest, wait). The server never includes a platform header for readiness.
- `resp.cpp`: a stateless parser that takes the unconsumed tail of a client's buffer and returns complete, incomplete or error. It also accepts the inline form, so `nc` works.
- `store.cpp`: the keyspace, TTL bookkeeping, lazy and active expiry.
- `commands.cpp`: the command table and the rewrite of each write into a time-independent form for the AOF.
- `aof.cpp`: buffered append, the three fsync policies, replay with torn-tail recovery.
- `server.cpp`: accept, read, pipelined execution, write, backpressure and the cron.

DESIGN.md has the full diagram, the invariants and the trade-offs.

### Implementation

A few details carried most of the weight.

Offsets instead of erasing. Each client has an input buffer with a consumed offset and an output buffer with a sent offset. Parsing a command advances the offset; the buffer is cleared when fully consumed and compacted only when the dead prefix is over 64 KB and more than half the buffer. Erasing the front after every command would make a deep pipeline quadratic.

Copy arguments once. The parser first scans a multibulk frame recording only offsets. Arguments are copied into `std::string`s only when the whole frame has arrived, so an 8 MB value that trickles in across hundreds of reads is copied once.

A dense vector of TTL keys. Active expiry needs a uniformly random key among those with a TTL. I kept a `std::vector` of pointers to the `unordered_map` nodes that have a TTL, with each entry storing its own index in that vector. Removing a TTL is swap-with-last and pop, O(1). This is only legal because `std::unordered_map` never moves nodes on rehash. The invariant (every TTL node is in the vector exactly once, at the index it records) is checked by `Store::check_invariants` after each step of a 20,000 step randomized test.

AOF entries do not depend on replay time. `SET k v EX 10` is logged as `SET k v PXAT <absolute ms>`, `EXPIRE` as `PEXPIREAT`, an `EXPIRE` with a non-positive TTL as `DEL`, and no-op writes are not logged at all. Replaying the file a day later gives the same deadlines, and keys that expired while the server was down are purged right after replay.

Reply after log. In each loop iteration the AOF buffer is written, and fsynced under `always`, before any reply is sent. So an acknowledged write is in the file. The Python test checks this directly by writing 200 keys with `appendfsync always`, killing the server with SIGKILL and finding all 200 after restart.

Backpressure by pausing reads. If a client pipelines faster than it reads replies, its output buffer grows without bound. Redis disconnects such clients at a hard limit. I stop reading from the client once its unsent output passes a soft limit (1 MB) and resume when it drains.

Testing is layered: unit tests for the parser, glob matcher, store and command semantics; in-process TCP tests that run the real server on an ephemeral port in a thread (pipelining 20,000 commands, an 8 MB value, 16 concurrent clients doing 2,000 INCRs each, backpressure, AOF restart); and `scripts/client_test.py`, an independent stdlib-only RESP client that drives the real binary, including SIGTERM, SIGKILL recovery and a torn AOF tail. It stands in for `redis-cli`, which is not installed on this machine. Everything runs under both the -O2 build and a Debug build with AddressSanitizer and UBSan.

### Problems

Apple clang's sanitizer runtime hangs. The Debug build with `-fsanitize=address,undefined` hung at startup before reaching `main`. It is not my code: an empty `int main(){return 0;}` built with Apple clang 17.0.0 and `-fsanitize=address` hangs on this macOS 26.5 machine. I rechecked while finishing up, and it was still hung when a 15 second timeout killed it. The fix was to point the `asan` CMake preset at Homebrew LLVM's `clang++` and keep Apple clang for the release build. Both builds pass the same 50 tests.

A backpressure test that passed in one build and failed in the other. The first version of `Server.BackpressurePausesAndResumesReads` relied on the default 1 MB soft limit being crossed. Whether it was crossed depended on how fast the kernel drained socket buffers relative to the server, which differed between the fast -O2 build and the slow ASan build, so the "reads were paused" assertion was not deterministic. The test now sets a 64 KB soft limit and sends about 30 MB of replies, so the pause always happens.

A paused client can stall forever. When a client is paused, its input buffer may already hold complete commands. Resuming read interest is not enough: the socket may have nothing new to read, so no readable event arrives and those commands sit there forever. The resume path in `try_write` therefore calls `process_input` immediately after re-enabling reads, and loops if that produces more output. Invariant 7 in DESIGN.md records this and the backpressure test exercises it.

Benchmarking on a machine that was 20 times oversubscribed. The first full benchmark run happened while other heavy jobs were running: the 1-minute load average was 200 to 365 on 14 cores (`results/run1_loaded/bench_pipeline.jsonl` records 364.7 at start). The numbers were useless as a measure of the server. SET at 50 connections without pipelining gave a median of 17,325 ops/s with a p99 of 51.9 ms, and one `workload-mix` repetition gave 480,780 ops/s while the other two gave 40,969 and 27,832 (`results/run1_loaded/bench_workload.jsonl`). Two changes made the data usable. First, every row now records the load average and the server's CPU seconds during the run, so throughput per server CPU second can be compared across conditions. Second, when the load dropped I reran everything one experiment at a time, with `kvbench` capped at 4 threads (a new `--threads` flag on the runner) so the benchmark itself did not add much load. The loaded run is kept in `results/run1_loaded/` for comparison.

The expiry experiment raced itself. The experiment loads 200,000 keys that all expire at one absolute instant (`PXAT`) and then watches `DBSIZE`. If the deadline is chosen before encoding and sending the keys, a slow machine can reach the deadline while keys are still being loaded, and the "drop" is smeared across the load. The script now encodes everything first, measures how long loading 200,000 persistent keys took, and sets the deadline to twice that plus 2 seconds. It records the margin actually achieved: 1,779 ms in the final run (`results/bench_expiry.jsonl`, `expiry-meta`).

A too-short run. My first single-connection fsync experiment used 20,000 requests per run. It showed the effect but the AOF-off baseline swung between 15,340 and 51,910 ops/s across repetitions, which is more noise than signal. I raised it to 100,000 requests per run and replaced the file.

The session was interrupted. Work stopped partway through to reduce CPU load on the shared machine. When I resumed, the code, tests and first benchmark run were intact; I rebuilt both configurations with `-j2`, ran the two test suites one at a time, and redid the benchmarks as described above.

### Experiments

All experiments ran on an Apple M4 Pro (14 cores, 48 GB) over loopback, with the release build (`-O2`, Apple clang 17), `kvbench` using 4 client threads, 3 repetitions per configuration. I report medians. `scripts/run_benchmarks.py` produces the raw `results/bench_*.jsonl` files and `scripts/plot_results.py` produces `results/summary.csv` and the charts. The 1-minute load average during the final runs was between 6 and 13, recorded per row.

1. Pipeline depth. 50 connections, SET with 16 byte values, pipeline depth 1 to 128, AOF off. (`results/bench_pipeline.jsonl`, `results/pipeline.png`)
2. Connection count. Pipeline 1, SET, 1 to 256 connections, AOF off. (`results/bench_clients.jsonl`, `results/clients.png`)
3. Workload mix. 50 connections, pipeline 16, 100,000 keys pre-populated, pure SET vs pure GET vs 50:50. (`results/bench_workload.jsonl`)
4. Persistence cost with many clients. 50 connections, SET at pipeline 1 and 16, with AOF off and with `appendfsync` no, everysec and always. (`results/bench_aof.jsonl`, `results/aof.png`)
5. Persistence cost with one client. 1 connection, pipeline 1, 100,000 SETs, the same four modes. (`results/bench_fsync1.jsonl`)
6. Active expiry. 200,000 persistent keys plus 200,000 keys that all expire at the same instant, no client touching them afterwards, `DBSIZE` sampled every 50 ms. (`results/bench_expiry.jsonl`, `results/expiry.png`)
7. AOF replay time. Fill an AOF with 100,000, 400,000 and 1,600,000 SETs, then time startup replay. (`results/bench_replay.jsonl`)

`redis-cli` and `redis-benchmark` are not installed here, so there is no side-by-side run against real Redis. The Python client in `scripts/client_test.py` plays the interoperability role.

### Results

Pipelining is the whole story for throughput. At 50 connections, SET throughput goes from 107,160 ops/s at depth 1 to 999,599 at depth 16 and 2,942,811 at depth 128, a 27x gain with no change to the server (`results/bench_pipeline.jsonl`). Throughput per server CPU second tracks wall-clock throughput within a few percent at every depth (112,360 vs 107,160 at depth 1; 2,941,176 vs 2,942,811 at depth 128), which says the single server thread was the bottleneck, not the load generator. So one command costs the server about 9 us of CPU at depth 1 and about 0.34 us at depth 128. Nearly all of the cost at depth 1 is syscalls and wakeups, not the hash table.

Little's law checks out. With 50 connections at depth 1, the law predicts a mean latency of 50 / 107,160 s = 467 us; the measured p50 is 438 us. At depth 128 there are 6,400 requests in flight, predicting 2.2 ms against a measured p50 of 2.05 ms (`results/bench_pipeline.jsonl`). The p99 stays within about 2.5x of the p50 all the way up to depth 128 (4.4 ms against 2.05 ms), so the loop is fair: no connection is starved.

More connections do not help without pipelining. One connection does 40,193 ops/s at a p50 of 21.7 us. Throughput saturates at about 115,000 to 120,000 ops/s from 8 connections upward (115,697 at 8, 120,252 at 16, 112,758 at 256), and from then on each extra connection only adds queueing: p50 is 2.18 ms at 256 connections, again what Little's law predicts (256 / 112,758 s = 2.27 ms) (`results/bench_clients.jsonl`).

Reads and writes cost about the same. At 50 connections and depth 16, GET did 1,198,850 ops/s, SET 1,072,646 and a 50:50 mix 1,046,355 (`results/bench_workload.jsonl`). The spread between repetitions (SET ranged from 1,032,764 to 1,240,625) is larger than the gap between the workloads.

With many clients, `appendfsync always` was almost free. At 50 connections and depth 16, AOF off gave 1,217,265 ops/s and `always` gave 1,042,970, with `no` at 987,323 and `everysec` at 1,077,584 (`results/bench_aof.jsonl`). `no` coming out slowest shows these differences are mostly noise. The reason `always` is cheap is group commit: the loop does one `write()` and one `fsync()` per iteration, and with 50 busy connections one fsync covers every write in that iteration. Each file was 70.8 MB for 1.2 M SETs, 59 bytes per command.

With one client, `always` doubles latency. On a single connection at depth 1 there is nobody to share the fsync with. Median latency went from 22.2 us with AOF off to 59.0 us with `always`, and throughput from 40,120 to 14,165 ops/s (`results/bench_fsync1.jsonl`, via `results/summary.csv`). So a plain `fsync` on this machine costs roughly 35 us. That is cheap because macOS `fsync` does not flush the drive's own cache; `F_FULLFSYNC` would, and I did not measure it. Merely writing the AOF (`no`, `everysec`) cost 30.7 to 32.5 us p50, 8 to 10 us more than AOF off, which I attribute to the extra `write()` syscall per iteration.

Active expiry is fast when the machine is quiet. All 200,000 keys shared one deadline. `DBSIZE` was still 400,000 at 63 ms after the deadline, 294,740 at 116 ms and 200,000 at 171 ms, and stayed there (`results/bench_expiry.jsonl`). That is two 10 Hz cron cycles, each allowed at most 25 ms, reclaiming about 100,000 keys per cycle. Under heavy load the same experiment took until 791 ms after the deadline to finish, in eight uneven steps (`results/run1_loaded/bench_expiry.jsonl`, load average 222). The cycle budget is wall-clock time, so a descheduled server burns its budget without doing work.

Replay grows faster than linearly. Replaying 100,000 SETs (5.9 MB) took 24 ms, 400,000 (23.6 MB) took 121 ms and 1,600,000 (94.4 MB) took 854 ms (`results/bench_replay.jsonl`). Per command that is 240 ns, 302 ns and 534 ns. Wall and CPU time agree, so this is not disk. My guess, not yet tested, is that the hash table falls out of cache and that the all-at-once rehashes of `std::unordered_map` get more expensive as it grows. Reserving the table size before replay would test the second half of that guess.

Measurement conditions matter more than any tuning I did. The same pipeline experiment at load average 244 to 293 gave medians of 17,325 ops/s at depth 1 and 295,883 at depth 128, with p99 of 51.9 ms and 162.9 ms (`results/run1_loaded/bench_pipeline.jsonl`). That is 6x to 10x worse on throughput and 50x worse on tail latency than the quiet run, from exactly the same binary. Throughput per server CPU second degraded much less (92,593 vs 112,360 at depth 1), which is why I added that column.

### What I would change

Replace `std::unordered_map` with an incremental-rehash table like Redis's `dict`. The replay numbers suggest the one-shot rehash is already visible at a million keys, and in a live server it is a latency spike for every client.

Move the `everysec` fsync to a background thread, as Redis does. It is inline in the cron now, so a slow disk shows up as client latency.

Add AOF rewrite. The file only grows, including records for keys that were later deleted or expired.

Measure against real Redis. Installing `redis-server` and running the same `kvbench` sweep against both would say whether my numbers are good, not just internally consistent. `redis-benchmark` against `kvd` would also be the real interoperability test.

Build and run the epoll backend on Linux. It is written against the same interface but has never been compiled, because there was no Linux machine or container in this session. Until it runs, the abstraction is a claim, not a result.

Benchmark on a quiet machine from the start, and treat the load average as a first-class column. I lost the first full benchmark run to other jobs and only noticed because the p99 numbers were absurd.

### Verification

On 2026-09-27 an independent clean release rebuild passed all 50 tests (49 GoogleTest cases plus the end-to-end Python client test). The ASan/UBSan build and the benchmarks were not rerun independently. The benchmarks were measured at 1-minute load averages of 6 to 13, recorded per row, so treat them as indicative rather than definitive; the earlier run in `results/run1_loaded/` shows how much load distorts them. The epoll backend has never been compiled, and there was no comparison against real Redis.
