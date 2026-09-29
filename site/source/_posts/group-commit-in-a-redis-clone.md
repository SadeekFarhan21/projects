---
layout: post
title: "Group Commit in a Single-Threaded Redis Clone"
tags:
  - databases
  - storage
  - cpp
description: >-
  A single-threaded Redis-compatible server in C++ with TTLs, pipelining and an
  append-only file. One fsync per event-loop pass makes durability nearly free
  with many clients.
date: 2026-09-29 02:17:09
---


I wrote kvd, a single-threaded TCP server that speaks a subset of RESP2, the Redis wire protocol<sup>[[1]](#ref-1)</sup>. It stores strings with optional TTLs, executes pipelined requests, expires keys lazily and actively the way Redis does<sup>[[2]](#ref-2)</sup>, and logs every write to an append-only file (AOF)<sup>[[3]](#ref-3)</sup> that it replays on startup, including after a `kill -9` or a torn final record. A pipelined C++ load generator ships with it.

The server is 1,922 lines of C++20, headers and comments included. There are **49 GoogleTest cases** plus an end-to-end test written in stdlib-only Python against the real binary, and on 2026-09-27 an independent clean release rebuild passed all **50**. The AddressSanitizer<sup>[[4]](#ref-4)</sup> plus UBSan build also passed in my own runs, but it was not rerun independently, and neither were the benchmarks.

The headline measurement is how little the server's own work matters. At 50 connections, SET throughput went from **107,160 ops/s** with one request in flight per connection to **2,942,811 ops/s** with 128 in flight, a factor of 27 with no change to the server. Server CPU per command fell from about 9 µs to about 0.34 µs over that range, which says nearly all the cost at depth 1 is system calls and wakeups, not the hash table. All benchmark numbers were measured on a shared Apple M4 Pro that was also running other jobs, with the 1 minute load average recorded in every row (7 to 13 for the reported runs). They are indicative, not a clean benchmark.

Two gaps should be stated up front. The epoll backend exists as source but has **never been compiled**, because there was no Linux machine in the session, so the Linux half of the poller abstraction is a claim rather than a result. And there was **no comparison against real Redis**, because neither `redis-server` nor `redis-benchmark` was installed. Every number below is internally consistent, and none of them says whether kvd is fast relative to the thing it imitates.

The argument of this post is that a single-threaded server's speed is decided by how many system calls each command shares, and that the hardest thing to keep honest was the measurement rather than the code. Skip to [Problems](#problems) for the bugs.

## What I Wanted to Build

Redis runs every command on one thread and is still fast enough that most people never need anything else. I wanted to understand why by building the core of it and measuring where it breaks.

The v0 scope was fixed before writing code. One event loop on kqueue, with an epoll path behind the same interface. RESP2 with pipelining and the inline form that makes `nc` usable. `GET`, `SET` with every expiry and condition flag, `DEL`, `EXISTS`, the `EXPIRE` family, `TTL`, `INCR` and friends, and `KEYS` with glob patterns. Lazy and active expiry modelled on Redis. An AOF with three fsync policies, replay and torn-tail recovery. And a load generator, so the design choices could be measured rather than asserted.

Multithreaded I/O, RDB snapshots, AOF rewrite, lists and hashes and sorted sets, pub/sub and replication were left for later milestones. `COMMAND` and `CONFIG GET` are stubs that return empty arrays, so a `redis-cli` handshake should succeed, though I could not check that here.

## Theory

### Why One Thread Can Be Enough

A request/response server spends most of its life waiting on the network. With a thread per connection the waiting is done by blocked threads, and the cost is context switches and stack memory. An event loop inverts that. One thread asks the kernel which sockets are ready (kqueue<sup>[[5]](#ref-5)</sup> on macOS, epoll<sup>[[6]](#ref-6)</sup> on Linux) and touches only those. As long as every command is short, one core can serve thousands of connections, and because only one thread ever touches the data, every command is atomic without a lock.

The catch is the word short. One slow command, a `KEYS` over a million keys or an fsync on a slow disk, stalls every client at once.

### Where the Time Goes

A `GET` that finds its key in a hash table costs well under a microsecond. The `read()`, `send()` and `kevent()` calls around it cost several. So the throughput of a server like this is set mostly by how many system calls each command pays for.

Pipelining attacks exactly that<sup>[[7]](#ref-7)</sup>. If a client sends 16 commands before reading any replies, the server picks them up with one `read()`, executes them back to back, and answers with one `send()`. The syscall cost is shared 16 ways.

### Little's Law as a Sanity Check

In a closed-loop benchmark, where each connection keeps a fixed number of requests outstanding, Little's law<sup>[[8]](#ref-8)</sup> ties the three quantities together.

$$
\text{requests in flight} = \text{throughput} \times \text{mean latency}
$$

With 50 connections and one request in flight each, a server doing 100,000 requests per second must show a mean latency of $50 / 100{,}000\ \text{s} = 500\ \mu\text{s}$, however fast each individual command is. Latency in a saturated closed-loop benchmark is set by the queue, not by the work. I used this to check every result below.

### Expiry Without a Timer per Key

A key with a TTL must disappear on time, but nobody wants a timer per key. Redis combines two cheap mechanisms<sup>[[2]](#ref-2)</sup>. Lazy expiry checks the deadline whenever a key is touched, so an expired key is never visible to a client. Active expiry runs about ten times a second, samples 20 random keys that have a TTL, deletes the expired ones, and repeats while more than a quarter of the sample was expired. That bounds how much memory dead but untouched keys can hold, at a cost proportional to the garbage rather than to the keyspace.

### Durability Is a Policy

The AOF is a log of write commands in the same RESP format clients send, and replaying it rebuilds the dataset. The durability knob is when to call `fsync`<sup>[[3]](#ref-3)</sup>. `always` means before acknowledging a write, `everysec` means once a second, and `no` leaves it to the operating system.

## Architecture

Everything runs on one thread.

<figure class="excal" data-diagram="redis-event-loop"><a href="/img/diagrams/redis-event-loop.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/redis-event-loop.webp" alt="The single-threaded event loop of the Redis clone: commands fill an AOF buffer and per-client reply buffers, and before_sleep runs Aof::flush before try_write on every iteration, so no reply leaves before its write is durable." width="2400" height="3554" loading="lazy" decoding="async"></a></figure>

One loop iteration does four things in order.

1. Before going to sleep, the loop writes the AOF buffer to the file, fsyncs it under `always`, and only then tries to `send()` every client's pending replies.
2. The poller blocks until a socket is ready or the next cron tick is due. If a client still has unsent output that the kernel did not refuse, the timeout is zero.
3. Events are dispatched. The listening socket accepts until `EAGAIN`. A readable client gets one `read()` of up to 16 KB, and every complete command in its buffer is parsed and executed.
4. If the cron is due, it runs one active expiry cycle and the `everysec` fsync check.

The poller is a four-method interface with kqueue and epoll implementations behind it, so the server never includes a platform readiness header.

## Implementation

### The Loop That Makes Pipelining Work

Pipelining is not a feature the server implements so much as a property of this loop. Every complete frame in the input buffer is executed in one pass, and all the replies land in one output buffer that goes out in one `send()`.

```cpp
void Server::process_input(Client& c) {
    int64_t now = wall_ms();  // one clock read per batch, like Redis's cached mstime
    while (!c.close_after_write && c.in_pos < c.in.size()) {
        if (c.out.size() - c.out_pos > cfg_.out_soft_limit) {
            c.read_paused = true;               // backpressure, see below
            poller_->set_read(c.fd, false);
            ++stats_.read_pauses;
            break;
        }
        ParseResult r = parse_request(std::string_view(c.in).substr(c.in_pos), args_);
        if (r.status == ParseStatus::Incomplete) break;
        // ... protocol errors reply and close ...
        c.in_pos += r.consumed;
        if (args_.empty()) continue;
        if (proc_.execute(args_, now, c.out) == CommandProcessor::Action::Close)
            c.close_after_write = true;
    }
    // compact `in`, then queue the client for the pre-sleep flush if it has output
}
```

Each client carries an input buffer with a consumed offset and an output buffer with a sent offset. Advancing an offset is $O(1)$. Erasing the front of a `std::string` after every command would make a deep pipeline quadratic in its length, so the input buffer is cleared when fully consumed (the common case) and compacted only when the dead prefix is over 64 KB and more than half the buffer.

### A Parser That Copies Once

The RESP parser is stateless. It is handed the unconsumed tail of the input buffer and returns complete, incomplete or error. For a multibulk frame it makes a first pass that records only offsets, and copies arguments into `std::string`s only once the whole frame is present.

```cpp
struct Span { size_t off, len; };
std::vector<Span> spans;
for (int64_t k = 0; k < n; ++k) {
    // ... find "$<len>\r\n", validate len ...
    size_t need = data + static_cast<size_t>(len) + 2;
    if (buf.size() < need) return {};          // incomplete, nothing copied
    spans.push_back({data, static_cast<size_t>(len)});
    pos = need;
}
args.resize(spans.size());
for (size_t k = 0; k < spans.size(); ++k) args[k].assign(buf.data() + spans[k].off, spans[k].len);
```

An 8 MB value that trickles in across hundreds of reads is scanned for headers many times but copied once. There is a test that sends exactly that.

### A Dense Vector of TTL Keys

Active expiry needs a uniformly random key among those with a TTL, in $O(1)$. Redis gets that from a second hash table of expiring keys and a random bucket walk. I kept a `std::vector` of raw pointers to the `unordered_map` nodes that have a TTL, with each entry storing its own index into that vector.

```cpp
void Store::remove_ttl(Node& node) {
    Entry& e = node.second;
    if (e.expire_at_ms == kNoExpiry) return;
    size_t slot = e.ttl_slot;
    Node* last = ttl_nodes_.back();
    ttl_nodes_[slot] = last;          // swap the last pointer into our slot
    last->second.ttl_slot = slot;
    ttl_nodes_.pop_back();
    e.expire_at_ms = kNoExpiry;
}

void Store::erase(Map::iterator it) {
    remove_ttl(*it);  // must happen before erase, ttl_nodes_ points into the node
    map_.erase(it);
}
```

This is legal only because `std::unordered_map` never moves its nodes. Rehashing invalidates iterators but not pointers or references to elements<sup>[[9]](#ref-9)</sup>. The price of that guarantee is that the map is node-based, one allocation per key, which comes back in the replay results. An invariant checker walks the map and checks that every TTL node is in the vector at the slot it records and that the counts match, and a randomized test runs it after each of 20,000 mixed operations.

The map also uses a transparent hash, so a lookup by `string_view` does not allocate a temporary `std::string`.

### Active Expiry with a Wall-Clock Budget

```cpp
while (!ttl_nodes_.empty()) {
    size_t n = std::min(kSampleSize, ttl_nodes_.size());   // 20
    size_t expired_this_round = 0;
    for (size_t k = 0; k < n && !ttl_nodes_.empty(); ++k) {
        Node* node = ttl_nodes_[pick(rng_)];
        if (node->second.expire_at_ms <= now_ms) {
            erase(map_.find(node->first));
            ++expired_this_round;
        }
    }
    if (expired_this_round * 4 <= n) break;                 // at most 25% expired
    if (elapsed_us() >= budget_us) break;                   // 25% of a cron tick
}
```

At the default 10 Hz the budget is 25 ms per cycle, so expiry cannot stall the loop for longer than that. The budget is measured in wall-clock time, which matters later.

### Logs That Do Not Depend on When They Are Replayed

A relative TTL replayed a day later would extend every key's life by the downtime. So each write is rewritten before it is logged.

| Client sends | Logged as |
|---|---|
| `SET k v EX s`, `PX ms` or `EXAT s` | `SET k v PXAT <absolute ms>` |
| `SET k v NX` that succeeded | `SET k v` |
| `EXPIRE k s`, `PEXPIRE k ms` | `PEXPIREAT k <absolute ms>` |
| `EXPIRE k` with a deadline already past | `DEL k` |
| `DEL missing`, `SET NX` on an existing key | nothing |

Keys whose deadline passed while the server was down are purged in one sweep right after replay. Replay runs through a command processor with no propagation hook, so replayed commands are not appended to the file again.

### Reply After Log, and Group Commit

The ordering of the step that runs just before the loop sleeps is the durability guarantee.

```cpp
void Server::before_sleep() {
    if (aof_ && !aof_->flush() && !aof_error_logged_) { /* log once, retry next time */ }
    std::vector<int> batch;
    batch.swap(pending_);
    for (int fd : batch) { /* ... */ try_write(c); }
}
```

The AOF flush does one `write()` of everything this iteration produced, then one `fsync()` if the policy is `always`, and only then does any reply leave the process. So under `always` an acknowledged write is on disk (in the sense macOS `fsync` means, discussed below). The end-to-end test checks it directly by writing 200 keys with `appendfsync always`, killing the server with `SIGKILL`, restarting it and finding all 200.

Because the fsync is per loop iteration and not per command, it is also a group commit<sup>[[10]](#ref-10)</sup>. With 50 busy connections, one fsync covers every write any of them made in that iteration. That turns out to be the most important performance property of the AOF.

A torn final record, from a crash in the middle of a `write()`, is truncated on startup, which is what Redis does with `aof-load-truncated yes`<sup>[[3]](#ref-3)</sup>. Garbage anywhere else in the file stops the server from starting, on the view that corruption in the middle means something else went wrong and silently dropping everything after it is worse than refusing to run.

### Backpressure by Pausing Reads

A client that pipelines faster than it reads replies makes its output buffer grow without bound. Redis disconnects such a client at a hard limit<sup>[[11]](#ref-11)</sup>. I stop reading from it once its unsent output passes a soft limit, 1 MB by default, and resume when it drains. Because the check runs before every command, the output buffer is bounded by the soft limit plus one reply, and a fast pipelining client still makes progress at the speed it drains. The resume path has a subtlety that is in [Problems](#problems).

Two smaller choices keep the common path cheap. Write interest is registered with the poller only when `send()` hits `EAGAIN`, so a normal reply costs no extra `kevent` call. And readiness is level-triggered with one 16 KB read per wakeup, which is simpler than draining every socket under edge triggering<sup>[[6]](#ref-6)</sup> and fairer, since one busy client cannot monopolize an iteration.

### Testing

Unit tests cover the parser, the glob matcher, the store and command semantics. In-process TCP tests run the real server on an ephemeral port and cover a 20,000 command pipeline, an 8 MB value, 16 clients doing 2,000 `INCR`s each, backpressure and AOF restart. An independent stdlib-only Python RESP client drives the real binary through every command, real-time expiry, `SIGTERM`, `SIGKILL` recovery and a torn tail, standing in for `redis-cli`.

## Problems

### 1. Apple clang's Sanitizer Runtime Hangs

The Debug build with AddressSanitizer and UBSan hung at startup before reaching `main`. It is not my code. An empty `int main(){return 0;}` built with Apple clang 17 and ASan enabled hangs the same way on this macOS 26.5 machine, and it was still hanging when I rechecked at the end, killed by a 15 second timeout.

The fix was to build the sanitizer configuration with Homebrew LLVM's clang and keep Apple clang for the release build. Both builds passed the same 50 tests in my runs. The independent rebuild covered only the release build.

### 2. A Backpressure Test That Passed in One Build and Failed in the Other

The first version of the backpressure test relied on the default 1 MB soft limit being crossed. Whether it was crossed depended on how fast the kernel drained socket buffers relative to how fast the server produced replies, which differed between the fast optimized build and the slow ASan build. The assertion that reads had been paused was not deterministic.

The test now sets a 64 KB soft limit and pipelines 30,000 `GET`s of a 1 KB value, about 30 MB of replies, before reading anything. With the limit that far below the reply volume, the pause always happens regardless of build speed. It then reads all 30,000 replies, checks each one, and checks the pause counter is nonzero.

### 3. A Paused Client Can Stall Forever

This one is a hazard the design had to close rather than a bug I watched happen, but it is the kind that would only show up as a hung client in production. When a client is paused, its input buffer may already hold complete commands that were read but not executed. When its output drains, re-enabling read interest is not enough. The socket may have nothing new to read, so no readable event ever arrives, and the buffered commands sit there forever.

So the resume path runs them itself.

```cpp
if (!c.read_paused) return true;
// Output drained. Its input buffer may already hold complete commands that
// no new read event will ever announce, so process them now.
c.read_paused = false;
poller_->set_read(c.fd, true);
process_input(c);
if (c.out.empty()) return true;
// otherwise loop and try to send what that produced
```

The backpressure test exercises it, since the client in that test sends its whole pipeline before reading and the server necessarily pauses with commands still buffered.

### 4. Benchmarking on a Machine Twenty Times Oversubscribed

The first full benchmark run happened while other heavy jobs were running. Its own records show a 1 minute load average of 364.7 at the start, on 14 cores. The numbers looked like results and were not.

<figure data-figure="chart:projects/redis-from-scratch/redis-from-scratch-loaded"></figure>

SET at 50 connections without pipelining gave a median of 17,325 ops/s with a p99 of 51.9 ms, against 107,160 ops/s and 935.5 µs once the machine was quieter. The instability was as telling as the level. The three repetitions of the mixed-workload experiment in the loaded run gave 27,832, 40,969 and 480,780 ops/s, a factor of 17 between runs of the same thing.

I noticed because the p99 numbers were absurd. Two changes made the data usable. First, every row now records the load average before and after the run and the CPU time the server process used, read from `ps`. Throughput per server CPU second is a number that load degrades much less, because it measures what the server did while it had a core rather than how often it got one. At depth 1 it was 92,593 in the loaded run against 112,360 in the quiet one. When the server is saturated, wall-clock throughput and throughput per CPU second should agree, which is a quick check that the server, not the load generator, was the bottleneck. Second, when the load dropped I reran everything one experiment at a time, with the load generator capped at 4 client threads so the benchmark itself added little load. The loaded run is kept alongside the clean one rather than deleted.

The CPU time comes from `ps`, which has 10 ms resolution. The shortest runs used 0.33 s of server CPU, so rounding alone can move a per CPU second figure by about 3 percent. It is a coarse instrument.

### 5. The Expiry Experiment Raced Itself

The expiry experiment loads 200,000 keys that all expire at one absolute instant, via `PXAT`, and then watches `DBSIZE`. If the deadline is chosen before encoding and sending the keys, a slow machine can reach the deadline while keys are still loading, and the drop smears across the load.

The experiment now encodes every command first, measures how long loading 200,000 persistent keys took, and sets the deadline to twice that plus 2 seconds. It records the margin it actually achieved, 1,779 ms before the deadline in the reported run.

### 6. A Run Too Short to Measure

My first single-connection fsync experiment used 20,000 requests per run. It showed the effect, but the AOF-off baseline swung so much between repetitions that the noise was larger than the thing being measured. I raised it to 100,000 requests per run and discarded the short run. Even at 100,000 requests the AOF-off runs ranged from 27,562 to 44,434 ops/s, which is worth keeping in mind below.

### 7. The Session Was Interrupted

Work stopped partway through to reduce CPU load on the shared machine. The code, tests and loaded run survived, and on resuming I rebuilt both configurations with limited parallelism, ran the two test suites one at a time, and redid the benchmarks as described above.

## Experiments

All experiments ran on an Apple M4 Pro (14 cores, 48 GB) over loopback TCP, with the optimized release build (Apple clang 17), the load generator using 4 client threads, and 3 repetitions per configuration. I report medians of each metric over the three runs. Every run is recorded as one row of raw data. The 1 minute load average during the reported runs was between 7.2 and 12.9, recorded per row.

The load generator is closed-loop. Each connection keeps up to $P$ requests in flight, topping the window back up as replies arrive. Latency is measured from the moment a request's batch was written to the moment its reply was parsed, so at $P > 1$ it includes time queued behind earlier requests in the same pipeline. That is the latency a pipelining client actually sees.

1. **Pipeline depth.** 50 connections, SET with 16 B values, depth 1 to 128, AOF off.
2. **Connection count.** Depth 1, SET, 1 to 256 connections, AOF off.
3. **Workload mix.** 50 connections, depth 16, 100,000 keys, pure SET against pure GET against a 50/50 mix.
4. **Persistence with many clients.** 50 connections, SET at depth 1 and 16, with AOF off and with `appendfsync` set to `no`, `everysec` and `always`.
5. **Persistence with one client.** 1 connection, depth 1, 100,000 SETs, the same four modes.
6. **Active expiry.** 200,000 persistent keys plus 200,000 keys sharing one deadline, no client touching them afterwards, `DBSIZE` sampled about every 50 ms.
7. **AOF replay.** Fill an AOF with 100,000, 400,000 and 1,600,000 SETs, then time replay at startup.

There is no side-by-side run against real Redis. The end-to-end Python client plays the interoperability role, which checks correctness against the protocol as I understood it, not against Redis's behavior.

## Results

### Pipelining Is the Whole Story for Throughput

<figure data-figure="chart:projects/redis-from-scratch/redis-from-scratch-pipeline"></figure>

At 50 connections SET throughput went from 107,160 ops/s at depth 1 to 999,599 at depth 16 and 2,942,811 at depth 128, with no change to the server. Throughput per server CPU second tracks wall-clock throughput within about 5 percent at every depth, 112,360 against 107,160 at depth 1 and 2,941,176 against 2,942,811 at depth 128. The single server thread was the bottleneck throughout, not the load generator.

That makes the per-command cost easy to read off. At depth 1 one SET costs the server about $1 / 112{,}360\ \text{s} \approx 8.9\ \mu\text{s}$ of CPU. At depth 128 it costs about 0.34 µs. The hash table insert did not get 26 times cheaper. What changed is how many commands share each `read()`, `send()` and `kevent()`.

The spread between runs grows with depth. At depth 64 the three runs gave 1,187,745, 2,276,344 and 2,631,208 ops/s, so the curve's exact shape above depth 16 should not be read closely. The p99 at depth 16 (1.74 ms) is also higher than at depth 32 (1.48 ms), which is the kind of wobble to expect from three runs on a shared machine.

### Little's Law Checks Out

With 50 connections at depth 1, the law predicts a mean latency of $50 / 107{,}160\ \text{s} = 467\ \mu\text{s}$, and the measured p50 is 438.0 µs. At depth 128 there are 6,400 requests in flight, predicting 2.17 ms against a measured p50 of 2.05 ms. The law is about the mean and I am comparing against the median, so close agreement is what I would expect rather than a proof, but a load generator that miscounted in-flight requests or timestamps would fail this check.

The p99 stayed between 1.7 and 2.4 times the p50 at every depth, 4.39 ms against 2.05 ms at depth 128. No connection was being starved, which is what level-triggered, one-read-per-event dispatch is supposed to buy.

### More Connections Do Not Help Without Pipelining

One connection did 40,193 ops/s at a p50 of 21.7 µs. Its throughput per server CPU second was 111,111, almost three times the wall-clock rate, which says the server was idle most of the time, waiting for the client's round trip. Throughput rose to 115,697 at 8 connections and 120,252 at 16, and then stayed between 104,740 and 114,065 all the way to 256. Past that point each extra connection only adds queueing. The p50 at 256 connections was 2.18 ms, again what Little's law predicts ($256 / 112{,}758\ \text{s} = 2.27\ \text{ms}$).

So at depth 1 the server saturates at around 115,000 to 120,000 SETs per second on this machine, and the only way past that is to put more commands in each system call.

### Reads and Writes Cost About the Same

At 50 connections and depth 16, GET did 1,198,850 ops/s, SET 1,072,646 and the 50/50 mix 1,046,355. The spread between repetitions is larger than the gap between workloads. SET alone ranged from 1,032,764 to 1,240,625 and GET from 940,801 to 1,269,237.

### With Many Clients, appendfsync always Was Almost Free

At 50 connections and depth 16, AOF off gave 1,217,265 ops/s and `always` gave 1,042,970, with `no` at 987,323 and `everysec` at 1,077,584. `no` coming out slowest, when it does strictly less work than `always`, shows that the differences between modes here are mostly noise. At depth 1 the four modes landed between 112,334 and 118,052.

The reason `always` is cheap is group commit. With 50 busy connections one fsync per loop iteration covers every write made in that iteration, so the fsync cost is divided among dozens of commands. Each AOF was 70.8 MB for 1.2 million SETs across the depth 1 and depth 16 runs, 59 bytes per logged command.

### With One Client, always Nearly Triples Latency

<figure data-figure="chart:projects/redis-from-scratch/redis-from-scratch-fsync"></figure>

On a single connection at depth 1 there is no one to share the fsync with. Median latency went from 22.2 µs with AOF off to 59.0 µs with `always`, and throughput from 40,120 to 14,165 ops/s. Against the `no` mode, which also writes the file but never fsyncs on the reply path, the gap is 26.5 µs, so a plain `fsync` on this machine costs a few tens of microseconds.

That is cheap because macOS `fsync` pushes data to the drive but does not flush the drive's own write cache. `F_FULLFSYNC` would<sup>[[12]](#ref-12)</sup>, at a much higher cost, and I did not measure it. So `always` here protects against a process crash, which the `SIGKILL` test demonstrates, and not against power loss.

Writing the AOF without an fsync on the reply path (`no` and `everysec`) put the p50 at 30.7 to 32.5 µs, 8 to 10 µs above AOF off, which I would attribute to the extra `write()` per iteration. I hold that loosely. In the first repetition AOF off measured 28.0 µs and `no` 29.5 µs, almost no gap, and the per-run medians of the AOF-off mode ranged from 21.5 to 28.0 µs.

### Active Expiry Is Fast When the Machine Is Quiet

<figure data-figure="chart:projects/redis-from-scratch/redis-from-scratch-expiry"></figure>

All 200,000 TTL keys shared one deadline and no client touched them, so only active expiry could remove them. `DBSIZE` was still 400,000 at 63 ms after the deadline, 294,740 at 116 ms and 200,000 at 171 ms, and stayed there. The server's shutdown log confirms 200,000 keys were expired actively.

That is two 10 Hz cron cycles, the first removing about 105,000 keys and the second the remaining 95,000. Each cycle is allowed at most 25 ms. If each one ran to its budget, which is my inference and was not measured, deleting a sampled expired key costs about 0.24 µs.

Under heavy load the same experiment took until 791 ms after the deadline to finish, in eight uneven steps, one of which removed only 20 keys, at a load average of 222. The cycle budget is wall-clock time, so a server that gets descheduled in the middle of a cycle burns its budget without doing work. That bounds how long clients wait, but it means expiry slows down exactly when the machine is busiest.

### Replay Grows Faster Than Linearly

Replaying 100,000 SETs (5.9 MB) took 24 ms, 400,000 (23.6 MB) took 121 ms and 1,600,000 (94.4 MB) took 854 ms. Per command that is 240 ns, 302 ns and 534 ns. Wall time and CPU time agree to within a few milliseconds (850 ms of CPU for the largest), so this is not waiting on disk.

My guess, which I have not tested, is some mix of two things. The hash table and its node allocations fall out of cache as the keyspace grows, and `std::unordered_map`'s all-at-once rehashes get more expensive as the table grows. Calling `reserve` before replay would test the second half of that.

The loaded run makes a separate point about wall time. At load average 251 to 257, the three 1.6 million command replays took 1,685, 4,785 and 3,601 ms of wall time but 1,248, 1,241 and 1,233 ms of CPU. The server's own work barely changed. The scheduler added the rest. That is why the startup log prints both.

### Measurement Conditions Mattered More Than Any Tuning I Did

Load cost the pipeline experiment 5.5 to 10 times its throughput and 37 to 55 times its p99 (the loaded chart above). Nothing I changed in the code moved any number by that much.

## What I Would Change

### Replace the Standard Hash Map with an Incremental-Rehash Table

Redis's `dict` keeps two tables during a resize and moves a few buckets per operation<sup>[[13]](#ref-13)</sup>, so no single command pays for the whole rehash. `std::unordered_map` rehashes everything at once, which is a latency spike for every client each time the table doubles. The replay numbers suggest something that scales badly is already visible at a million keys, and an open-addressing table would also drop the per-key node allocation. The dense TTL vector relies on node stability, so it would need to store keys or indices instead of pointers.

### Move the everysec fsync off the Loop

It runs inline in the cron now, so a slow disk shows up as client latency. Redis does it on a background thread<sup>[[3]](#ref-3)</sup>.

### Add AOF Rewrite

The file only grows, including records for keys that were later deleted or expired. A rewrite that snapshots the live keyspace into a fresh file, with a `fork()` for copy-on-write, is the v2 milestone.

### Measure Against Real Redis

Installing `redis-server` and running the same load-generator sweep against both would say whether these numbers are good, not just consistent with each other. Running `redis-benchmark` and `redis-cli` against kvd would be the real interoperability test that the Python client only stands in for.

### Build and Run the epoll Backend

It is written against the same interface but has never been compiled. Until it runs on Linux, the portable poller is a claim, not a result.

### Benchmark on a Quiet Machine from the Start

I lost the first full benchmark run to other jobs. Recording load and server CPU per row from the first run, and checking them before trusting a number, would have saved a rerun.

## Reproducibility

The build needs CMake 3.24 or newer, Ninja and a C++20 compiler, and GoogleTest is fetched at configure time. There are two configurations, an optimized release build and a Debug build with AddressSanitizer and UBSan, and the sanitizer build uses Homebrew LLVM because Apple clang 17's ASan runtime hangs on macOS 26.5. Each configuration runs the 49 GoogleTest cases and the end-to-end Python client test. The benchmark sweep runs every configuration three times with the load generator on 4 client threads, and every row it records carries the load average and the server's CPU seconds, so a reader can tell a clean run from a loaded one.

Find the code at [github.com/SadeekFarhan21/projects/kvd-resp-server](https://github.com/SadeekFarhan21/projects/tree/main/kvd-resp-server).

## References

1. <span id="ref-1"></span>Redis. *Redis serialization protocol specification*. Redis documentation. [link](https://redis.io/docs/latest/develop/reference/protocol-spec/)
2. <span id="ref-2"></span>Redis. *EXPIRE*, appendix "How Redis expires keys". Redis documentation (redis-doc repository). [link](https://github.com/redis/redis-doc/blob/master/commands/expire.md)
3. <span id="ref-3"></span>Redis. *Redis persistence*. Redis documentation. [link](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)
4. <span id="ref-4"></span>Konstantin Serebryany, Derek Bruening, Alexander Potapenko et al. *AddressSanitizer: A Fast Address Sanity Checker*. USENIX Annual Technical Conference (USENIX ATC), 2012. [link](https://www.usenix.org/conference/atc12/technical-sessions/presentation/serebryany)
5. <span id="ref-5"></span>Jonathan Lemon. *Kqueue: A Generic and Scalable Event Notification Facility*. USENIX Annual Technical Conference, FREENIX Track, 2001. [link](https://www.usenix.org/conference/2001-usenix-annual-technical-conference/kqueue%E2%80%93-generic-and-scalable-event-notification)
6. <span id="ref-6"></span>Linux man-pages project. *epoll(7): I/O event notification facility*. Linux manual page. [link](https://man7.org/linux/man-pages/man7/epoll.7.html)
7. <span id="ref-7"></span>Redis. *Redis pipelining*. Redis documentation. [link](https://redis.io/docs/latest/develop/using-commands/pipelining/)
8. <span id="ref-8"></span>John D. C. Little. *A Proof for the Queuing Formula: L = λW*. Operations Research 9(3), 1961. [doi:10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383)
9. <span id="ref-9"></span>cppreference.com. *std::unordered_map* (iterator invalidation). C++ reference. [link](https://en.cppreference.com/w/cpp/container/unordered_map.html)
10. <span id="ref-10"></span>David J. DeWitt, Randy H. Katz, Frank Olken et al. *Implementation Techniques for Main Memory Database Systems*. Proceedings of the 1984 ACM SIGMOD International Conference on Management of Data, 1984. [doi:10.1145/602259.602261](https://doi.org/10.1145/602259.602261)
11. <span id="ref-11"></span>Redis. *Redis client handling* (output buffer limits). Redis documentation. [link](https://redis.io/docs/latest/develop/reference/clients/)
12. <span id="ref-12"></span>Apple. *fsync(2)*. Mac OS X Manual Pages, Apple Developer Documentation Archive. [link](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html)
13. <span id="ref-13"></span>Redis. *dict.c: in-memory hash tables with incremental rehashing*. Redis source code. [link](https://github.com/redis/redis/blob/unstable/src/dict.c)
