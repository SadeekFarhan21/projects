# kvd design

This document describes how the v0 server is put together, the data structures it relies on, the invariants the code maintains, and the trade-offs behind each choice. The code is small enough (about 1,900 lines of C++ in `src/`, headers and comments included) that each section points at the file that implements it.

## Architecture

One process, one thread. Everything below runs on the event-loop thread, so no component needs a lock and every command is atomic by construction.

```
                    TCP clients (redis-cli, kvbench, client_test.py, ...)
                         |  bytes in                         ^  bytes out
                         v                                   |
 +-----------------------------------------------------------------------------+
 | Server (server.cpp)                                                          |
 |                                                                              |
 |   Poller (poller.h)            Client {in buf, in_pos, out buf, out_pos}     |
 |   kqueue | epoll   --ready-->  on_readable: read() 16 KB into `in`           |
 |        ^                         |                                           |
 |        |                         v                                           |
 |        |                   process_input: loop                               |
 |        |                     parse_request (resp.cpp) -> argv                |
 |        |                     CommandProcessor::execute (commands.cpp)        |
 |        |                        |            |                               |
 |        |              reply bytes|            | propagate(argv') for writes   |
 |        |                        v            v                               |
 |        |                  Client.out     Aof buffer (aof.cpp)                |
 |        |                        |            |                               |
 |        |   before_sleep():  (1) Aof::flush  write(), fsync if "always"       |
 |        +--- write interest  (2) try_write   send() pending replies           |
 |             only on EAGAIN                                                   |
 |                                                                              |
 |   cron every 1000/hz ms: Store::active_expire_cycle, Aof::tick (everysec)    |
 +-----------------------------------------------------------------------------+
                         |                               |
                         v                               v
                 Store (store.cpp)                 appendonly.aof on disk
       unordered_map<key, Entry> + ttl_nodes_    (RESP arrays, replayed on start)
```

### One loop iteration

1. `before_sleep()` writes the AOF buffer to the file (and fsyncs it when the policy is `always`), then tries to `send()` every client's pending replies.
2. `poller.wait()` blocks until a socket is ready or the next cron tick is due. If some client still has unsent output that was not blocked by the kernel, the timeout is 0.
3. Events are dispatched. The listening socket accepts until `EAGAIN`. A readable client gets one `read()` of up to 16 KB, and then every complete command in its buffer is parsed and executed. A writable client gets its output flushed.
4. If the cron is due, it runs one active expiry cycle and the AOF `everysec` fsync check.

### Data flow of a write

`SET k v EX 10` arrives, is parsed into `["SET","k","v","EX","10"]`, executed against the `Store`, and produces `+OK\r\n` in the client's output buffer. The command also calls `propagate(["SET","k","v","PXAT","<now+10000>"])`, which appends that RESP array to the AOF buffer. At the end of the iteration the AOF buffer is written to the file first and the `+OK` is sent second.

## Key data structures

### Keyspace (`src/store.h`)

```
map_       : unordered_map<string, Entry>          Entry { string value; int64 expire_at_ms; size_t ttl_slot; }
ttl_nodes_ : vector<pair<const string, Entry>*>    dense list of the map nodes that have a TTL
```

`ttl_nodes_` exists so that active expiry can pick a uniformly random key *that has a TTL* in O(1). Redis gets the same effect from a second hash table (`db->expires`) and a random-bucket walk. I used a dense vector of node pointers instead, which is legal because `std::unordered_map` never moves its nodes. Rehashing invalidates iterators but not pointers or references to elements. Each entry records its own index in the vector (`ttl_slot`), so removing a TTL is a swap with the last element and a pop, O(1).

The map uses a transparent hash (`StringHash` with `is_transparent`) so lookups take a `string_view` without allocating a temporary `std::string`.

### Client buffers (`src/server.h`)

Each client owns an input string plus a consumed offset (`in_pos`) and an output string plus a sent offset (`out_pos`). Offsets instead of erasing the front on every command keep parsing and sending O(1) per command. The consumed prefix is dropped when the buffer is fully consumed (the common case, a plain `clear()`) or when it is more than half of the buffer and over 64 KB (amortized compaction).

### RESP parser (`src/resp.cpp`)

The parser is stateless. It is handed the unconsumed tail of the input buffer and returns `Ok` with the number of bytes used, `Incomplete`, or `Error`. A multibulk frame is validated in a first pass that records only offsets. Argument strings are copied only once the entire frame is present, so a 100 MB value that trickles in over many reads is scanned for headers many times but copied once. The inline form (`SET k v\r\n`) is also accepted, which is what makes `nc` usable for poking at the server.

### AOF (`src/aof.cpp`)

The file is a plain concatenation of RESP arrays, byte for byte what a client would have sent. Commands are rewritten before they are logged so that replay is independent of the time it happens at:

| Client sends | Logged as |
|---|---|
| `SET k v EX s` / `PX ms` / `EXAT s` | `SET k v PXAT <absolute ms>` |
| `SET k v NX` (succeeded) | `SET k v` |
| `EXPIRE k s`, `PEXPIRE k ms` | `PEXPIREAT k <absolute ms>` |
| `EXPIRE k <= 0` (key deleted) | `DEL k` |
| `INCR`, `DEL`, `PERSIST`, `FLUSHALL` | unchanged |
| failed or no-op writes (`DEL missing`, `SET NX` on an existing key) | nothing |

## Invariants

These are the properties the code relies on. The ones marked *tested* have a direct check in `tests/`.

1. For every key in `map_` with `expire_at_ms != -1`, `ttl_nodes_[entry.ttl_slot]` points back at that key's node, and `ttl_nodes_` has no other elements. (*tested* by `Store::check_invariants`, including a 20,000 step randomized test.)
2. A pointer in `ttl_nodes_` is removed before the node it points at is erased. Every erase goes through `Store::erase`, which calls `remove_ttl` first. (*tested* under ASan.)
3. An expired key is never observable. `find`, `pttl`, `EXISTS`, `GET` and `KEYS` treat a key whose deadline is `<= now` as missing, whether or not it has been physically deleted yet. (*tested*)
4. Replies on one connection come back in request order, and every request gets exactly one reply. This holds because a connection's commands are executed sequentially into a single output buffer. (*tested* with a 20,000 command pipeline.)
5. A reply to a write is not sent before the write has reached the AOF file with `write()`. `before_sleep` flushes the AOF before any client output. With `appendfsync always` it also fsyncs first, so an acknowledged write survives `kill -9`. (*tested* by the Python end-to-end test, which kills the server with SIGKILL and checks all 200 acknowledged keys.)
6. Replaying the AOF at any later time reproduces the same keys and the same absolute deadlines. Keys whose deadline passed while the server was down are purged right after replay. (*tested*)
7. A connection whose unsent output exceeds `out_soft_limit` (1 MB) is not read from until that output drains, and on resume any commands already buffered are processed immediately rather than waiting for a new read event. (*tested*)

## Trade-offs chosen

### Single thread

No locks, commands are atomic, and the code stays short. The cost is that one core caps throughput and any slow command (`KEYS` on a large keyspace, a big fsync) stalls every client. This is the Redis bet, and the point of v0 is to measure where it breaks.

### Level-triggered readiness, one read per event

With edge-triggered mode the handler must drain the socket to `EAGAIN` or it can miss data forever. Level-triggered with a single 16 KB read per wakeup is simpler and fairer, since one busy client cannot monopolize an iteration.

### Write first, register interest only on `EAGAIN`

Replies are sent directly from `before_sleep`. Write interest is only added to the poller when the kernel buffer is full, so the common path costs no extra `kevent`/`epoll_ctl` syscalls.

### Fsync once per loop iteration (group commit)

With `appendfsync always`, `before_sleep` issues one `write()` and one `fsync()` for everything the iteration produced, before any reply goes out. With 50 busy connections one fsync covers dozens of writes, so `always` measured within noise of AOF off (results/summary.csv, experiment `aof`). With a single connection there is nothing to share it with and the median latency goes from 22 us to 59 us (experiment `fsync1`).

### Absolute deadlines in the AOF

Relative TTLs replayed later would extend every key's life by the downtime.

### Active expiry modelled on Redis

Sample 20 keys with a TTL, delete the expired ones, repeat while more than 25% of the sample was expired, stop at a time budget of 25% of a cron tick. Memory held by expired keys is bounded without a full scan, and the cycle cannot stall the loop for more than 25 ms at `hz 10`.

### Torn tail truncation, hard failure elsewhere

A crash mid-write leaves a partial last record, which is safe to drop. Garbage in the middle means something else went wrong, and refusing to start is better than silently losing everything after it.

### Backpressure by pausing reads

Redis disconnects clients whose output buffer passes a hard limit. Pausing reads instead lets a fast pipelining client make progress at the speed it drains replies, with the output buffer bounded by the soft limit plus one reply, because the limit is checked before every command.

## Trade-offs rejected for v0

### Timer wheel or heap for expiry

Exact expiry with a min-heap costs O(log n) per TTL update and a lot of memory for entries that are later overwritten. Sampling is approximate but O(1) per operation.

### Incremental rehashing hash table (Redis `dict`)

`std::unordered_map` rehashes all at once, which means a latency spike each time the table doubles (millions of keys moved in one go). Writing a two-table incremental dict is the right fix but it is a milestone on its own.

### Background fsync thread

Redis runs `everysec` fsyncs on a background thread so a slow disk does not block the loop. Here the fsync runs inline in the cron. It is simple and correct, but a slow fsync shows up as a latency spike for every client.

### `F_FULLFSYNC` on macOS

macOS `fsync` does not flush the drive's own write cache. `F_FULLFSYNC` does, at a much higher cost. I kept plain `fsync`, which matches Redis on Linux, and documented the gap.

### Streaming AOF replay

Replay reads the whole file into memory first. That doubles peak memory during startup for large files, but keeps the loader to a few lines.

### Multithreaded I/O, AOF rewrite, RDB snapshots

Listed as milestones in the README.

