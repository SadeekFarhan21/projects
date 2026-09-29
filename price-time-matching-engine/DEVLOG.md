# Building an exchange matching engine from scratch

### What I wanted to build

The heart of every exchange is a matching engine: a program that holds resting buy and sell orders, decides which ones trade when a new order arrives, and tells the world what happened. I wanted to build one the way a real venue would, small enough to finish in one sitting but with the properties that matter in production.

The v0 goals were concrete. One symbol. A limit order book with price-time priority. Five order types (limit, market, IOC, FOK, post-only), plus cancel and modify, where a modify that increases size loses its place in the queue. Integer prices and quantities. An event-sourced design, so every input and every output is a plain event that can be logged and replayed, a market-data feed that publishes L2 book updates and trades, and a replay tool that proves the engine is deterministic. Then two kinds of evidence: randomized tests that compare the engine against a deliberately naive reference matcher on millions of orders, and latency percentiles (p50, p99, p99.9) for the add, cancel and match paths.

Everything in that list got built. Multiple symbols, auctions, self-trade prevention and a network gateway are the next milestones.

### Theory

A limit order says "buy up to Q at price P or better." Orders that cannot trade immediately rest in the book. The book has two sides, bids sorted from highest price down and asks sorted from lowest price up. The best bid and best ask form the touch, and the gap between them is the spread.

Price-time priority decides who trades first when an aggressive order arrives. The best price always goes first. Among orders at the same price, the one that arrived first goes first. The trade prints at the resting order's price, so the aggressor gets any price improvement. This rule is what makes queue position valuable, and it is why the modify rules matter: if you could increase your size without losing your place, you could sit at the front of the queue with a tiny order and grow it only when you liked the price. So a size increase or a price change sends the order to the back.

The order types differ only in what happens to the part that does not fill:

* a limit order rests
* a market order has no price bound and cancels any remainder
* IOC (immediate or cancel) has a price bound and cancels the remainder
* FOK (fill or kill) must fill completely or not at all, so the engine checks available quantity before trading anything
* post-only must not take liquidity; if it would cross, it is rejected, which lets market makers guarantee they pay the maker fee

Event sourcing means the engine's state is a pure function of its input log. If the engine has no hidden inputs (no clock, no randomness, no thread timing), you can rebuild any past state by replaying the log, and two replicas fed the same log stay identical. That is how real exchanges do failover and audit. It also makes testing easier, because a failing case is just a log file.

Fixed-point arithmetic matters because prices must compare exactly. With `double`, 0.1 + 0.2 is not 0.3, and a price level lookup keyed on a float can miss. So a price is an integer count of ticks and a quantity is an integer count of lots, and decimal strings are converted only at the edges.

### Architecture

The engine is a single-threaded state machine. Inputs come in as `InputEvent`s (new, cancel, modify), each is assigned the next sequence number, and the engine appends `OutputEvent`s to a caller-owned vector. For each input the outputs come in a fixed order: one of accepted, rejected or modified; then trades in match order; then a cancel for any unfilled IOC, market or FOK remainder; then L2 book updates, sorted.

```
 input log --> MatchingEngine --> output events --> output log
                 |  IdMap                    |----> MarketDataFeed (L2 mirror, trade tape)
                 |  vector<Level> per side   |----> StreamDigest (replay check)
                 |  order pool + free list
                 v
 tests: same inputs --> ReferenceMatcher --> must produce identical output events
```

Downstream consumers only ever see events. The `MarketDataFeed` rebuilds the public L2 book purely from `BookUpdate` events, and tests check that it always equals the engine's real book. If the engine ever forgot to publish a level change, the mirror would drift and a test would fail.

The `exch` CLI wraps it all: `gen` writes a random input log from a seed, `replay` runs a log and prints a digest plus the final book, `verify` checks a replay against a saved output log or digest, and `fuzz` runs the engine and the reference side by side. DESIGN.md has the full diagram, the invariants and the list of rejected alternatives.

### Implementation

#### The book

Each side is a `std::vector<Level>` sorted worst price first, so the best level is `back()`. Most activity happens at or near the touch, so matching reads `back()` and a level that empties is removed with `pop_back()`. Finding a level for an insert or a cancel checks the last 8 levels linearly and then falls back to binary search.

Each level owns a FIFO queue of orders implemented as an intrusive doubly linked list through a pool. The pool is a `std::vector<Order>` with a free list. Orders link to each other by 32-bit slot index rather than pointer, which keeps an order at 40 bytes and keeps links valid when the vector grows. Cancel from the middle of a queue is O(1) once you have the slot.

#### The id index

Cancel and modify arrive with an order id, so the engine needs id to slot lookup. I wrote `IdMap`, an open-addressing table with linear probing, load factor at most one half, a splitmix64 hash and backward-shift deletion (no tombstones). It is about 100 lines. Whether it was worth writing is one of the experiments below, and the answer turned out to be more interesting than I expected.

#### L2 publishing

Before the engine changes a level's quantity it records the level and its quantity in a small `touched_` list, once per level per input. After the input is done, it sorts that list (bids then asks, ascending price) and emits a `BookUpdate` with the new absolute quantity for every level that actually changed. A market order that eats ten orders at one price produces one L2 message, and a modify that nets to no change produces none.

#### The reference matcher

The oracle is written to be obviously correct rather than fast. All resting orders sit in one flat vector with an arrival stamp. Finding the next maker is a linear scan for the best price and then the earliest stamp. Its L2 updates come from building a full `std::map` snapshot of the book before and after each input and diffing them, so it shares none of the engine's incremental bookkeeping. If the engine's touched-level tracking missed a case, the two would disagree.

#### Randomized flow

`OrderFlow` generates a deterministic stream from a seed with its own splitmix64 generator, because `std::uniform_int_distribution` is not specified bit for bit across standard libraries and I wanted logs to be reproducible anywhere. It is open loop: it never looks at the book, so it naturally produces cancels of orders that already filled, modifies of dead orders, duplicate ids, zero quantities and negative prices. That exercises every reject path for free. The mid price drifts randomly so the book moves.

#### Tests

37 GoogleTest cases plus two CLI tests:

* scenario tests for each rule (price priority, time priority, sweeps, each order type, cancel, both modify cases, validation, L2 coalescing, sequence numbers)
* an `IdMap` test against `std::unordered_map` under 400,000 random operations on a small key space, to force wraparound and deletion from the middle of probe chains
* differential tests over five flow shapes (default, narrow and heavily crossing, wide with many levels, aggressive order mix, and 50 short seeds), comparing every output event and, every 257 events, the full priority-ordered order list, the invariants and the market-data mirror
* replay tests: the log codec round trips, the same inputs give the same digest, a decoded log replays identically, and dropping one input changes the digest
* a CLI round trip (`gen`, `replay --out`, `verify --expect --digest`) and a golden file test that replays `examples/basic.log` and compares against reviewed output

### Problems

#### Apple clang's AddressSanitizer hangs on this machine

The first sanitizer build compiled fine and then every binary hung forever, including the test discovery step. I reduced it to `int main(){return 0;}` built with `-fsanitize=address`: it still hung, with or without the command sandbox. UBSan alone was fine. `ASAN_OPTIONS=verbosity=2` showed the runtime stuck in `FindDynamicShadowStart`, the step where ASan looks for a free region of address space for its shadow memory. So this is Apple clang 17's ASan runtime on macOS 26.5, not my code. Homebrew LLVM 23 builds a working ASan binary, so the `asan` preset uses that compiler. It then produced DWARF 5 debug info that Apple's linker warned about on every object, which `-gdwarf-4` silences.

#### The machine was shared with heavy jobs

Other large builds and experiments were running the whole session. The 1-minute load average stayed between 172 and 352 on 14 cores during the benchmark (`results/bench/machine.txt`). In my first benchmark attempt the benchmark process got about 5% of a core, and the same mixed-flow run varied by roughly a factor of ten in wall-clock throughput from one run to the next. I changed three things. Throughput is measured in thread CPU time (`CLOCK_THREAD_CPUTIME_ID`) as well as wall time, since CPU time excludes time spent descheduled. Every configuration runs three times with the variants interleaved, and the reported number is the median. The load average is logged before and after every run. CPU-time throughput still varied between runs (9.2 to 17.4 M events/s for the default engine in `results/bench/throughput_idmap_r*.csv`), which I attribute to the scheduler moving the thread between performance and efficiency cores and to cache pollution from neighbours. Maximum latencies of tens to hundreds of milliseconds in the CSVs are preemptions, not engine behaviour, so I do not report them as results.

#### The timer is coarser than the thing being timed

`steady_clock` on Apple Silicon advances in steps of 41 to 42 ns (a 24 MHz counter): the smallest nonzero step I measured was 41 ns and back-to-back reads have a p50 of 41 ns (`results/bench/timer_idmap_r1.txt`). An add or a match takes a few of those ticks, so every latency percentile is a multiple of about 41.7 ns (42, 83, 125, 167, 208 and so on) and a p50 of 84 ns means "two ticks". User space cannot read the cycle counter on Apple Silicon without kernel help, so I kept per-order timing for the distribution shape and added burst throughput runs with no per-order timers to get averaged per-order costs below the timer resolution.

#### A misleading throughput column

My first benchmark printed an ops/s figure for each latency scenario by dividing the timed operations by the wall time of the whole loop. That loop also contains the untimed step that restores the book (a cancel after each add, an add after each cancel), plus benchmark bookkeeping in an `unordered_map`, so the figure was neither the add rate nor the cancel rate. I deleted it and added separate burst runs that time 1M adds into an empty book, 1M cancels in random order, and 1M aggressive orders that each fill exactly one maker.

#### A priority bug the event diff did not catch right away

To check that the fuzzer can actually find bugs, `scripts/mutation_test.sh` plants six known bugs in a copy of the engine and runs `exch fuzz` on each. All six were caught (`results/mutation.txt`). Four showed up as a direct output mismatch within the first 600 events and a fifth (no L2 update on an in-place amend) at event 3,487. The exception was the most realistic bug: letting a modify that increases size keep its queue position. That mutant produced identical output events at the moment of the modify, because queue position is invisible until a later trade picks the wrong maker. It was caught at event 3000 by the periodic comparison of the full priority-ordered order list, not by the stream diff. That is why both the tests and the fuzzer compare state as well as outputs.

#### std::unordered_map was faster at inserting

I expected my open-addressing `IdMap` to beat `std::unordered_map` everywhere. It did not. On the burst of 1M adds, the standard map ran at 9.8 M orders per CPU second against 4.4 M for mine (`results/bench/summary_burst.csv`). The reason is the hash. libc++ hashes integers with the identity function, so sequential order ids land in sequential buckets and inserts walk memory in order. My splitmix64 hash scatters them randomly over a table of 2^21 slots of 16 bytes each, which is 32 MB, so each insert is a likely cache miss.

#### An identity hash made matching 14 times slower

So I tried the obvious fix, an identity hash in `IdMap`, as a third benchmark variant. Adds became as fast as the standard map (9.8 M/s), but the match burst dropped to 0.27 M orders per CPU second, 14 times slower than splitmix64's 3.8 M/s (`results/bench/summary_burst.csv`, per-run numbers 0.23 to 0.32 M/s in `results/bench/stdout_identity_r*.txt`). Sequential ids with an identity hash fill one long contiguous run of occupied slots. Backward-shift deletion walks forward from the hole to the end of the run. Matching removes makers in FIFO order, which is roughly ascending id order, so each deletion walks almost the whole remaining run. That is quadratic. Cancels in random order break the run up quickly, which is why the cancel burst did not show it. The identity hash also collapses on ids that share their low bits: 20,000 ids that are multiples of 2^16 inserted at 0.06 M/s against 6.5 M/s for splitmix64. A client choosing ids could use that as a denial of service. I kept splitmix64.

### Experiments

All experiments ran in this session on an Apple M4 Pro (14 cores, 48 GB) with macOS 26.5 and Apple clang 17 at `-O2`, on the loaded machine described above. `bench_engine` sets the thread's QoS class to user-interactive to favour performance cores, since macOS does not allow pinning.

#### Differential fuzzing

`exch fuzz` in Release, comparing every output event and, every 1000 events, the full order list, the invariants and the market-data mirror. Eight seeds of 1M events each with default flow shapes (four processes, `results/fuzz_seed{1,3,5,7}.txt`), plus one run of 300,000 events with a 401-tick price band and up to 5,000 live ids (`results/fuzz_deep_book.txt`). Each 2M-event run took 158 to 164 s of wall time on the loaded machine, almost all of it in the reference matcher's per-event `std::map` snapshots.

#### Mutation check of the fuzzer

`scripts/mutation_test.sh 200000` builds six mutants (modify-up keeps priority, FOK off by one, LIFO instead of FIFO within a level, no L2 update on an in-place amend, trade at the taker's price, post-only allowed to lock the touch) and runs 2 seeds of 200,000 events against each. Output in `results/mutation.txt`.

#### Latency by path and book size

For each book size (1k, 10k, 100k, 1M resting orders spread over 100 price levels per side) and each path, 300,000 timed operations after a warmup of 30,000. The book is held in a steady state: each timed add is followed by an untimed cancel of a random order, each timed cancel by an untimed add, and each timed match (an aggressive order that fills exactly one maker at the touch) by an untimed replacement maker. The benchmark checks that the book size is unchanged at the end. Three repetitions, median of each percentile. Raw data in `results/bench/latency_<variant>_r<k>.csv`, CDFs in `cdf_*.csv`, summary in `summary_latency.csv`.

#### Throughput

Two measurements with no per-order timers. Burst: 1M non-crossing adds into an empty book, then 1M cancels of those orders in random order, then (on a fresh book filled with the same adds) 1M aggressive orders each filling one maker. Mixed flow: 3M pre-generated events from `OrderFlow` (about 50% new orders across all five types, the rest cancels, modifies and a small share of invalid inputs; 1,495,188 new orders out of 3M in the first run of `throughput_idmap_r1.csv`), three seeds per run. Raw data in `burst_*.csv` and `throughput_*.csv`, summaries in `summary_burst.csv` and `summary_throughput.csv`.

#### Id index ablation

The same engine and benchmark compiled three ways: `idmap` (open addressing with splitmix64, the default), `stdmap` (`std::unordered_map`) and `identity` (open addressing with an identity hash). The variants run interleaved within each repetition so that load changes hit all three. The burst suite also inserts 20,000 ids that are multiples of 2^16.

#### Replay determinism

`exch gen --n 1000000 --seed 2026`, then two independent `exch replay` runs and an `exch verify` against the first run's output log and digest (`results/replay/`). A 20,000-event replay also wrote its L2 feed and trade tape as CSV (`results/replay/l2_seed7_20k.csv`, `trades_seed7_20k.csv`).

### Results

#### Correctness

* Release test suite: 39 of 39 passed (`results/tests_release.txt`). The ASan and UBSan suite (Homebrew LLVM clang++, Debug) also passed 39 of 39, in 132 s at `ctest -j2` with a load average of about 11 (`results/tests_asan_ubsan.txt`). An earlier sanitizer run under a load average near 300 needed over 8 minutes for each of the two longest differential tests.
* Differential fuzzing: 8.3 M random events with zero divergences from the reference matcher: 8 M in `results/fuzz_seed*.txt` (about 16.4 M output events and 2.83 M trades) and 300,000 in `results/fuzz_deep_book.txt`.
* Mutation check: 6 of 6 planted bugs caught (`results/mutation.txt`).
* Replay: 1,000,000 inputs produced 2,072,205 outputs, 358,590 trades and digest `fa2ff178540c7510` on both runs, and `verify` matched all 2,072,205 lines (`results/replay/replay_run1.txt`, `replay_run2.txt`, `verify.txt`).

#### Latency (default engine, median of 3 runs, `results/bench/summary_latency.csv`)

| Path | Resting orders | p50 | p99 | p99.9 |
|---|---|---|---|---|
| add | 10,000 | 167 ns | 541 ns | 3,208 ns |
| cancel | 10,000 | 209 ns | 916 ns | 7,333 ns |
| match (one fill) | 10,000 | 84 ns | 542 ns | 1,583 ns |
| add | 1,000,000 | 84 ns | 625 ns | 1,500 ns |
| cancel | 1,000,000 | 459 ns | 1,417 ns | 23,834 ns |
| match (one fill) | 1,000,000 | 209 ns | 958 ns | 6,083 ns |

Matching at the touch is the cheapest path because the best level is `back()` and the maker is the head of its queue. Cancel is the most sensitive to book size: its p50 goes from 167 ns at 1k orders to 459 ns at 1M, because it does a hash lookup and then touches a random order slot, and at 1M orders both the 40 MB pool and the id table are far bigger than cache. The add p50 at 1M (84 ns) being lower than at 100k (208 ns) is within the run-to-run spread for that cell (83 to 166 ns across the three runs in `latency_idmap_r*.csv`) and I would not read anything into it. The p99.9 values are dominated by the shared machine; the same cell differs by up to 10 times between runs (cancel at 10k: 5,209 to 73,417 ns).

Plots: `results/bench/latency_percentiles.png`, `latency_vs_book_size.png`, `latency_cdf.png`.

#### Throughput (default engine)

* Mixed random flow: median 13.1 M events per CPU second, which is 6.5 M new orders per CPU second, and best run 17.4 M events/s. Wall-clock median 4.8 M events/s (`results/bench/summary_throughput.csv`).
* Bursts of 1M orders, median per CPU second: add 4.4 M, cancel 2.2 M, match 3.8 M (`results/bench/summary_burst.csv`).

#### Id index ablation (`results/bench/summary_burst.csv`, `summary_latency.csv`, plot `idmap_ablation.png`)

| Variant | add burst | cancel burst | match burst | strided ids | cancel p50 at 1M |
|---|---|---|---|---|---|
| idmap (splitmix64, default) | 4.4 M/s | 2.2 M/s | 3.8 M/s | 6.5 M/s | 459 ns |
| std::unordered_map | 9.8 M/s | 1.4 M/s | 2.2 M/s | 10.4 M/s | 750 ns |
| identity hash | 9.8 M/s | 1.5 M/s | 0.27 M/s | 0.06 M/s | 459 ns |

Burst figures are orders per CPU second. The custom table wins where the id index is on the critical path of a large book: cancels in the steady-state benchmark had a p50 of 416 to 500 ns in all three `idmap` runs against 750 to 792 ns in all three `stdmap` runs at 1M orders, and the match burst was 1.7 times faster. It loses on inserting sequential ids, where the standard library's identity hash gets perfect locality. On the mixed flow, where the book stays small, the three variants were within run-to-run noise of each other (medians of 13.1, 16.0 and 15.9 M events per CPU second for idmap, stdmap and identity in `summary_throughput.csv`, with individual runs ranging from 9.2 to 22.1 M).

### What I would change

#### Exchange-assigned order ids

Most of the id index trouble comes from accepting arbitrary client ids. If a gateway maps each client id to an exchange id that increases by one per order, the index can be a flat array indexed by `id - base` with no hashing at all: sequential inserts, O(1) lookups, and no probe chains to degrade. That is the design I would build with the TCP gateway.

#### A faster reference matcher

The oracle spends most of the fuzz time rebuilding two `std::map` snapshots per event, which limits the fuzzer to about 12,000 events per second on this machine and makes the differential tests dominate the sanitizer run of the suite. Keeping the reference's L2 aggregate as a map updated alongside the flat vector would still be independent of the engine's touched-level logic and would be an order of magnitude faster.

#### Quieter benchmarking

The latency tails in this devlog describe the machine as much as the engine. I would rerun the suite on an idle machine, report the per-run spread next to every median, and try the kperf counters to get cycle-level timing instead of 41.7 ns ticks.

#### A price ladder for liquid instruments

The sorted vector of levels is simple and fast at the touch, but inserting a new level deep in the book moves every level behind it. For an instrument with a known tick band, a dense array indexed by price with a bitmap for finding the next non-empty level would make every level operation O(1).

#### A binary log

The text log is pleasant to read and diff, but replaying the 1M-event log took 3.4 s of wall time without writing outputs and 6.9 s with the output log (`results/replay/replay_run2.txt`, `replay_run1.txt`), while the engine itself processes a million events in well under a tenth of a CPU second. A fixed-size binary record per event would make replay limited by the engine instead of by `snprintf`.

#### Post-only semantics on modify

A post-only order that is modified across the spread currently trades as a plain limit order. A real venue would reject the modify or reprice it, and the event model should carry the original order type so the engine can apply the right rule.

### Verification

On 2026-09-27 an independent clean Release rebuild passed all 39 tests. A fresh differential fuzz run with a new seed (seed 11, 200,000 events) matched the reference matcher with zero divergence. The ASan and UBSan suite passed in the builder's rerun (`results/tests_asan_ubsan.txt`). The benchmarks were not rerun independently, and they were measured at 1-minute load averages of 172 to 352 on 14 cores, so every latency and throughput number above should be read as an indicative upper bound rather than a clean measurement. The 41.7 ns timer tick also quantizes every latency, so each percentile is a multiple of one tick.
