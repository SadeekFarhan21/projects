---
layout: post
title: fuzzing a price-time matching engine
tags:
  - trading
  - market-microstructure
  - testing
  - cpp
description: >-
  A price-time priority matching engine in C++, checked event by event against a
  deliberately simple reference on 8.3 million random events.
date: 2026-09-29 02:17:09
---


I wrote the core of a small exchange in C++20, a limit order book and matching engine for one symbol on one thread, with price-time priority, five order types, cancel, and modify with the usual queue position rules. Every input and output is a plain event, so the engine is a deterministic state machine that can be replayed from its log.

The part I trust most is not the engine but the test that watches it. A second matcher, written to be obviously correct rather than fast, processes the same random order flow, and the two are compared on every output event and periodically on the full priority-ordered contents of the book. Across **8.3 million random events** the two never disagreed. On 2026-09-27 an independent rerun with a seed I had not used (seed 11, 200,000 events) also matched with **zero divergence**, and a clean Release rebuild passed all **39 tests**.

A fuzzer that passes proves little unless it can fail. So I planted six known bugs, one at a time, in copies of the engine and ran the fuzzer against each. It caught **6 of 6**, the slowest within 3,487 events. The most instructive one, a modify that grows an order without sending it to the back of the queue, produced exactly the right output at the moment it happened and was only visible when the book's internal order was compared. That result is the reason the test compares state as well as outputs.

The latency numbers are less solid. They were measured on a shared Apple M4 Pro at 1 minute load averages of 172 to 352 on 14 cores, not rerun independently, and quantized by a 41.7 ns timer tick. With 10,000 resting orders the median add took 167 ns, cancel 209 ns and a single-fill match 84 ns, which I treat as indicative upper bounds.

Code is in `projects/06-price-time-matching-engine`. Skip to [problems](#problems) for the bugs.

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

A matching engine holds resting buy and sell orders, decides which ones trade when a new order arrives, and tells the world what happened. I wanted one built the way a real venue would build it, with a v0 scope fixed up front.

- One symbol, one thread, a limit order book with price-time priority.
- Limit, market, IOC, FOK and post-only orders, plus cancel and modify, where a modify that increases size or changes price loses its place in the queue.
- Integer prices and quantities, with no floating point anywhere in the engine.
- An event-sourced interface, so every input and output can be logged and replayed.
- An L2 market-data feed and a trade tape, and a replay tool that proves the engine is deterministic.
- Randomized differential tests against a naive reference matcher on millions of orders.
- Latency percentiles (p50, p99, p99.9) for the add, cancel and match paths, and throughput.

Everything on that list got built. Multiple symbols, opening and closing auctions, self-trade prevention and a network gateway are the next milestones. Later projects in this series are meant to consume this engine's event and market-data streams.

## theory

### price-time priority

A limit order says "buy up to Q at price P or better". An order that cannot trade immediately rests in the book. The book has two sides, bids sorted from the highest price down and asks from the lowest price up. The best bid and best ask are the touch, and the gap between them is the spread.

When an aggressive order arrives, price-time priority decides who trades with it. The best price always goes first. Among orders at the same price, the one that arrived first goes first. The trade prints at the resting order's price, so the aggressor keeps any price improvement.

### what each order type does with its leftovers

The five order types differ only in what happens to the part that does not fill.

| Type | Price bound | Unfilled remainder |
|---|---|---|
| limit | yes | rests until cancelled |
| market | none | cancelled |
| IOC | yes | cancelled |
| FOK | yes | the whole order is cancelled unless it can fill completely |
| post-only | yes | rests, and the order is rejected if it would trade on arrival |

FOK must check available quantity within its limit before it prints a single trade, because trades are irrevocable. Post-only lets a market maker be sure it pays the maker fee, and I reject it even when it would only lock the touch, since at that price it would trade.

### why modify has to cost queue position

At a given price the front of the queue fills first. If an order could grow without losing its place, a trader could sit at the front with one lot and increase it only when they liked the market. So reducing size at the same price keeps priority, and increasing size or changing price sends the order to the back, as if it had been cancelled and re-entered. Breaking this rule produces a bug that is invisible in the output, as problem 1 shows.

### event sourcing and determinism

If the engine has no hidden inputs, no clock, no randomness and no dependence on thread timing, then its state is a pure function of its input log. Replaying the log into a fresh engine reproduces every output, including trade ids, bit for bit. That is how exchanges do failover and audit, and any failure becomes a log file that reproduces it. Time priority does not need a wall clock either, because arrival order is exactly the sequence number the engine assigns.

### fixed point

A price level is looked up by price, so prices must compare exactly, which rules out `double`. A price is an `int64` count of ticks, a quantity is an `int64` count of lots, and decimal strings are converted only at the edges. The parser refuses input that would need rounding, so `101.255` at a scale of two decimals is an error, not 101.26. Quantities are capped at 2^40, which keeps a level's aggregate quantity far from `int64` overflow.

### differential testing

Unit tests check the cases I thought of, and the bugs that hurt live in combinations I did not, such as a modify of a partly filled order at a level about to empty. Differential testing writes the specification twice, once fast and once so simply it is hard to get wrong, feeds both the same random inputs and compares everything they produce. The two share no data structures, so a bug in the engine's incremental bookkeeping has nothing in the oracle to hide behind. The weak point is that the comparison only sees what it compares, and a bug that changes internal state without changing any output yet can hide for a long time.

## architecture

The engine is a single-threaded state machine. Inputs arrive as `InputEvent`s (new, cancel, modify), each gets the next sequence number, and the engine appends `OutputEvent`s to a vector the caller owns. There are no callbacks or virtual calls in the hot path.

```
                     input log (text, one event per line)
                                   |
                                   v
                   +--------------------------------------+
                   |            MatchingEngine            |
                   |   validate -> match -> rest/cancel   |
                   |                                      |
                   |   IdMap           two book sides     |
                   |   id -> slot      vector<Level>      |
                   |                   worst ... best     |
                   |   order pool      each Level is a    |
                   |   + free list     FIFO of pool slots |
                   +--------------------------------------+
                                   |
                   OutputEvents, each tagged with input seq
                                   |
           +-----------------------+------------------------+
           v                       v                        v
      output log            MarketDataFeed             StreamDigest
      (exch replay)         L2 mirror + tape           FNV-1a over outputs

   test side: same InputEvents -> ReferenceMatcher -> identical outputs required
```

For one input the outputs always come in the same order. First exactly one of `ACCEPTED`, `REJECTED` or `MODIFIED`. Then trades, in match order. Then a `CANCELED` for any unfilled market, IOC or FOK remainder. Then `L2` book updates, one per price level whose aggregate changed, bids first and then asks, each by ascending price. Here is part of the hand-written golden session in `examples/`, with each input shown as a comment above the outputs it caused.

```
# N 8 B LMT 10020 12   aggressive buy, takes 10010 and part of 10020
7 ACCEPTED 8 B 10020 12
7 TRADE 1 taker=8 maker=4 B 10010 10
7 TRADE 2 taker=8 maker=5 B 10020 2
7 L2 S 10010 0
7 L2 S 10020 13
# N 10 S FOK 9990 100  cannot fill completely, so no trades at all
9 ACCEPTED 10 S 9990 100
9 CANCELED 10 S 9990 100 fok_unfilled
```

L2 updates carry the new absolute quantity at a level rather than a delta, so a late joiner only needs a snapshot and a dropped message cannot corrupt every later view. The `MarketDataFeed` rebuilds the public book purely from these events, and tests check that it always equals the engine's real book.

The `exch` tool wraps this in four commands, `gen`, `replay`, `verify` and `fuzz`.

## implementation

### the book

Each side is a `std::vector<Level>` sorted from the worst price to the best, so the best level is always `back()`. Most activity happens near the touch, so matching reads `back()` and removes an emptied level with `pop_back()`, both O(1). Inserting a level deep in the book costs a `memmove`, which I accepted. Finding a level scans the last 8 entries and falls back to binary search.

```
 bids  vector<Level>, worst -> best              asks  vector<Level>, worst -> best
 [9980][9985][9990]<- back() = best bid          [10030][10020][10010]<- back() = best ask
                 |
                 v  Level{price, total, count, head, tail}
            head -> slot 17 <-> slot 4 <-> slot 52 <- tail     (FIFO through the pool)
```

Each level owns a FIFO queue, an intrusive doubly linked list through a pool, which is a `std::vector<Order>` plus a free list. Orders link by 32-bit slot index rather than pointer, which keeps an order at 40 bytes and keeps links valid when the vector grows. Once the engine knows an order's slot, cancelling it from the middle of a queue is O(1).

### the matching loop

This is the whole matching path, trimmed, with the construction of the trade event folded into a helper. It walks the opposite side from the best level inward and each level's queue from the head.

```cpp
while (qty > 0 && !book.empty()) {
    Level& L = book.back();                       // best opposite level
    if (has_limit && !crosses(side, limit, L.price)) break;
    touch(opp, L.price, L.total);                 // remember it for L2
    while (qty > 0 && L.head != kNil) {
        Order& m = pool_[L.head];                 // oldest order at this price
        const Qty fill = std::min(qty, m.qty);
        out.push_back(trade(taker, m.id, next_trade_++, side, L.price, fill));
        m.qty -= fill; L.total -= fill; qty -= fill;
        if (m.qty == 0) {                         // maker done: pop the queue head
            const std::uint32_t mi = L.head;
            L.head = m.next;
            if (L.head != kNil) pool_[L.head].prev = kNil; else L.tail = kNil;
            --L.count;
            ids_.erase(m.id);
            free_.push_back(mi);
        }
    }
    if (L.head == kNil) book.pop_back();
}
return qty;                                       // the caller decides: rest or cancel
```

The trade prints at `L.price`, the maker's price. One of the planted bugs later changed exactly that line to print at the taker's limit, and the fuzzer caught it on the ninth event.

FOK runs a read-only pass first. `fillable` sums `total` over crossing levels from the best inward, and matching starts only if the sum reaches the order size, which is why the failed FOK above produced no trades.

### modify

Modify is where the priority rule lives, in one condition.

```cpp
if (in.price == o.price && in.qty <= o.qty) {
    // Same price, size not increased: amend in place, keep queue position.
    touch(side, L.price, L.total);
    L.total -= o.qty - in.qty;
    o.qty = in.qty;
    out.push_back(m);
    return;
}
// Price change or size increase: loses priority. Leave the book and
// re-enter as a fresh limit order under the same id, which can trade.
remove(oi);
out.push_back(m);
const Qty left = match(id, side, true, in.price, in.qty, out);
if (left > 0) rest(id, side, in.price, left);
```

A modify across the spread therefore trades immediately, with the modified order as taker. A modified order always becomes a plain limit order, so a post-only order modified across the spread trades instead of being rejected, a simplification rather than a feature.

### L2 coalescing

Before the engine changes any level it calls `touch(side, price, quantity_before)`, which records the level once per input and ignores later calls for the same level. After the input is handled, it sorts the touched list into canonical order and emits an update for each level whose quantity now differs from the recorded value.

```cpp
for (const Touched& t : touched_) {
    const Qty now = level_qty(t.side, t.price);
    if (now == t.before) continue;   // net no-op, publish nothing
    out.push_back(book_update(seq_, t.side, t.price, now));
}
touched_.clear();
```

A market order that eats ten orders at one price produces one L2 message, and an input that leaves a level's total unchanged produces none. The cost is that every code path that changes a level must remember to call `touch` first. Forgetting it in one branch is another planted bug, and it was the hardest of the six for the fuzzer to hit.

### the id index

Cancel and modify arrive with an order id, so the engine needs a map from id to pool slot. I wrote `IdMap`, an open addressing table with linear probing, a power-of-two capacity, a load factor of at most one half, a splitmix64 hash so that sequential ids spread out, and backward shift deletion so that there are no tombstones. The deletion is the interesting part.

```cpp
// After finding the key at slot i, pull later entries of the probe run
// back into the hole whenever their home slot allows it.
std::size_t hole = i, j = i;
while (true) {
    j = (j + 1) & mask_;
    if (slots_[j].key == 0) break;               // end of the probe run
    std::size_t h = home(slots_[j].key);
    bool movable = (hole <= j) ? (h <= hole || h > j) : (h <= hole && h > j);
    if (movable) { slots_[hole] = slots_[j]; hole = j; }
}
slots_[hole] = Slot{};
```

Deletion costs time proportional to the rest of the probe run. With a good hash, runs are short. Problem 3 found a way to make one run as long as the table. The table has its own randomized test against `std::unordered_map`, 400,000 operations on a small key space to force wraparound and deletions from the middle of probe chains.

### the reference matcher

The oracle keeps every resting order in one flat vector with an arrival stamp. Finding the next maker is a linear scan.

```cpp
int ReferenceMatcher::best_opposite(Side taker_side) const {
    const Side want = opposite(taker_side);
    int best = -1;
    for (std::size_t i = 0; i < orders_.size(); ++i) {
        const Resting& o = orders_[i];
        if (o.side != want) continue;
        if (best < 0) { best = int(i); continue; }
        const Resting& b = orders_[best];
        const bool better_price = want == Side::Buy ? o.price > b.price : o.price < b.price;
        if (better_price || (o.price == b.price && o.time < b.time)) best = int(i);
    }
    return best;
}
```

That function is the definition of price-time priority with no queues or levels to get wrong. The oracle's L2 updates come from diffing full `std::map` snapshots of the book taken before and after each input, sharing none of the engine's touched-level logic. The price is speed, since those two snapshots per event dominate every fuzz run.

### the order flow generator

`OrderFlow` produces a deterministic stream from a seed with its own splitmix64 generator, because `std::uniform_int_distribution` is not specified bit for bit across standard libraries. By default about 30 percent of events are cancels and 10 percent modifies. New orders are 3 percent market, 7 percent IOC, 5 percent FOK and 10 percent post-only, the rest plain limits, priced in a band around a mid that drifts one tick at a time, and half a percent of events are deliberately invalid. The generator never looks at the book, so it naturally cancels filled orders, modifies dead ones and reuses live ids, which exercises every reject path for free.

### the fuzz loop

The comparison itself is short. Every output event must match, and every `check_every` events the full state must match too.

```cpp
eng.process(e, a_out);
ref.process(e, b_out);
if (a_out != b_out) { print_divergence(seed, i, e, a_out, b_out); return 1; }
for (auto& o : a_out) md.on_event(o);              // feed the L2 mirror
if (i % check_every == 0) {
    eng.check_invariants();                         // aborts on failure
    if (eng.orders_in_priority(Side::Buy)  != ref.orders_in_priority(Side::Buy)  ||
        eng.orders_in_priority(Side::Sell) != ref.orders_in_priority(Side::Sell) ||
        md.depth(Side::Buy)  != eng.depth(Side::Buy) ||
        md.depth(Side::Sell) != eng.depth(Side::Sell)) {
        std::printf("STATE DIVERGENCE seed %llu event %llu\n", seed, i);
        return 1;
    }
}
```

`orders_in_priority` lists every resting order on a side in the exact order it would fill. `check_invariants` aborts if levels are unsorted or empty, if a level's total or count disagrees with its queue, if back links do not mirror forward links, if the id map holds anything but the resting orders, if a pool slot is neither resting nor free, or if the book is crossed.

### tests

The suite is 37 GoogleTest cases plus two command line tests. Scenario tests cover each rule. Differential tests run five flow shapes (default, narrow and heavily crossing, a wide book, an aggressive mix, and 50 short seeds) and compare state every 257 events. Replay tests check that the same inputs give the same digest and that dropping one input changes it, and the golden session replays against its reviewed output.

## problems

### 1. a priority bug with no wrong output

To check that the fuzzer could find bugs at all, `scripts/mutation_test.sh` copies the engine, applies one `sed` edit that plants a known bug, rebuilds, and runs `exch fuzz` for two seeds of 200,000 events with a state check every 100 events. The most realistic mutant is a one-token change to the modify condition.

```diff
- if (in.price == o.price && in.qty <= o.qty) {
+ if (in.price == o.price) {
```

With this change, an order that grows at the same price keeps its queue position. I expected the fuzzer to catch it immediately through the output diff. It did not. At the moment of the modify, the correct engine and the mutant emit identical events. Both emit `MODIFIED` and then one `L2` update with the level's new total, because the correct path removes the order and re-rests it at the same price with no trade in between. The only difference is where the order sits in its queue, and queue position is invisible until a later aggressive order reaches that level and fills the wrong maker first.

The mutant was caught by the periodic comparison of `orders_in_priority`, which reported a state divergence at event 3000 of seed 1. The other five mutants were caught by the output diff, most of them quickly. Printing at the taker's price failed on event 9, LIFO queues on event 86, a post-only that may lock the touch on event 112, and an off-by-one in the FOK check (`<=` instead of `<`) on event 593.

<figure data-figure="chart:projects/exchange-from-scratch/exchange-from-scratch-mutation"></figure>

The slowest catch was the missing `touch` in the in-place amend branch, at event 3,487. My reading of why it took so long is that the bug needs a modify whose new price lands exactly on the order's current price with a size that does not increase, and the generator draws modify prices uniformly from a band about 41 ticks wide, so such modifies are rare. I have not measured that rate directly. Once one happens, the L2 stream is visibly short an update and the diff fails on that event.

The lesson I took is that an output diff tests the interface and a state diff tests the model, and a bug can live in the model for a long time before it reaches the interface. Both the tests and the fuzzer compare state for this reason.

### 2. the standard library map was faster at inserting

I wrote `IdMap` expecting it to beat `std::unordered_map` everywhere. It did not. On a burst of 1M adds into an empty book, the standard map ran at 9.8M orders per CPU second against 4.4M for mine. The reason is the hash. libc++ hashes integers with the identity function, so sequential order ids go into sequential buckets and inserts walk memory in order. My splitmix64 hash scatters them over a table of 2^21 slots of 16 bytes each, which is 32 MB, so each insert is likely a cache miss.

### 3. an identity hash made matching 14 times slower

The obvious fix was to use the identity hash in `IdMap` too, so I added it as a third benchmark variant. Adds became as fast as the standard map, 9.8M per CPU second. Matching fell to 0.27M per CPU second, 14 times slower than the 3.8M of splitmix64, with per-run values between 0.23M and 0.32M.

The cause is the backward shift loop shown above. Sequential ids with an identity hash fill one long contiguous run of occupied slots. Deletion walks forward from the hole to the end of the run. Matching removes makers in FIFO order, which is roughly ascending id order, so every deletion near the start of the run walks almost the whole remaining run. The total cost is quadratic. Cancels in random order break the run into pieces quickly, which is why the cancel burst did not show the problem and why a benchmark with only random cancels would have hidden it.

The identity hash also collapses on ids that share their low bits. Inserting 20,000 ids that are multiples of 2^16 ran at 0.06M per CPU second against 6.5M for splitmix64, because they all probe from the same home slot. A client that picks its own ids could use that to slow the engine down on purpose. I kept splitmix64, and the real fix is in what I would change.

### 4. the timer is coarser than the thing being timed

`steady_clock` on Apple Silicon advances in steps of 41 to 42 ns. It is a 24 MHz counter. The smallest nonzero step I measured was 41 ns and back-to-back reads had a p50 of 41 ns. An add or a match takes a few of those ticks, so every latency percentile is a multiple of about 41.7 ns (42, 83, 125, 167, 208 and so on), and a match p50 of 84 ns means "two ticks", not 84 ns to three significant figures. The CDF of any path is a staircase.

User space cannot read the cycle counter on Apple Silicon without help from the kernel. I kept per-order timing for the shape of the distribution and added burst runs with no timers inside the loop, which give an averaged cost per order below the timer's resolution.

### 5. the machine was not mine

Other heavy jobs shared the machine the whole session, and during the benchmark the 1 minute load average stayed between 172 and 352 on 14 cores. In my first attempt the benchmark process got about 5 percent of a core and the same mixed-flow run varied by roughly a factor of ten in wall-clock throughput.

I changed three things. Throughput is measured in thread CPU time (`CLOCK_THREAD_CPUTIME_ID`) as well as wall time, since CPU time excludes time spent descheduled. Every configuration runs three times with the variants interleaved, so a burst of load hits all of them, and the reported number is the median. The benchmark also requests the user-interactive QoS class, since macOS does not allow pinning. None of this makes the numbers clean. CPU-time throughput for the default engine on the mixed flow still ranged from 9.2M to 17.4M events per second across runs, which I attribute to the scheduler moving the thread between performance and efficiency cores and to cache pollution from neighbours. Maximum latencies in the raw files reach tens to hundreds of milliseconds. Those are preemptions, and I do not report them.

### 6. a throughput column that measured nothing

My first benchmark printed an operations per second figure for each latency scenario, dividing timed operations by the wall time of a loop that also contained the untimed steps keeping the book at a steady size. The number was neither the add rate nor the cancel rate. I deleted it and wrote separate burst runs, 1M adds into an empty book, 1M cancels in random order, and 1M aggressive orders that each fill exactly one maker.

### 7. Apple clang's AddressSanitizer hangs

The first sanitizer build compiled and then every binary hung, including the test discovery step. I reduced it to `int main() { return 0; }` built with `-fsanitize=address`, which still hung. UBSan alone was fine. `ASAN_OPTIONS=verbosity=2` showed the runtime stuck in `FindDynamicShadowStart`, where ASan searches for free address space for its shadow memory. So the problem was Apple clang 17's ASan runtime on macOS 26.5, not my code. Homebrew LLVM builds working ASan binaries, so the `asan` preset uses that compiler, plus `-gdwarf-4` to silence the linker warning its DWARF 5 debug info produced on every object.

## experiments

### differential fuzzing

`exch fuzz` in Release, with a state check every 1,000 events. Eight seeds of 1M events, run as four processes of two seeds, with price bands from 11 to 101 ticks wide depending on the seed, plus one run of 300,000 events with a 401-tick band and up to 5,000 live ids for a deeper book. Each 2M-event process took 157.5 to 163.9 s of wall time on the loaded machine and the deep book run 325.4 s, almost all of it in the reference matcher.

### mutation check

The six mutants from problem 1, each run for two seeds of 200,000 events with a state check every 100 events.

### latency by path and book size

For each book size (1k, 10k, 100k and 1M resting orders over 100 price levels per side) and each path, 300,000 timed operations after 30,000 of warmup. Each timed add is followed by an untimed cancel, each timed cancel by an untimed add, and each timed single-fill match by an untimed replacement maker, so the book stays the same size. Three repetitions, median of each percentile.

### throughput

Burst runs as described in problem 6, and a mixed flow of 3M pre-generated events from `OrderFlow`, about half of them new orders across all five types (1,495,188 of 3M in the first run) and the rest cancels, modifies and invalid inputs, with three seeds per run and three runs per variant.

### id index ablation

The same engine and benchmark compiled with the default `IdMap`, with `std::unordered_map`, and with `IdMap` using an identity hash, interleaved within each repetition.

### replay determinism

`exch gen --n 1000000 --seed 2026`, two independent `exch replay` runs, and an `exch verify` of a replay against the first run's output log and digest.

## results

All benchmark numbers come from an Apple M4 Pro (14 cores, 48 GB) running macOS 26.5, Apple clang 17 at `-O2`, on the loaded machine described in problem 5. They were not rerun independently. The correctness results were.

### correctness

| Check | Result | Source |
|---|---|---|
| Release test suite | 39 of 39 passed, and 39 of 39 again in an independent clean rebuild | `results/tests_release.txt` |
| ASan and UBSan suite | 39 of 39 passed, 132 s at `ctest -j2`, in the builder's rerun | `results/tests_asan_ubsan.txt` |
| differential fuzz, 8 seeds of 1M | 8M events, 16.4M output events, 2.83M trades, 0 divergences | `results/fuzz_seed*.txt` |
| differential fuzz, deep book | 300,000 events, 101,890 trades, 0 divergences | `results/fuzz_deep_book.txt` |
| independent fuzz rerun, seed 11 | 200,000 events, 0 divergences | verification rerun |
| planted bugs caught | 6 of 6 | `results/mutation.txt` |
| replay of 1M inputs | same digest `fa2ff178540c7510` on both runs, verify matched all 2,072,205 output lines | `results/replay/` |

The 1M-event replay produced 2,072,205 outputs, 358,590 trades and 675,925 L2 updates, and left 95 orders resting on 17 bid and 14 ask levels, identically on both runs. A sanitizer run earlier in the session, at a load average near 300, needed over 8 minutes for each of the two longest differential tests. The 132 s figure is from a rerun at a load average of about 11.

The fuzzer shows the engine agrees with the reference on the flows the generator produces, and the mutation check shows the comparison is sensitive to six specific kinds of error. It does not show the reference itself is right. The reference could share a misreading of the rules with the engine, since I wrote both. The golden session in `examples/`, whose output I reviewed by hand, is the check against that, and it is small.

### latency

| Path | Resting orders | p50 | p99 | p99.9 |
|---|---|---|---|---|
| add | 10,000 | 167 ns | 541 ns | 3,208 ns |
| cancel | 10,000 | 209 ns | 916 ns | 7,333 ns |
| match, one fill | 10,000 | 84 ns | 542 ns | 1,583 ns |
| add | 1,000,000 | 84 ns | 625 ns | 1,500 ns |
| cancel | 1,000,000 | 459 ns | 1,417 ns | 23,834 ns |
| match, one fill | 1,000,000 | 209 ns | 958 ns | 6,083 ns |

<figure data-figure="chart:projects/exchange-from-scratch/exchange-from-scratch-latency"></figure>

Matching at the touch is the cheapest path, because the best level is `back()` and the maker is the head of its queue. Cancel is the most sensitive to book size. Its p50 goes from 167 ns at 1k orders to 459 ns at 1M, because it does a hash lookup and then touches a random order slot, and at 1M orders both the 40 MB pool and the id table are far larger than cache.

Two cells should not be over-read. The add p50 at 1M (84 ns) is lower than at 100k (208 ns), but that cell ranged from 83 to 166 ns across the three runs, so I read it as noise plus tick quantization rather than an effect. The p99.9 column mostly describes the machine. The same cell differs by up to ten times between runs, with cancel at 10k ranging from 5,209 to 73,417 ns.

### throughput

On the mixed random flow the default engine's median was 13.1M events per CPU second, which is 6.5M new orders per CPU second, with a best run of 17.4M. The wall-clock median was 4.8M events per second, which says more about the load than about the engine. On the bursts of 1M orders the medians were 4.4M adds, 2.2M cancels and 3.8M single-fill matches per CPU second.

### the id index ablation

<figure data-figure="chart:projects/exchange-from-scratch/exchange-from-scratch-id-index-burst"></figure>

| Variant | add burst | cancel burst | match burst | strided ids | cancel p50 at 1M |
|---|---|---|---|---|---|
| IdMap, splitmix64 (default) | 4.4M/s | 2.2M/s | 3.8M/s | 6.5M/s | 459 ns |
| `std::unordered_map` | 9.8M/s | 1.4M/s | 2.2M/s | 10.4M/s | 750 ns |
| IdMap, identity hash | 9.8M/s | 1.5M/s | 0.27M/s | 0.06M/s | 459 ns |

Burst figures are orders per CPU second. The custom table wins where the id index is on the critical path of a large book. At 1M resting orders, all three `idmap` runs had a cancel p50 of 416 to 500 ns and all three `stdmap` runs 750 to 792 ns, so the ranges do not overlap even on this noisy machine, and the match burst was 1.7 times faster. It loses on inserting sequential ids, where the standard library's identity hash gets perfect locality.

<figure data-figure="chart:projects/exchange-from-scratch/exchange-from-scratch-cancel-book-size"></figure>

On the mixed flow, where the book stays small (about 1,500 orders resting at the end of each run), the three variants were within run-to-run noise of each other, with medians of 13.1M, 16.0M and 15.9M events per CPU second for idmap, stdmap and identity and individual runs from 9.2M to 22.1M. I would not claim the default is slower on that flow, and I would not claim it is faster either.

## what I would change

### exchange-assigned order ids

Most of the id index trouble comes from accepting arbitrary client ids. If a gateway mapped each client id to an exchange id that increases by one per order, the index could be a flat array indexed by `id - base`, with no hashing at all. Inserts would be sequential, lookups O(1), and there would be no probe runs to degrade and no ids for a client to choose adversarially. That is the design I would build with the TCP gateway.

### a faster reference matcher

The oracle's per-event snapshots limit the fuzzer to about 12,000 events per second on this machine. Keeping the reference's L2 aggregate in a map updated alongside its flat vector would stay independent of the engine's touched-level logic and should be much faster, and more events per second means more seeds for rare bugs like the missing amend update.

### measure how often the generator hits each rule

The amend bug took 3,487 events because same-price modifies are rare in the generated flow. I would count how often each branch of the engine fires during a fuzz run, and bias the generator toward the rare ones, rather than guess at their frequency as I did above.

### quieter benchmarking

I would rerun the suite on an idle machine, report the per-run spread next to every median, and try the kperf counters for cycle-level timing instead of 41.7 ns ticks.

### a price ladder for liquid instruments

The sorted vector of levels is simple and fast at the touch, but inserting a new level deep in the book moves every level behind it. For an instrument with a known tick band, a dense array indexed by price with a bitmap for finding the next non-empty level would make every level operation O(1).

### a binary log

Replaying the 1M-event text log took 3.4 s of wall time without writing outputs and 6.9 s with the output log, while the engine itself processes 3M events of the mixed flow in 0.17 to 0.33 CPU seconds. A fixed-size binary record per event would make replay limited by the engine rather than by `snprintf`.

### post-only semantics on modify

A post-only order modified across the spread currently trades as a plain limit order. A real venue would reject the modify or reprice it, so the event model should carry the original order type and let the engine apply the right rule.

## reproducibility

Build. The `asan` preset needs Homebrew LLVM (`brew install llvm`) because of problem 7. GoogleTest is fetched by CMake at configure time.

```sh
cd projects/06-price-time-matching-engine
cmake --preset release && cmake --build --preset release
cmake --preset asan && cmake --build --preset asan
```

Tests.

```sh
ctest --preset release          # about 1 minute
ctest --preset asan -j2         # about 2 minutes on an idle machine
```

Differential fuzzing and the mutation check.

```sh
./build/release/exch fuzz --n 1000000 --seed 1 --seeds 2
./build/release/exch fuzz --n 300000 --seed 21 --width 200 --max-live 5000
./build/release/exch fuzz --n 200000 --seed 11
scripts/mutation_test.sh 200000
```

Replay determinism.

```sh
./build/release/exch gen --n 1000000 --seed 2026 --out /tmp/in.log
./build/release/exch replay --in /tmp/in.log --out /tmp/out.log
./build/release/exch verify --in /tmp/in.log --expect /tmp/out.log
```

Benchmarks and plots. Raw CSVs, per-run output, the load average log and the plots land in `results/bench/`.

```sh
scripts/run_bench.sh 3 300000 3000000
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/plot_bench.py results/bench
```

Code is in `projects/06-price-time-matching-engine`, with the engine, reference matcher and market-data feed in `src/` and `include/exchange/`, the `exch` tool in `tools/`, the benchmark in `bench/`, the tests in `tests/`, the data structures, invariants and rejected alternatives in `DESIGN.md`, and the build log with its verification note in `DEVLOG.md`.
