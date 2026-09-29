# Design

This document describes the v0 engine: one symbol, one thread, a limit order book with price-time priority, an event-sourced interface, an L2 market-data feed and a replay tool. Everything here refers to code under `include/exchange/` and `src/`.

## Architecture

```
                       input log (text, one event per line)
                                   |
                                   v
   +-----------+   InputEvent   +------------------------------------+
   | gateway / |--------------->|           MatchingEngine           |
   | exch gen  |   (seq n)      |                                    |
   | replay    |                |  validate -> match -> rest/cancel  |
   +-----------+                |      |          |                  |
                                |      v          v                  |
                                |  +--------+  +-----------------+   |
                                |  | IdMap  |  | book side x 2   |   |
                                |  | id ->  |  | vector<Level>   |   |
                                |  | slot   |  | worst..best     |   |
                                |  +--------+  |  Level: FIFO of |   |
                                |      |       |  pool slots     |   |
                                |      v       +-----------------+   |
                                |  +-------------------+             |
                                |  | order pool        |  touched    |
                                |  | vector<Order> +   |  levels --> |--+
                                |  | free list         |  (L2 diff)  |  |
                                |  +-------------------+             |  |
                                +------------------------------------+  |
                                                  |                     |
                         OutputEvent stream, all tagged with seq n      |
          (Accepted, Rejected, Trade, Canceled, Modified, BookUpdate) <-+
                                                  |
                 +--------------------+-----------+------------+
                 v                    v                        v
         output log (text)    MarketDataFeed            StreamDigest
         `exch replay --out`  L2 mirror + trade tape    FNV-1a, replay check
                              `--l2`, `--trades` CSV     `exch verify --digest`

   Test side:  same InputEvents --> ReferenceMatcher (flat vector, brute force)
               outputs compared event by event, state compared periodically
```

The engine is a deterministic state machine. It has no clock, no threads, no I/O and no randomness. Its only input is the ordered stream of `InputEvent`s and its only output is the ordered stream of `OutputEvent`s. That is what makes replay trivial: feed the same log into a fresh engine and the output stream, including trade ids, is bit-for-bit the same.

### Components

| Component | File | Role |
|---|---|---|
| `MatchingEngine` | `engine.hpp`, `engine.cpp` | Book, matching, validation, L2 publishing |
| `IdMap` | `id_map.hpp` | Order id to pool slot, open addressing |
| `ReferenceMatcher` | `reference.hpp`, `reference.cpp` | Test oracle, O(n) per decision |
| `MarketDataFeed` | `market_data.hpp`, `market_data.cpp` | Downstream consumer rebuilding L2 and the tape from events only |
| `StreamDigest` | `market_data.hpp` | FNV-1a digest of an output stream |
| `OrderFlow`, `Rng` | `order_flow.hpp` | Deterministic random order flow |
| event codec | `events.hpp`, `events.cpp` | Text log format, fixed-point parse/format |
| `exch` | `tools/exch.cpp` | `gen`, `replay`, `verify`, `fuzz` |

## Events

### Inputs

| Kind | Fields | Meaning |
|---|---|---|
| `N` New | id, side, type, price, qty | type is `LMT`, `MKT`, `IOC`, `FOK` or `POST` |
| `C` Cancel | id | remove a resting order |
| `M` Modify | id, price, qty | set the order's price and open quantity |

Each input gets the next sequence number when the engine processes it. The log line format is in `events.cpp` (`encode`/`decode`).

### Outputs

Every output carries the `seq` of the input that caused it. For one input the order is always the same.

1. `Accepted`, `Rejected` or `Modified` (exactly one, first)
2. `Trade` events in match order (best price first, FIFO within a price)
3. `Canceled` for an unfilled IOC, market or FOK remainder
4. `BookUpdate` (L2) events, one per price level whose aggregate changed, sorted bids first then asks, each by ascending price

`BookUpdate` carries the new absolute aggregate quantity at a level, with 0 meaning the level is gone. Absolute values rather than deltas mean a consumer that joins late only needs a snapshot, and a dropped delta cannot silently corrupt the view.

## Order semantics

| Type | Price bound | Remainder | Notes |
|---|---|---|---|
| Limit | yes | rests (good till cancel) | |
| Market | no | cancelled (`unfilled`) | price field ignored, reported as 0 |
| IOC | yes | cancelled (`unfilled`) | |
| FOK | yes | whole order cancelled (`fok_unfilled`) | fillability checked before any trade, so a failed FOK produces no trades and no L2 |
| PostOnly | yes | rests | rejected (`post_only_would_cross`) if it would trade on arrival, including locking the touch |

Trades always print at the resting (maker) order's price.

Cancel reports the open quantity that was removed. Cancelling an id that is not resting, including one that already filled, is rejected with `unknown_id`.

Modify follows the usual exchange rule:

* same price and quantity not increased: amended in place, keeps its queue position
* quantity increased or price changed: loses priority. The order is removed and re-enters as a fresh limit order with the same id, so a price change that crosses the spread trades immediately with the modified order as taker.

A modified order always becomes a plain limit order. A post-only order that is modified across the spread trades rather than being rejected; this is a simplification noted in the roadmap.

Validation, in this order: id 0 is `invalid_id`; an id already resting is `duplicate_id`; quantity must be in `[1, 2^40]`; price for non-market orders must be in `[1, 2^40]`. Id uniqueness is scoped to live orders in v0, so an id can be reused once its order is gone.

## Data structures

### Fixed point

`Price` is `int64` ticks, `Qty` is `int64` lots. Floating point never enters the engine. `parse_fixed` and `format_fixed` convert decimal strings at the edge and refuse inputs that would need rounding (`101.255` at scale 2 is an error, not 101.26). Limiting quantities to 2^40 keeps level totals far from int64 overflow: a level could hold 2^23 maximal orders.

### Order pool

`std::vector<Order>` plus a free list of slot indices. An order is 40 bytes after padding: id, price, qty, prev and next slot indices (uint32), side. A level is 32 bytes. Links are 32-bit indices instead of pointers, which halves link size and keeps them valid when the vector grows. Freed slots are reused LIFO, so a hot slot stays in cache.

### Price levels

Each side is a `std::vector<Level>` sorted from worst price to best, so the best level is always `back()`. A `Level` holds price, total quantity, order count and the head and tail of an intrusive doubly linked FIFO through the pool.

* matching at the touch reads `back()` and, when a level empties, `pop_back()`, both O(1)
* inserting a new level at the touch is an append; deeper levels cost a `memmove` of the levels behind them
* `find_level` checks the last 8 levels linearly (most traffic is near the touch) and falls back to binary search

### Id index

`IdMap` is an open-addressing hash table from order id to pool slot: linear probing, power-of-two capacity, load factor at most 1/2, a splitmix64 hash so sequential ids spread out, and backward-shift deletion so there are no tombstones. It exists because the cancel path is a hash lookup followed by pointer chasing, and `std::unordered_map` allocates one node per entry. The effect is measured in the id map ablation (see DEVLOG).

### L2 publishing

Before the engine changes a level it records `(side, price, quantity before)` in a small `touched_` list, once per level per input. After the input is handled it sorts that list and emits a `BookUpdate` for every level whose quantity now differs from the recorded value. This coalesces a sweep of ten orders at one level into one update and suppresses no-op changes.

## Invariants

`MatchingEngine::check_invariants()` aborts if any of these fail. Tests call it after every scenario step and periodically during randomized runs.

1. Each side's levels are strictly sorted, worst to best.
2. No level is empty: `count > 0` and head and tail are set.
3. A level's `total` equals the sum of its orders' quantities and `count` equals the number of orders in its list.
4. The FIFO list is well formed: `prev` links mirror `next` links, the last node is `tail`, and there are no cycles.
5. Every resting order has quantity > 0, and its side and price match its level.
6. The id map maps every resting order's id to its slot, and holds nothing else (`ids.size()` equals the number of resting orders).
7. Every pool slot is either resting or on the free list.
8. After any input completes, the book is not crossed: best bid < best ask.

Two cross-checks live outside the engine:

* the `ReferenceMatcher` must produce an identical output stream and an identical priority-ordered order list
* the `MarketDataFeed`, which only sees output events, must hold the same L2 book as the engine

## Trade-offs

### Chosen

| Decision | Why |
|---|---|
| Sorted `vector<Level>` per side, best at back | Real books concentrate activity near the touch. Contiguous levels are cache friendly and O(1) at the touch. |
| Intrusive FIFO with 32-bit slot indices | O(1) cancel from the middle of a queue, no allocation per order, small links. |
| Custom open-addressing id map | One probe into a flat array instead of a node allocation and pointer chase. Measured, not assumed. |
| Absolute L2 quantities, coalesced per input | Idempotent updates and one message per changed level. |
| Output events appended to a caller-owned vector | No virtual calls or callbacks in the hot path; the caller chooses what to do with them. |
| Text log format | Human readable and diffable; replay speed is not the bottleneck in v0. |
| Single thread, no locks | Matching for one symbol is inherently sequential. Parallelism belongs across symbols. |

### Rejected

| Alternative | Why not |
|---|---|
| `std::map<Price, Level>` | O(log n) at the touch, a heap node per level and poor locality. Kept as the model for the reference matcher's brute-force snapshot instead. |
| Dense array indexed by price (a price ladder) | O(1) everywhere, but needs a bounded price band and a scan to find the next best level when the touch empties. A good v1 option for a single instrument with known tick bands. |
| `std::list` or `std::deque` per level | Allocation per node or no O(1) removal from the middle. |
| `double` prices | Rounding makes equality of prices unreliable, which breaks level lookup and determinism. |
| Delta L2 updates | Smaller, but a lost message corrupts every downstream book until the next snapshot. |
| Wall-clock timestamps in events | Would make replay output differ between runs. Time priority uses arrival order, which is what the sequence number already is. |
| Binary log format | Faster to parse, but not needed yet. The event structs are plain data, so switching is a codec change only. |
