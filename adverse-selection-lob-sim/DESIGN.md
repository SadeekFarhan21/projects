# Design

### Architecture

```
                      +----------------------------------------------+
                      |  Scheduler: heap of (time, seq, fn, args)    |
                      |  pops the earliest event, sets t, calls fn   |
                      +----------------------------------------------+
                         ^ schedule wake-ups / cancels / markouts   |
                         |                                          v
 +-------------+   +-----+-------+   +--------------+   +----------------------+
 | Fundamental |   | NoiseTaker  |   | Informed     |   | NoiseLiquidity       |
 | V: RW+jumps |-->| Poisson mkt |   | reads V+eps, |   | joins/behind best,   |
 | step every  |   | orders,     |   | trades if    |   | cancels after Exp()  |
 | fund_dt     |   | price-elast.|   | V outside    |   +----------+-----------+
 +------+------+   +-----+-------+   | the quote    |              |
        | V (analysis    |           +------+-------+              |
        |  and informed  | market()        | market()              | limit()/cancel()
        |  only)         v                 v                       v
        |          +---------------------------------------------------------+
        |          | Market: order entry API, stamps mid_before and V on     |
        +--------->| each trade, appends to the tape, calls on_fill() on      |
                   | maker and taker, then every trade listener              |
                   |        +------------------------------------+           |
                   |        | OrderBook (price-time priority)    |           |
                   |        +------------------------------------+           |
                   +---------------------------+-----------------------------+
                         on_fill / on_trade    |    limit()/cancel()
                                               v    ^
                                  +-----------------+------------------+
                                  | MarketMaker (Avellaneda-Stoikov)   |
                                  | fair value, inventory skew,        |
                                  | requote on trades and on a timer,  |
                                  | schedules 10-unit markout checks   |
                                  +------------------------------------+
                                               |
   Recorder (every sample_dt): t, V, bid, ask, mid, spread, last, MM inventory, MM wealth, MM spread
                                               v
                        SimResult -> metrics.summarize / mm_pnl_decomposition -> experiments/*
```

Data flows one way. Agents act only through `Market` (`limit`, `market`, `cancel`) and learn about the world through callbacks (`on_fill`, `on_trade`) and read-only book queries. The fundamental value V is read by exactly two things: the informed traders, and the analysis stamps on trades and fills. The market maker never reads V to make decisions.

### Key data structures

#### Order book

- `levels: dict[price, OrderedDict[order_id, Order]]` per side. The OrderedDict is the FIFO queue for a level. Reading the head, appending and removing any order by id are all O(1).
- `heap: list[int]` per side, holding `-price` for bids and `price` for asks, so `heap[0]` is the best price. Deletion is lazy: a level that empties is removed from `levels` right away, and its heap entry is popped the next time `best()` sees it at the top.
- `level_qty: dict[price, int]` holds cached total size per level, so depth queries do not walk the queues.
- `orders: dict[order_id, Order]` is a global index, which makes cancel O(1).
- Prices are integer ticks. Trades print at the resting order's price.

#### Scheduler

The scheduler is a binary heap of `(time, seq, fn, args)`. `seq` is a global counter. It breaks ties by insertion order, so the run is deterministic, and events never have to be compared.

#### SimResult

This holds numpy column arrays for the sampled book state (`samples`), the market-maker fill log (`mm_fills`) and the full tape (`trades`), plus counters: event count, wall time, and per-agent cash, inventory, PnL and edge against V.

### Invariants

These are checked in `OrderBook.check_invariants()` and in the tests.

1. The book is never crossed after an operation completes: best bid < best ask.
2. No level is empty. Every resting order has qty > 0, and its price and side match the level it sits in.
3. `level_qty[p]` equals the sum of order quantities at p. The `orders` index and the levels hold the same set of orders.
4. Every live level has a heap entry. The heap may also hold extra stale entries.
5. Within a level, fills happen in arrival order. The randomized fuzz test compares fill-by-fill (maker id, price, qty) against a naive list-scan reference book over 15,000 random operations.

Conservation holds across the whole simulation. Every trade has a buyer and a seller, so the sum of cash over all agents is 0 and the sum of inventory is 0. Edge against V is zero-sum too.

The PnL decomposition is exact. For the market maker, spread capture + adverse selection + inventory carry equals cash + inventory x final mid, to floating point (`test_pnl_decomposition_is_exact`).

Runs are reproducible. The same `SimConfig` gives bit-identical arrays. Each agent has its own RNG stream, spawned from one `SeedSequence(seed)`, so adding draws in one agent does not shift another agent's randomness.

### Market-maker model

- Reservation price: `r = fair - q * gamma * sigma^2 * tau`.
- Half-spread: `h = gamma * sigma^2 * tau / 2 + ln(1 + gamma/k) / gamma`, plus the EWMA of measured markout loss if `mm_adaptive` is set.
- `tau` is held constant (a rolling horizon), so quotes do not collapse toward a terminal time.
- Fair value learns from the tape. Each trade moves `fair` by `w * taker_side * h_base`. In "gm" mode `w = mm_learning * alpha`: the market maker knows the informed fraction, as in Glosten-Milgrom, and the first-order Bayesian update is proportional to alpha.
- Inventory is also treated as information. Every requote tick, `fair -= beta * q * skew`, so a position that persists slowly moves the market maker's belief.
- The market maker is post-only. It never crosses the book, and it keeps a quote whose price has not changed, so the quote keeps its queue priority.

### Trade-offs chosen and rejected

#### Chosen: pure Python book with heaps and OrderedDict levels

For v0, clarity and testability matter more than speed. A run of 20k time units is about 120k events and takes seconds. Rejected alternatives:

- A sorted list of prices with bisect. Inserting a new level is O(n), and the heap is simpler.
- Per-price arrays with a fixed price band. This is fast, but it needs a band, and V wanders without bound.
- Plain `dict` levels. See DEVLOG, Problems: repeated head deletion made head lookup O(n).

The C++ core is a later milestone.

#### Chosen: one global event heap, with zero-delay reactions scheduled rather than called

When the market maker is filled, it schedules its requote at the same timestamp instead of requoting inside the fill callback. This avoids re-entering the matching engine while it is still dispatching fills. The ordering stays well defined: the requote runs after everything already queued at that time. Rejected: synchronous callbacks, which are faster but make reentrancy bugs easy.

#### Chosen: noise flow as one aggregated Poisson agent per type

Total arrival rate is what matters for the statistics, so one agent with rate `(1 - alpha) * lambda` is equivalent to many small agents and much cheaper. Rejected: thousands of agent objects.

#### Chosen: learning from trade direction, plus inventory as information

The first version updated fair value from trade prices. That fed the market maker's own inventory skew back into its beliefs, and prices ran away (see DEVLOG). The replacement is a Kyle-lambda style update. Rejected: a full Bayesian posterior over V. It is the right answer, but it needs a model of the informed traders' signal, and it is heavier than v0 needs.

#### Chosen: noise LPs that join or sit behind the best quote

This gives the book depth without letting noise set the inside spread, so the spread experiment measures the market maker. The cost is that stale LP orders left behind after the market maker moves sometimes sit inside its new quotes. That is why the book spread narrows at high alpha while the market maker's own spread does not, so both are reported. Rejected: LPs placed at a random distance from mid. They dominated the inside spread, and the market maker's response became invisible.

#### Chosen: mid-based markouts at a 10-unit horizon for the decomposition

A desk can observe the mid, and an identity makes the three terms sum exactly to PnL. Edge against the true V is also recorded as a ground-truth check that only a simulator can provide. Rejected: markouts against V as the primary measure, since no real market maker can compute them.

#### Rejected for v0: latency, variable order sizes, multiple venues, RL

These are roadmap items. Each would add a new axis of results before the base ones were trustworthy.
