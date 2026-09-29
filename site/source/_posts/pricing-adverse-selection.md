---
layout: post
title: "Measuring What Informed Traders Cost a Market Maker"
tags:
  - market-microstructure
  - simulation
  - trading
  - python
description: >-
  An agent-based market where a market maker quotes against noise and informed
  traders, and informed flow eats more than half of its edge per fill.
date: 2026-09-29 02:17:10
---


I built a small agent-based market to watch one textbook claim happen in a real order book. A dealer earns the spread from uninformed traders and pays some of it back to informed ones, and as the informed share of order flow grows, a dealer who does not widen should make less money. In the simulator it does. As the informed fraction of taker arrivals goes from 0 to 0.6, adverse selection per fill for a fixed-formula market maker rises from **0.605 to 1.266 ticks**, and its PnL per fill falls from **2.210 to 0.881 ticks**, a drop of 60%. A second market maker that adds its own measured markout loss to its half-spread widens from **6.85 to 8.16 ticks** and keeps PnL per fill at 2.0 or above everywhere.

The market has four kinds of agents on one event heap. Noise traders send market orders as a Poisson process. Informed traders see a noisy signal of a hidden fundamental value that follows a random walk with jumps. Noise liquidity providers post passive limit orders. One Avellaneda-Stoikov market maker quotes around a learned fair value and skews by inventory. It never sees the fundamental, but in the default mode it is told the true informed fraction and scales its belief updates by it, so the headline numbers describe a dealer who knows the informed share exactly. Everything is pure Python and seeded, and an independent rerun reproduced the sweep, the ablation and the stylized facts byte for byte.

The market-making result was the easy part. Getting there took three bugs in the market maker's model of the world, one of which crashed the price by 98 ticks with no news, and one performance bug in CPython's `dict` that I would not have guessed. The stylized facts are more modest than the market-making story. Fat tails are present but mostly come from the jumps I put in, and volatility clustering dies within about five lags.

Code is in `projects/10-adverse-selection-lob-sim`. Every number below is read from a file in its `results/` directory, and the file is named next to the number. The exceptions are in the verification note under [reproducibility](#reproducibility), which come from the project's devlog.

*Reading note.* The problems section is the most useful part. If you only want the claim and its limits, read [results](#results) and [what i would change](#what-i-would-change).

## table of contents

- [what i wanted to build](#what-i-wanted-to-build)
- [theory](#theory)
- [architecture](#architecture)
- [implementation](#implementation)
- [problems](#problems)
- [experiments](#experiments)
- [results](#results)
- [what i would change](#what-i-would-change)
- [reproducibility](#reproducibility)

## what i wanted to build

I wanted a market small enough to reason about completely. That meant a real limit order book with price-time priority, a clock that decides who acts when, and agents whose behavior comes from the standard microstructure models rather than from tuning. The target was the core story of market making. The dealer is paid by noise, pays informed traders, and should respond to more informed flow by quoting wider.

The second goal was honesty about stylized facts. Real returns have fat tails, almost no linear autocorrelation, and long-lived autocorrelation in absolute returns. Agent-based models are often shown reproducing these, and I wanted to know which ones a model this simple produces on its own and which ones it only produces because I built them in.

The v0 scope was an event scheduler, a Python order book, four agent types, stylized-fact measurements, and a sweep over the informed fraction. The roadmap after that is an RL market maker, latency, multiple venues, calibration to real data, and a C++ matching core shared with the exchange project. None of that is in this post.

## theory

Three models do the work here, plus one accounting identity.

### Glosten-Milgrom

A competitive dealer faces a stream of traders. A fraction $\alpha$ of them know the true value $V$, and the rest buy or sell at random. The dealer cannot tell them apart, so it sets the ask to the expected value conditional on the next trade being a buy, and the bid to the expected value conditional on a sell.

$$
\text{ask} = \mathbb{E}[V \mid \text{buy}], \qquad \text{bid} = \mathbb{E}[V \mid \text{sell}]
$$

A buy is more likely to come from an informed trader when $V$ is high, so the ask sits above the prior mean. To first order the half-spread is about $\alpha D$, where $D$ is the informed trader's information advantage. Three consequences matter later. The spread exists purely because of adverse selection. It widens with $\alpha$. And each trade moves the dealer's belief by an amount proportional to $\alpha$, because a trade from a population that is mostly noise carries little information. That last point turned out to be the fix for a bug.

### Avellaneda-Stoikov

Glosten-Milgrom has no inventory. Avellaneda-Stoikov (2008) is the opposite, a risk-averse market maker with CARA utility and risk aversion $\gamma$, facing mid-price volatility $\sigma$ and fills that arrive with intensity $A e^{-k\delta}$ at distance $\delta$ from mid. The optimal quotes sit around a reservation price that is shaded by inventory $q$.

$$
r = s - q\,\gamma\sigma^2 (T - t), \qquad \delta^a + \delta^b = \gamma\sigma^2 (T - t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right)
$$

A long dealer shades both quotes down so that it is more likely to sell. I hold $T - t$ constant at a rolling horizon $\tau = 10$ time units, which makes the quotes stationary instead of collapsing toward a terminal time. With the defaults ($\gamma = 0.1$, $\sigma = 1$, $k = 0.5$, $\tau = 10$) the half-spread is

$$
h = \frac{\gamma\sigma^2\tau}{2} + \frac{1}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right) = 0.5 + 10\ln 1.2 \approx 2.32 \text{ ticks},
$$

and the skew is $\gamma\sigma^2\tau = 1$ tick per unit of inventory. Quotes are rounded outward to whole ticks, bid down and ask up, so a half-spread of 2.32 gives a quoted spread of 5 or 6 ticks depending on where the reservation price falls between ticks. The measured mean for the fixed market maker is 5.64 to 5.69 across the whole sweep, which is the first sanity check the numbers pass.

The model assumes fill intensity falls with distance from the mid. That assumption turned out to matter more than anything else in it, because it is what gives the inventory skew a restoring force.

### Kyle

In Kyle (1985) price impact is linear in signed order flow. My market maker's fair-value update ended up in this form. Each trade moves fair value by a fixed step in the direction of the aggressor, and in the Glosten-Milgrom mode that step is scaled by $\alpha$.

### PnL decomposition

For one market-maker fill with sign $s$ (+1 for a buy), price $p$, size $q$, mid $m_0$ just before the aggressing order, mid $m_h$ one markout horizon later and final mid $m_T$,

$$
\underbrace{s q (m_T - p)}_{\text{fill PnL}} = \underbrace{s q (m_0 - p)}_{\text{spread capture}} + \underbrace{s q (m_h - m_0)}_{\text{adverse selection}} + \underbrace{s q (m_T - m_h)}_{\text{inventory carry}}.
$$

It telescopes, so it is an identity. Summed over all fills, the left side is exactly the dealer's cash plus inventory marked at $m_T$, since it starts flat with no cash. I use a 10-unit horizon for $h$. The split is only as meaningful as the choice of $h$, but the sum is exact, and I test it as an identity rather than a statistical claim.

### stylized facts

Real asset returns have positive excess kurtosis with a tail index near 3, almost no linear autocorrelation beyond very short lags, and autocorrelation in $|r|$ that stays positive for a long time. Kurtosis also shrinks as the return horizon grows. These are the targets I measure against, not ones I tuned toward.

## architecture

Everything runs on one event heap keyed by `(time, seq)`. Agents act only through a `Market` object and hear about the world through callbacks.

<figure data-figure="diagram:market-sim"></figure>

Data flows one way. The hidden fundamental $V$ is read by exactly two things, the informed traders and the analysis stamps on each trade. The market maker never reads it to make a decision. That separation is what makes "edge against $V$" a ground-truth check that only a simulator can offer and a real desk cannot.

### the scheduler

The scheduler is twenty lines, and its one design choice is the `seq` counter.

```python
def at(self, t, fn, *args):
    if t < self.t:
        raise ValueError(f"cannot schedule in the past: {t} < {self.t}")
    heapq.heappush(self._heap, (t, self._seq, fn, args))
    self._seq += 1

def run(self, t_end):
    heap = self._heap
    while heap and heap[0][0] <= t_end:
        t, _, fn, args = heapq.heappop(heap)
        self.t = t
        self.n_events += 1
        fn(*args)
    self.t = t_end
```

`seq` breaks ties by insertion order, so two events at the same timestamp always run in the order they were scheduled. That gives determinism, and it means the heap never has to compare two functions, which would raise in Python.

### zero-delay reactions are scheduled, not called

When the market maker is filled, it does not requote inside the fill callback. It schedules a requote at the current time. The matching loop is still dispatching fills at that moment, and a synchronous requote would re-enter the book halfway through a match. Scheduling at `t + 0` puts the requote after everything already queued at that time, so the ordering stays well defined, and a `_requote_pending` flag collapses several fills from one aggressive order into a single requote.

### one aggregated agent per noise type

Only the total arrival rate matters for the statistics. One noise-taker agent with rate $(1 - \alpha)\lambda$ is equivalent to many small ones and much cheaper, so there is one of each agent type rather than thousands of objects.

### independent random streams

Each component gets its own generator, spawned from one `SeedSequence(seed)`. Adding a random draw inside one agent does not shift any other agent's randomness, so an ablation that changes the market maker leaves the order flow it faces unchanged up to the point where the book differs.

## implementation

### the order book

Each side of the book is a dict from price to an `OrderedDict` FIFO queue, a heap of prices, and a cached quantity per level. A global `orders` index makes cancel O(1).

```python
def best(self) -> int | None:
    heap, levels = self.heap, self.levels
    while heap:
        price = -self.sign * heap[0]
        if price in levels:
            return price
        heapq.heappop(heap)  # stale entry for an emptied level
    return None
```

Bids are stored negated so `heap[0]` is always the best price on either side. Deletion is lazy. A level that empties is removed from `levels` immediately, and its heap entry is popped the next time `best()` finds it at the top. Matching always takes the head of the best level and trades at the resting order's price.

```python
level = opp.levels[best]
while qty > 0 and level:
    oid = next(iter(level))
    maker = level[oid]
    traded = min(qty, maker.qty)
    fills.append(Fill(oid, maker.agent_id, agent_id, side, best, traded))
    ...
```

That `next(iter(level))` line is the site of Problem 5 below. I rejected a sorted price list with `bisect`, because inserting a new level is O(n), and per-price arrays over a fixed band, because $V$ wanders without bound and the band would have to move with it.

`check_invariants()` asserts that the book is never crossed, no level is empty, every cached level quantity equals the sum of its orders, the `orders` index and the levels hold the same set, and every live level has a heap entry.

### the agents

The fundamental is a Gaussian random walk with $\sigma = 0.3$ ticks per square-root time unit, plus Poisson jumps at rate 0.004 per unit with normal sizes of standard deviation 12 ticks, stepped once per time unit.

Informed traders arrive at rate $\alpha\lambda$, see $V$ plus $N(0, 0.5)$ noise, and buy or sell one unit only if the signal is outside the quote. Noise liquidity providers post one unit at or behind the best quote and cancel after an exponential lifetime with mean 30. Noise takers arrive at rate $(1 - \alpha)\lambda$ and are mildly price-elastic, for reasons that are Problem 3.

The market maker quotes one unit a side from the formulas above, is post-only, and keeps a quote whose price has not changed so it keeps queue priority. Its quoting is short.

```python
def quotes(self):
    r = self.fair - self.inventory * self.skew_per_unit
    h = self.half_spread()
    bid, ask = math.floor(r - h), math.ceil(r + h)
    if ask <= bid:
        ask = bid + 1
    return bid, ask
```

It also schedules its own markout check 10 units after each fill. That check is what the adaptive variant learns from.

```python
def _markout(self, idx):
    row = self.fills[idx]
    row[5] = self.market.mid()
    as_per_unit = -row[1] * (row[5] - row[4])   # how far the mid moved against us
    self.as_cost += self.cfg.mm_as_ewma * (as_per_unit - self.as_cost)
```

With `mm_adaptive` set, `half_spread()` adds `max(0, as_cost)` to the Avellaneda-Stoikov half-spread. This is the Glosten-Milgrom response done empirically. The maker never learns $\alpha$ for its spread. It measures what trading costs it and charges that.

### three market-maker variants

The sweep runs three variants.

- `fixed` uses the Avellaneda-Stoikov quotes, with the fair-value step scaled by $\alpha$ as in Glosten-Milgrom.
- `adaptive` is the same, plus its half-spread grows by the EWMA of its own measured 10-unit markout loss.
- `naive` uses a fixed learning rate of 0.3 whatever $\alpha$ is.

In the Glosten-Milgrom mode, which is the default, the maker is told $\alpha$, as the dealer is in the model. It never reads $V$, but it knows the informed share exactly, and the `fixed` and `adaptive` results below both assume a dealer with that knowledge. That is a reasonable stand-in for Glosten-Milgrom and a real simplification, and it is one reason the adaptive variant, whose spread comes from measured markouts rather than from $\alpha$, is the more interesting one.

### tests

There are 32 tests. The one I trust most is a fuzz test that runs 5 seeds of 3,000 random operations, 15,000 in total, against a naive reference book that keeps a flat list and scans it on every call, and compares every fill by maker id, price and quantity. The others check invariants after every simulated time unit, bit-identical determinism for a fixed seed, conservation of cash, inventory and edge across all agents, the exact PnL identity, the Avellaneda-Stoikov closed form, the variance of the fundamental's increments, and the estimators against known distributions.

## problems

Most of the time on this project went into the market maker's beliefs, not the book. Three of the eight problems below were the market maker being wrong about the world in a way that looked like a market phenomenon.

### 1. the market maker crashed the price by learning from its own quotes

My first fair-value update was the obvious one, move fair value toward each trade price.

```python
self.fair += self.learning * (tr.price - self.fair)
```

With informed traders on and jumps off, the mid still made moves of up to 98 ticks over 100 time units while $V$ barely moved. It showed up first as a kurtosis number that made no sense for a model with no jumps, and then as a price path with cliffs in it. I traced the trades around one crash. The market maker was long, so the inventory skew had shaded its bid down. Noise sellers hit that low bid. The update read the low trade price as news and lowered fair value by several ticks per trade. That lowered the bid further, the maker bought more, and the loop ran away.

The bug is that the market maker was learning from its own quotes. Its trade prices contain its inventory skew, and feeding them back into fair value turns the skew into a belief. The fix was to learn from trade direction only.

```python
if self.cfg.mm_learning_signal == "direction":
    self.fair += self.learning * tr.taker_side * self.base_half_spread
else:  # "price": the original, buggy update
    self.fair += self.learning * (tr.price - self.fair)
```

I kept the old rule behind `mm_learning_signal="price"` so the bug stays reproducible, and the stylized-facts script runs it as an ablation. In `results/stylized/summary.json` the `price_learning_bug` run has a largest 100-unit move of 98.0 ticks and a 100-unit excess kurtosis of 19.21. The fixed model with no jumps gives 11.5 ticks and 0.39. One caveat on that comparison. The bug run also has inelastic noise and no inventory learning, the configuration the bug was found in, so it differs from the no-jumps run in three settings, not one.

<figure data-figure="chart:projects/market-simulator/market-simulator-kurtosis"></figure>

### 2. adverse selection fell as informed flow rose

In my first sweep the market maker used a fixed learning rate and lost the most money at $\alpha = 0$, which is backwards. With no informed traders there should be nothing to be adversely selected by.

It was chasing noise. With nothing pulling the price back, every uninformed trade permanently moved its quotes toward the side that had just traded against it, so the mid moved against its fills and the markouts registered that as adverse selection. More informed traders helped, because they pushed the price back toward $V$.

This is exactly the Glosten-Milgrom point that belief updates should scale with $\alpha$. A trade from a population that is 95% noise should barely move the dealer. The `gm` learning mode sets the step to $w = \alpha$. The naive variant is still in the sweep to show the effect. At $\alpha = 0$ its adverse selection per fill is 1.005 ticks, against 0.605 for the Glosten-Milgrom learner (`results/sweep/aggregate.csv`). The two variants meet exactly at $\alpha = 0.3$, where both step sizes equal 0.3, and every column of the two rows in `aggregate.csv` is identical there. That coincidence doubles as a determinism check.

### 3. inventory drifted to the limit

With price-insensitive noise traders, inventory drifted to the ±50 limit and stayed near it. At $\alpha = 0$ mean absolute inventory was 29.4, and the maker sat at the limit 2.9% of the time (`results/inventory_ablation.csv`, the row with elastic noise and inventory learning both off).

The reason is that the Avellaneda-Stoikov skew only works if fills respond to price. Its fill intensity $A e^{-k\delta}$ is exactly that assumption. If noise flow ignores where the quotes are, shading them changes nothing, and inventory is a random walk with a wall at 50.

The first fix was to make noise takers mildly price-elastic, so they trade with probability $e^{-0.1 d}$, where $d$ is how far the quote they would hit is beyond a slow EWMA of the mid.

```python
if self.k > 0:
    quote = book.best_ask() if side == BUY else book.best_bid()
    if quote is not None:
        d = (quote - ref) if side == BUY else (ref - quote)
        go = self.rng.random() < math.exp(-self.k * max(d, 0.0))
```

That fixed $\alpha = 0$, where mean absolute inventory fell to 7.92. With informed flow it did not. Inventory was still 25.5 at $\alpha = 0.2$ and 27.3 at $\alpha = 0.4$. The reason took a while to see. Informed traders trade whenever $V$ is outside the quote, so they effectively pin the reservation price $r = \text{fair} - q \cdot \text{skew}$ to $V$. Whatever gap exists between the maker's fair value and $V$ then has to be absorbed by $q$. The maker's belief is wrong, and the market expresses that as inventory.

The second fix treats persistent inventory as information. Every requote tick, fair value moves by $-0.02\,q \cdot \text{skew}$. A position that will not go away means net order flow has been one-sided, and that is a signal. With both fixes, mean absolute inventory is 1.16, 1.44 and 1.62 at $\alpha$ = 0, 0.2 and 0.4, and PnL per fill barely moves (2.21, 1.81 and 1.39, against 2.50, 1.74 and 1.29 with elastic noise alone).

<figure data-figure="chart:projects/market-simulator/market-simulator-inventory"></figure>

I do not love this fix. It works, but it is a patch on a belief model that is too simple, and I come back to it under what I would change.

### 4. informed traders sometimes lost money

A test asserting that informed traders make money failed on some seeds. I had marked informed PnL at the final $V$. Informed traders never unwind, so by the end of a run they hold hundreds of units, and the mark is dominated by how $V$ drifted after they traded. In the baseline stylized run, informed PnL marked at the final $V$ is negative, while their edge at trade time is positive (`results/stylized/summary.json`).

The fix was to record each agent's edge at trade time, $\sum s q (V_t - p)$. Informed edge is then positive and noise edge negative on every run, and across all agents edge sums to zero, which is now a test. This is the only place a simulator can offer something a desk cannot, a markout against the true value.

### 5. the book was quadratic in level depth

The benchmark workload builds deep levels, with 31,603 resting orders left at the end, clustered near one price (`results/bench.json`), and throughput was poor. There were two culprits.

The first was ordinary. Matching iterated `for oid in list(level)`, which copies the whole level on every aggressive order.

The second was subtler. After I switched to reading the head with `next(iter(level))` on a plain `dict`, draining a level from the front was still slow. CPython dicts keep insertion order in a dense entries array, and deleting an entry leaves a dummy slot behind rather than compacting. Iteration from the start has to skip every dummy slot, and a FIFO queue deletes from exactly the start. So each head lookup costs time proportional to the number of orders already removed, and draining a level costs time quadratic in its depth. `OrderedDict` keeps a linked list and does not have this problem.

The microbenchmark in `results/level_container_microbench.txt` shows it plainly.

| Container | Drain 10,000 head-first | Drain 40,000 head-first |
| --- | --- | --- |
| `dict` | 0.112 s | 3.554 s |
| `OrderedDict` | 0.002 s | 0.006 s |

Four times the size costs the plain dict about 32 times the time. Levels are OrderedDicts now. The devlog records that the full sweep was bit-identical before and after the change, zero differing cells in `aggregate.csv`, which is a good regression check that FIFO semantics did not move. That comparison was not saved to `results/`, so it rests on the log.

### 6. there are two spreads

The book spread narrowed as $\alpha$ grew, from 5.48 to 4.31 ticks in the fixed market maker's runs, even though the Avellaneda-Stoikov formula does not depend on $\alpha$ at all. For a while I read that as a result.

It is stale orders. When informed flow moves the market maker's fair value, liquidity-provider orders placed against the old quotes can end up inside the new ones, and they set the inside spread until they are hit or cancel. I added the market maker's own quoted spread to the recorder, and the sweep reports both. The fixed maker's own spread stays flat at 5.64 to 5.69.

### 7. tick discreteness inflates short-horizon kurtosis

On a one-tick grid with small per-unit moves, many one-unit mid changes are exactly zero, and a distribution with a spike at zero has high kurtosis for a trivial reason. That is why I read fat tails at the 10-unit horizon and report 1, 10 and 100 units side by side.

### 8. the benchmark machine was shared

Fourteen other projects were building on the same 14-core machine, and `results/bench.json` records a one-minute load average of 291 at the start of the benchmark and 320 at the end. Wall time for the same 216-run sweep ranged from 61 to 226 s across reruns. The run that wrote `results/sweep/` recorded 61.2 s in `meta.json`, and `results/sweep_stdout.txt` holds the 225.7 s output of an earlier run with identical numbers. Under that load no timing in the committed files is a measurement of the code, which is why the throughput section below does not quote them as performance.

## experiments

Unless stated otherwise, every run uses the `SimConfig()` defaults. Taker arrivals total 1 per time unit, liquidity providers arrive at 0.5, the book is sampled every time unit, and returns are taken over 10 units.

### informed-fraction sweep

`experiments/informed_sweep.py` crosses 9 values of $\alpha$ from 0 to 0.6 with the 3 market-maker variants and 8 seeds, each run 20,000 time units long, for 216 runs. It writes per-run rows to `results/sweep/runs.csv`, means and standard errors to `results/sweep/aggregate.csv`, and the configuration to `results/sweep/meta.json`.

### stylized facts

`experiments/stylized_facts.py` runs one 200,000-unit baseline at $\alpha = 0.4$ with the fixed market maker, plus three ablations on the same seed. They are no jumps, no informed traders, and the original buggy market maker. Each gives 20,000 ten-unit returns. Outputs are `results/stylized/summary.json`, `acf.csv` and `jump_event_study.csv`.

### inventory ablation

`experiments/inventory_ablation.py` crosses noise elasticity in {0, 0.1} with inventory learning in {0, 0.02} and $\alpha$ in {0, 0.2, 0.4}, one seed per cell, and writes `results/inventory_ablation.csv`.

### benchmark

`experiments/bench.py` times five repeats of 500,000 pre-generated book operations (60% limit, 15% market, 25% cancel, sizes 1 to 4) and three 20,000-unit simulations at $\alpha = 0.2$, single-threaded. It writes `results/bench.json`.

## results

### spreads and adverse selection against informed flow

All numbers in this subsection are from `results/sweep/aggregate.csv`, means over 8 seeds, in ticks per unit filled.

| $\alpha$ | fixed PnL per fill | fixed capture | fixed adverse sel. | adaptive own spread | adaptive PnL per fill | informed edge per informed trade |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 2.210 | 2.758 | -0.605 | 6.85 | 2.803 | n/a |
| 0.05 | 2.121 | 2.730 | -0.680 | 7.00 | 2.789 | 25.05 |
| 0.20 | 1.814 | 2.649 | -0.873 | 7.39 | 2.676 | 6.47 |
| 0.40 | 1.399 | 2.478 | -1.089 | 7.80 | 2.449 | 2.93 |
| 0.60 | 0.881 | 2.157 | -1.266 | 8.16 | 2.003 | 1.85 |

The Glosten-Milgrom direction shows up cleanly. Adverse selection per fill for the fixed maker roughly doubles, from 0.605 to 1.266. Its PnL per fill falls by 60%, from 2.210 to 0.881. Spread capture falls too, from 2.758 to 2.157, probably because more of its fills come when the price is moving through its quotes. Standard errors are 0.025 or less on PnL per fill and 0.023 or less on the adaptive spread, so none of these differences is noise.

<figure data-figure="chart:projects/market-simulator/market-simulator-spreads"></figure>

The adaptive maker, which widens by its measured markout loss, quotes a spread that rises monotonically from 6.85 to 8.16. Note that it is already about 1.2 ticks wider than the fixed maker at $\alpha = 0$, where there is nobody informed to protect against. The 10-unit markout against the mid also captures the maker's own footprint. After a fill, its inventory skew and inventory learning move its own quotes, the mid moves with them, and that registers as adverse selection of about 0.6 ticks per fill even with no informed traders. The adaptive rule charges for that too. A markout against the mid is what a real desk can measure, which is why I use it, but it is not a pure measure of information.

<figure data-figure="chart:projects/market-simulator/market-simulator-pnl-per-fill"></figure>

### where the naive reading of Glosten-Milgrom breaks

One result goes against the static model. Each informed trade earns much less as $\alpha$ rises, 25.05 ticks at $\alpha = 0.05$ down to 1.85 at 0.6. More informed flow keeps the price closer to $V$. Mean $|\text{mid} - V|$ is 24.85 ticks at 0.05 and 1.63 at 0.6. So the loss per taker trade to informed traders, the break-even half-spread a zero-profit dealer would need, falls from 1.45 to 0.57.

The static formula $h \approx \alpha D$ treats the information gap $D$ as fixed. In a dynamic market $D$ shrinks as $\alpha$ grows, because informed trading is itself what closes the gap. The dealer's measured adverse selection still rises, because the informed share of trades rises faster than the gap shrinks, from 0.058 to 0.310 for the fixed maker. I did not expect this going in, and it is the result I find most interesting.

### stylized facts

From `results/stylized/summary.json` and `results/stylized/acf.csv`.

| run | excess kurtosis at 1 / 10 / 100 units | ACF of r, lag 1 | ACF of abs r, lags 1 / 5 / 20 |
| --- | --- | --- | --- |
| baseline ($\alpha$ 0.4, jumps) | 3.88 / 2.61 / 2.78 | -0.135 | 0.192 / 0.023 / -0.006 |
| no jumps | 3.51 / 1.09 / 0.39 | -0.321 | 0.171 / -0.003 / 0.003 |
| no informed traders | 2.28 / 0.78 / 0.38 | -0.211 | 0.096 / -0.007 / 0.003 |

### fat tails, mostly imported

At the 10-unit horizon the baseline has excess kurtosis 2.61 and a Hill tail index of 4.63, against roughly 3 for real equities. So the tails are fat but thinner than real ones. At 100 units kurtosis stays at 2.78 with jumps and drops to 0.39 without them. The long-horizon tails come from the jumps I put into the fundamental. They are not emergent.

Informed flow does thicken tails at the 10-unit horizon. Across the sweep, mean 10-unit kurtosis for the fixed maker is 1.06 at $\alpha = 0.2$ and 7.65 at $\alpha = 0.6$ (`results/sweep/aggregate.csv`). With many informed traders the price follows $V$ closely, jumps included, so the price inherits the jumps almost undiluted.

### volatility clustering, short-lived

The ACF of $|r|$ is 0.192 at lag 1 and falls inside the iid 95% band of ±0.014 by about lag 5. Real data stays positive for hundreds of lags. The short burst comes from price discovery after a jump and from the market maker's quote dynamics. Nothing in the model produces long memory, and I would rather say so than present the lag-1 value as a match.

### negative lag-1 autocorrelation

Linear autocorrelation at lag 1 is -0.135 in the baseline. It comes from the temporary impact of noise trades reverting. Fair value and inventory skew both move after a trade and then decay. Real data shows the same microstructure effect at short horizons.

### spread around jumps

From `results/stylized/jump_event_study.csv`, averaging over the 836 jumps with a full window around them, mean $|\text{mid} - V|$ goes from an average of 2.68 ticks over the 20 samples before a jump to 10.65 at the jump sample and 10.28 one unit after, then 4.25 fifty units later, and is back to 2.69 after 150 units. The book spread barely moves. It averages about 4.84 before a jump and peaks at 4.99 about 27 units after.

The fixed Avellaneda-Stoikov maker has no channel to widen on news. Its spread depends only on inventory, which inventory learning keeps small. A maker that widened on order-flow imbalance would show a real response, and that belongs to the RL milestone.

### market-maker PnL decomposition

For the baseline run, 71,520 fills over 200,000 units, total PnL was 97,116 ticks. Spread capture contributed 177,634, adverse selection -80,589 and inventory carry 71 (`results/stylized/summary.json`). The three terms sum to the total exactly. Adverse selection took 45% of gross spread capture. Carry is negligible because mean absolute inventory is 1.7 units. In the run with no informed traders, adverse selection per fill falls from 1.127 to 0.611, and what remains is the maker's own footprint on the mid described above.

### throughput, not yet measured

I do not have a throughput number I would stand behind. `results/bench.json` records 33,200 book operations per second and 13,854 simulation events per second, but it was taken single-threaded at a one-minute load average near 300 on a 14-core machine, and those figures say more about the machine than about the code. The same file set shows how unstable they are. The four 200,000-unit stylized runs in `results/stylized/summary.json`, at 1.18 to 1.26 million events each, recorded 31,271 to 33,389 events/s, more than twice the benchmark rate for the same code.

The independent verification gives a sense of how far off the loaded numbers are. At a load average near 6, a small smoke run of the benchmark put one 20,000-unit simulation at about 376,000 events/s, and each 200,000-unit stylized run took about 3 s against 35 to 64 s in the committed files. That smoke run was a single short simulation and its tiny book workload does not build the deep levels of the real one, so it is not a benchmark either. The honest status is that throughput has not been measured on a quiet machine, and the committed figures are very conservative lower bounds.

## what i would change

### a real belief model for the market maker

The maker should hold a Bayesian belief over $V$ instead of a Kyle-style step plus an inventory-decay term. Both fixes in Problem 3 work, but they are patches. A posterior needs a model of the informed traders' signal, which is heavier than v0 needed and is the right answer. It would also let the maker estimate $\alpha$ from the flow instead of being told it, which is the assumption the fixed and adaptive results currently rest on.

### smarter liquidity providers

Stale liquidity-provider orders inside the quote muddy every spread measurement. Either they should requote when the maker moves, or they should go and the maker should provide depth at several levels, so that "the spread" means one thing.

### a mechanism for long memory

Hawkes-process order arrivals or a regime-switching fundamental would give volatility clustering a cause. Then clustering would be a result I could switch on and off rather than an artifact of price discovery.

### a markout that separates information from footprint

The adaptive maker charges itself about 0.6 ticks per fill at $\alpha = 0$ for its own impact on the mid. A markout against a reference that excludes the maker's own quotes would separate the two.

### measure performance properly

Rerun `experiments/bench.py` in full on an idle machine before quoting any throughput, and profile the event loop before starting the C++ core. At about 120,000 events per run, the Python book is not obviously the bottleneck. The scheduler, the recorder and per-event allocation matter about as much.

### variable order sizes

Unit sizes keep the Glosten-Milgrom comparison clean, but they hide the depth and market-impact effects that the latency and multi-venue milestones will need.

## reproducibility

You need [uv](https://docs.astral.sh/uv/). The project is pinned to Python 3.12.

```bash
cd projects/10-adverse-selection-lob-sim
uv sync
uv run pytest -q                                   # 32 tests

uv run python experiments/informed_sweep.py        # 216 runs, results/sweep/
uv run python experiments/stylized_facts.py        # 4 runs of 200k units, results/stylized/
uv run python experiments/inventory_ablation.py    # results/inventory_ablation.csv
uv run python experiments/bench.py                 # results/bench.json
```

The sweep uses every core by default. Pass `--workers N` to limit it. Each script prints a summary table, and the saved output is in `results/*_stdout.txt`. A single run from Python looks like this.

```python
from marketsim import SimConfig, run
from marketsim.metrics import summarize

res = run(SimConfig(seed=1, informed_frac=0.2, mm_adaptive=True, t_end=20_000))
print(summarize(res))
```

Runs are bit-identical for the same `SimConfig`. The code is in `projects/10-adverse-selection-lob-sim`, with the book in `src/marketsim/orderbook.py`, the scheduler and market in `src/marketsim/engine.py`, the agents in `src/marketsim/agents.py`, and the metrics and PnL decomposition in `src/marketsim/metrics.py`.

### verification

An independent reviewer who did not build the project reran it on 2026-09-26, with threads capped at 2 and every process pool limited to 2 workers because the machine was shared. The 32 tests passed. The full informed-fraction sweep (216 runs) wrote an `aggregate.csv` byte identical to the committed one, and `runs.csv` matched in every column except the timing field `events_per_s`. The inventory ablation was byte identical, and the stylized-facts runs reproduced `acf.csv` and `jump_event_study.csv` byte for byte, with `summary.json` differing only in wall time and events per second. So every headline number in this post, from the 0.605 to 1.266 ticks of adverse selection to the 97,116 = 177,634 - 80,589 + 71 PnL split, holds exactly. The reviewer also corrected the sweep wall time range to 61 to 226 s. Timing was deferred. The full benchmark was not rerun, only a small smoke run, and throughput still needs to be measured on a quiet machine.
