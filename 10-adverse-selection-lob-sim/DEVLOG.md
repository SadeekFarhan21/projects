# DEVLOG: building a market simulator from an empty directory

### What I wanted to build

I wanted a small market I could reason about completely. It needed a real limit order book with price-time priority, a clock that decides who acts when, and a few agent types whose behavior comes from textbook microstructure models. The goal was to reproduce the core story of market making. A dealer earns the spread from uninformed traders and pays it back to informed traders. As the informed share of order flow grows, the spread a rational dealer quotes should widen, and a dealer who does not widen should make less money. I also wanted to measure, honestly, which stylized facts of real returns a model this simple does and does not produce.

The v0 scope was an event scheduler, a Python order book, four agent types (noise takers, informed takers, noise liquidity providers, an Avellaneda-Stoikov market maker), stylized-fact measurements, and a sweep over the informed fraction. Everything had to be seeded and reproducible.

### Theory

Glosten-Milgrom (1985). A competitive dealer faces a stream of traders. A fraction alpha of them know the true value V, and the rest trade randomly. The dealer sets the ask at E[V | buy] and the bid at E[V | sell]. A buy is more likely to come from an informed trader when V is high, so the ask sits above the prior mean by roughly alpha times the informed trader's information advantage. The spread exists purely because of adverse selection, it widens with alpha, and each trade moves the dealer's belief by an amount proportional to alpha. That last point became important later.

Avellaneda-Stoikov (2008). A market maker with CARA risk aversion gamma, facing mid-price volatility sigma and fills that arrive with intensity A exp(-k delta) at distance delta from mid, quotes around a reservation price r = s - q gamma sigma^2 (T - t) with total spread gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k). The q term is the inventory skew: a long dealer shades both quotes down to sell. I hold T - t constant (a rolling horizon of 10 time units), which gives stationary quotes. With the defaults (gamma 0.1, sigma 1, k 0.5, tau 10), the half-spread is 0.5 + 10 ln(1.2) = 2.32 ticks, and the skew is 1 tick per unit of inventory.

Kyle (1985). Here price impact is linear in signed order flow. My market maker's fair-value update ended up in this form: each trade moves fair value by a fixed amount in the direction of the aggressor.

PnL decomposition. Take a market-maker fill with sign s (+1 for a buy), price p, size q, mid m0 just before the aggressing order, mid m_h one markout horizon later, and final mid m_T:

    s q (m_T - p) = s q (m0 - p) + s q (m_h - m0) + s q (m_T - m_h)
                    spread capture   adverse selection   inventory carry

Summed over all fills, the left side is exactly the dealer's cash plus inventory marked at m_T. The split is therefore an identity, and I test it as one.

Stylized facts. Real asset returns have fat tails (positive excess kurtosis, tail index around 3), almost no linear autocorrelation, and positive autocorrelation of absolute returns that persists for a long time (volatility clustering). Kurtosis also shrinks as the return horizon grows (aggregational Gaussianity).

### Architecture

Everything runs on one event heap keyed by (time, seq). Agents act only through a `Market` object (`limit`, `market`, `cancel`) and hear about the world through `on_fill` and `on_trade` callbacks. The market maker requotes by scheduling an event at the current time instead of calling back into the book mid-dispatch. That keeps the matching loop free of reentrancy, and it makes the ordering well defined. The hidden fundamental V is read only by the informed traders and by analysis stamps on trades. The market maker never sees it. `DESIGN.md` has the diagram, the data structures and the invariants.

### Implementation

- `orderbook.py`: each side is a dict from price to an OrderedDict FIFO queue, a heap of prices with lazy deletion, and a cached per-level quantity. A global id index makes cancels O(1). Market orders are immediate-or-cancel. Limit orders match first, then rest the remainder.
- `engine.py`: the `Scheduler`; the `Fundamental` process (Gaussian step with sigma 0.3 ticks per sqrt time unit, plus Poisson jumps at rate 0.004 with normal sizes of sd 12 ticks); and `Market`, which stamps each trade with the pre-trade mid and V, then dispatches fills.
- `agents.py`:
  - Noise takers arrive at rate (1 - alpha) lambda and are mildly price-elastic (explained under Problems).
  - Informed traders arrive at rate alpha lambda, see V plus N(0, 0.5) noise, and buy or sell one unit only if the signal is outside the quote.
  - Noise LPs post one unit at or behind the best quote and cancel after an Exp(30) lifetime.
  - The market maker quotes one unit a side using the AS formulas, keeps unchanged quotes to keep queue priority, is post-only, and schedules its own markout checks.
- `simulation.py` spawns one RNG stream per component from `SeedSequence(seed)`, runs the heap, and returns numpy arrays.
- `metrics.py` has kurtosis, ACF, the Hill estimator, the PnL decomposition and a `summarize()` that produces the rows the experiments aggregate.
- Tests (32, all passing):
  - a fuzz test that checks the book fill-by-fill against a naive list-scan reference over 15,000 random operations
  - invariant checks after every simulated time unit
  - determinism
  - cash, inventory and edge conservation
  - the exact PnL identity
  - the AS closed form
  - the variance of the fundamental's increments
  - estimator sanity checks on known distributions

Three market-maker variants appear in the sweep:
- `fixed`: AS quotes, and the fair-value step is scaled by alpha, as in GM.
- `adaptive`: the same, plus its half-spread grows by an EWMA of its own measured 10-unit markout loss.
- `naive`: a fixed learning rate of 0.3 no matter what alpha is.

### Problems

1. Spontaneous crashes from the market maker learning from its own quotes. My first fair-value update was `fair += w * (trade_price - fair)`. With informed flow on and jumps off, the mid still made moves of up to 98 ticks over 100 time units while V barely moved. I traced trades around one crash. The market maker was long, so its bid was shaded down by the inventory skew. Noise sells hit that low bid, and the update read the low trade price as news, lowering fair value by several ticks per trade. That lowered the bid further, the market maker bought more, and the loop ran away. The fix was to learn from trade direction only: `fair += w * side * h_base`. I kept the old rule behind `mm_learning_signal="price"` so the bug stays reproducible. In `results/stylized/summary.json` (the `price_learning_bug` entry, with no jumps) the largest 100-unit move is 98.0 ticks and 100-unit excess kurtosis is 19.21. The fixed model with no jumps gives 11.5 ticks and 0.39.

2. The first market maker got less adverse selection as alpha rose, which is backwards. In my first sweep the market maker used a fixed learning rate and lost the most money at alpha = 0. It was chasing noise: with nothing pulling prices back, every uninformed trade permanently moved its quotes against the fills it had just done. More informed traders actually helped it, because they pushed the price back toward V. This is exactly the GM point that belief updates should scale with alpha. The `gm` learning mode sets w = alpha. The naive variant is still in the sweep to show the effect. At alpha = 0 its adverse selection per fill is 1.005 ticks, against 0.605 for the GM-learning market maker (`results/sweep/aggregate.csv`).

3. Inventory drifted to the ±50 limit. With price-insensitive noise traders, the AS skew has no restoring force. Skewing quotes changes nothing when noise flow ignores price. Mean |inventory| was 29.4, and the market maker sat at the limit 2.9% of the time at alpha = 0 (`results/inventory_ablation.csv`, row noise_k=0, beta=0). I made noise takers mildly price-elastic: they trade with probability exp(-0.1 d), where d is how far the quote is beyond a slow EWMA of the mid. This is the fill-intensity model AS assumes, and it fixed alpha = 0 (mean |inventory| 7.92). With informed flow, though, inventory was still 25.5 at alpha = 0.2. The reason: informed traders pin the reservation price r = fair - q skew to V, so whatever gap exists between fair and V turns into inventory. The second fix treats persistent inventory as information: each tick, fair moves by -0.02 q skew. With both fixes, mean |inventory| is 1.16, 1.44 and 1.62 at alpha = 0, 0.2 and 0.4, and PnL per fill is about the same (2.21, 1.81, 1.39 against 2.50, 1.74, 1.29 without the second fix), per the same file.

4. A test for "informed traders make money" failed on some seeds. I had marked informed PnL at the final V. Informed traders never unwind, so they end up holding hundreds of units, and the mark was dominated by V's drift after the trade. The fix was to record each agent's edge at trade time, the sum of s q (V_t - p). Informed edge is then positive and noise edge negative on every run, and across all agents it sums to zero, which is now a test.

5. The book was quadratic in level depth. The benchmark workload builds deep levels (12,732 to 31,603 resting orders left at the end of a run, clustered near one price; `results/bench.json`), and throughput was poor. First culprit: matching iterated `for oid in list(level)`, which copies the whole level on every aggressive order. Second, subtler culprit: after I switched to reading the head with `next(iter(level))` on a plain dict, CPython still has to skip the dummy slots left behind by deleted head entries. On its own, draining a 40k-entry dict from the head took 3.554 s against 0.006 s for an OrderedDict (`results/level_container_microbench.txt`). Levels are OrderedDicts now. The full sweep was bit-identical before and after the change (0 differing cells in `aggregate.csv`), which is a nice regression check that FIFO semantics did not move.

6. The book spread and the market maker's spread are different things. The book spread narrowed as alpha grew (5.48 to 4.31 ticks for the fixed market maker), even though the AS formula does not depend on alpha. The cause is stale LP orders. When informed flow moves the market maker's quotes, LP orders placed against the old quotes can end up inside the new ones. I added the market maker's own quoted spread to the recorder, and the sweep reports both.

7. Tick discreteness inflates short-horizon kurtosis. With a 1-tick grid and small per-unit moves, many 1-unit mid changes are exactly zero, and a return distribution with lots of zeros has high kurtosis for trivial reasons. That is why I read fat tails at the 10-unit horizon and report 1, 10 and 100 side by side.

8. The benchmark machine was shared. Fourteen other projects were building on the same machine, and the load average was around 290 to 320 on 14 cores during `bench.py` (the load average is recorded in `results/bench.json`). Wall time for the same 216-run sweep ranged from 61 s to 226 s across reruns in this session (`results/sweep/meta.json` records 61.2 s for the run that wrote `results/sweep/`, and `results/sweep_stdout.txt` holds the 225.7 s output of an earlier run with identical numbers). The throughput numbers below are therefore lower bounds.

### Experiments

Unless stated otherwise, every run uses `SimConfig()` defaults: taker rate 1 per time unit, LP rate 0.5, sample every 1 unit, returns over 10 units.

1. Informed-fraction sweep (`experiments/informed_sweep.py`). Alpha takes 9 values (0 to 0.6), crossed with 3 market-maker variants and 8 seeds, each run 20,000 time units long. That makes 216 runs. The run that wrote `results/sweep/` took 61.2 s on 14 workers (`results/sweep/meta.json`); an earlier run with identical numbers took 225.7 s (`results/sweep_stdout.txt`). Outputs are `results/sweep/runs.csv` (per run), `results/sweep/aggregate.csv` (means and standard errors), `results/sweep/meta.json` and `results/sweep/spread_and_pnl_vs_alpha.png`.
2. Stylized facts (`experiments/stylized_facts.py`). One 200,000-unit baseline at alpha = 0.4 with the fixed market maker, plus three ablations on the same seed: no jumps, no informed traders, and the original buggy market maker. Outputs are `results/stylized/summary.json`, `acf.csv`, `jump_event_study.csv`, and three PNGs (`returns_and_acf.png`, `spread_dynamics.png`, `mm_pnl_decomposition.png`).
3. Inventory ablation (`experiments/inventory_ablation.py`). Noise elasticity in {0, 0.1}, crossed with inventory learning in {0, 0.02} and alpha in {0, 0.2, 0.4}. Output is `results/inventory_ablation.csv`.
4. Benchmark (`experiments/bench.py`). Five repeats of 500k pre-generated book operations (60% limit, 15% market, 25% cancel), and three 20,000-unit simulations. Output is `results/bench.json`.

### Results

#### Adverse selection and spreads vs informed fraction

All numbers in this subsection are from `results/sweep/aggregate.csv`: means over 8 seeds, in ticks per unit filled.

| alpha | fixed MM PnL/fill | fixed capture | fixed adverse sel. | adaptive MM own spread | adaptive PnL/fill | informed edge per informed trade |
|---|---|---|---|---|---|---|
| 0.00 | 2.210 | 2.758 | -0.605 | 6.85 | 2.803 | n/a |
| 0.05 | 2.121 | 2.730 | -0.680 | 7.00 | 2.789 | 25.05 |
| 0.20 | 1.814 | 2.649 | -0.873 | 7.39 | 2.676 | 6.47 |
| 0.40 | 1.399 | 2.478 | -1.089 | 7.80 | 2.449 | 2.93 |
| 0.60 | 0.881 | 2.157 | -1.266 | 8.16 | 2.003 | 1.85 |

The GM direction shows up cleanly:

- Adverse selection per fill roughly doubles, from 0.605 to 1.266.
- The fixed market maker's PnL per fill falls by 60%, from 2.210 to 0.881.
- The adaptive market maker, which widens by its measured markout loss, quotes a spread that rises monotonically from 6.85 to 8.16. It keeps PnL per fill at 2.0 or above everywhere.

Standard errors are small: 0.025 or less on PnL per fill and 0.023 or less on the adaptive spread.

One result goes against the naive reading of GM. Each informed trade earns much less as alpha rises (25.05 ticks at alpha = 0.05 down to 1.85 at 0.6), because more informed flow keeps the price closer to V: mean |mid - V| is 24.85 at 0.05 and 1.63 at 0.6. Loss per taker trade to informed traders therefore falls from 1.45 to 0.57. The GM formula h = alpha D treats the information gap D as fixed, but in a dynamic market D shrinks as alpha grows. The dealer's measured adverse selection still rises, because the share of flow that is informed rises faster (the informed share of trades goes from 0.058 to 0.310 for the fixed market maker).

The book spread tells a different story from the market maker's spread. The fixed market maker's own spread stays flat at 5.64 to 5.69, while the book spread narrows from 5.48 to 4.31 (Problem 6).

#### Stylized facts

From `results/stylized/summary.json` and `results/stylized/acf.csv`, 200,000 units, with 20,000 ten-unit returns per run:

| run | excess kurtosis, h = 1 / 10 / 100 | ACF of r, lag 1 | ACF of abs r, lags 1 / 5 / 20 |
|---|---|---|---|
| baseline (alpha 0.4, jumps) | 3.88 / 2.61 / 2.78 | -0.135 | 0.192 / 0.023 / -0.006 |
| no jumps | 3.51 / 1.09 / 0.39 | -0.321 | 0.171 / -0.003 / 0.003 |
| no informed traders | 2.28 / 0.78 / 0.38 | -0.211 | 0.096 / -0.007 / 0.003 |

- Fat tails are present but mild. At 10 units the baseline has kurtosis 2.61 and a Hill tail index of 4.63; real equities sit near 3. At the 100-unit horizon, kurtosis stays at 2.78 with jumps and drops to 0.39 without them. The long-horizon tails come from the exogenous jumps and are not emergent.
- Across the sweep, informed flow thickens tails a lot. Mean 10-unit kurtosis for the fixed market maker is 1.06 at alpha = 0.2 and 7.65 at alpha = 0.6 (`results/sweep/aggregate.csv`). With many informed traders, the price jumps almost as fast as V does.
- Volatility clustering exists only at short lags. The ACF of |r| is 0.192 at lag 1 and is inside the iid 95% band (±0.014) by about lag 5. Real data stays positive for hundreds of lags. The short burst comes from price discovery after news and from the market maker's quote dynamics. Nothing in the model produces long memory.
- Linear autocorrelation is negative at lag 1 (-0.135 in the baseline). This comes from temporary impact of noise trades reverting (fair value and inventory skew both move, then decay), a microstructure effect real data also shows at short horizons.

#### Spread dynamics around jumps

From `results/stylized/jump_event_study.csv`, averaging over 836 jumps:

- |mid - V| goes from 2.68 ticks before a jump to 10.28 one unit after, then 4.25 after 50 units, and is back to 2.69 after 150 units.
- The book spread barely moves: 4.84 on average before a jump, peaking at 4.99 after.

The fixed AS market maker has no channel to widen on news. Its spread depends only on inventory, which the inventory learning keeps small. Note that the spread panel in `spread_dynamics.png` uses a zoomed y axis, so the wiggle looks bigger than it is. A market maker that widened on detected order-flow imbalance would show a real response, and that belongs in the RL milestone.

#### Market-maker PnL decomposition (baseline, 71,520 fills)

From `results/stylized/summary.json`: total PnL was 97,116 ticks, made up of spread capture 177,634, adverse selection -80,589 and inventory carry 71. Adverse selection took 45% of gross spread capture. Carry is negligible because inventory stays small (mean |inventory| 1.7). Without informed traders, adverse selection per fill falls from 1.127 to 0.611, and what remains is the market maker's own reaction to noise flow, measured by its own quote center.

#### Throughput

From `results/bench.json`, measured single-threaded at a load average near 300:

- Order book: 33,200 ops/s median (range 26,004 to 38,044 over 5 repeats of 500k operations).
- Full simulation: 13,854 events/s median over 3 seeds, about 123k events for a 20,000-unit run.

At these rates a 20k-unit run takes about 9 s. I expect several times that on an idle machine, but I did not get to measure one, so I am not claiming a number.

### What I would change

- Give the market maker a real Bayesian belief over V, instead of a Kyle-style step plus an inventory-decay hack. The two fixes in Problem 3 work, but they are patches on a belief model that is too simple.
- Make the noise LPs smarter. Their stale orders inside the quote muddy the spread measurements. A requote-on-move LP, or dropping them and giving the market maker depth at several levels, would make "the spread" mean one thing.
- Add a mechanism for long memory in volatility, such as Hawkes-process order arrivals or a regime-switching fundamental. Then clustering would be a result I could turn on and off, not an artifact of price discovery.
- Measure performance on an idle machine, and profile the event loop before starting the C++ core. At about 120k events per run, the Python book is not the bottleneck for these experiments. The scheduler, the recorder and per-event allocation matter about as much.
- Use variable order sizes. Unit sizes make GM comparisons clean, but they hide the depth and market-impact effects that the later latency and multi-venue milestones need.

### Verification

Independent verification on 2026-09-26 (evening, US Eastern), by a reviewer who did not build the project. Threads were capped at 2 (OMP, OpenBLAS, vecLib) and every process pool was limited to 2 workers because the machine was shared.

What I reproduced:

- Clean state: removed `__pycache__` and `.pytest_cache`, kept the uv environment, ran `uv sync` and `uv run pytest -q`. 32 passed.
- Informed-fraction sweep: full configuration (216 runs, 8 seeds, 20,000 time units each) with `--workers 2` and the output redirected to a scratch folder. `aggregate.csv` is byte identical to `results/sweep/aggregate.csv`. In `runs.csv` every column matches except `events_per_s`, which is a timing field. The printed table is identical to `results/sweep_stdout.txt` apart from the wall time line. So the headline numbers hold exactly: fixed market maker adverse selection 0.605 to 1.266 ticks per fill and PnL per fill 2.210 to 0.881 from alpha 0 to 0.6, adaptive quoted spread 6.85 to 8.16.
- Inventory ablation: full configuration with 2 workers. `inventory_ablation.csv` and the printed output are byte identical to the committed files.
- Stylized facts: full configuration (4 runs of 200,000 units) with 2 workers. `acf.csv` and `jump_event_study.csv` are byte identical, and `summary.json` differs only in `wall_s` and `events_per_s`. Baseline excess kurtosis 3.88 / 2.61 / 2.78 at horizons 1 / 10 / 100, ACF of |r| 0.192 at lag 1, and the PnL split of 97,116 = 177,634 - 80,589 + 71 all reproduce.
- Spot checks against `results/sweep/aggregate.csv`: 10-unit kurtosis 1.06 at alpha 0.2 and 7.65 at 0.6, informed share of trades 0.058 to 0.310, largest standard error on PnL per fill 0.0253 and on the adaptive spread 0.0230.

What I deferred:

- Throughput. I did not rerun the full benchmark. `experiments/bench.py` ran cleanly on a tiny configuration (20,000 book operations, one simulation) with output in a scratch folder. That single simulation ran at about 376,000 events/s at a load average near 6, against 13,854 events/s reported at a load near 300. The stylized runs also took about 3 s each here against 35 to 64 s in the committed files. The committed throughput numbers are therefore very conservative lower bounds, as the text says, and should be remeasured on a quiet machine before being quoted. The tiny book workload does not build the deep levels of the real one, so its ops/s is not comparable.

Fixes made:

- Experiments said the last sweep took 225.7 s on 14 workers, and Problems gave a range of 77 s to 226 s. The run that wrote `results/sweep/` recorded 61.2 s in `meta.json` (it was written at 23:35, after the 225.7 s run whose stdout is saved at 23:10). Both sentences now say so. The numbers of the two runs are identical, so no result changes.

Review notes (not fixed):

- In the default `gm` learning mode the market maker scales its fair value step by the true informed fraction alpha. It never reads V, but it is told alpha. That is a reasonable Glosten-Milgrom stand-in, and it is described in Implementation, but the fixed and adaptive results assume a dealer who knows the informed share exactly.
- `results/stylized_stdout.txt` and `results/stylized/summary.json` come from different executions (their wall times differ, 60.1 s against 37.8 s for the baseline). The numbers are the same.
- The tests are meaningful. The order book is fuzzed fill by fill against a naive list-scan reference, and cash, inventory, edge and the PnL identity are checked on real runs.
