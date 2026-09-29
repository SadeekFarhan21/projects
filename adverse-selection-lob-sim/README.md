# marketsim: an agent-based market simulator

This is a small agent-based market built on a discrete-event scheduler and a price-time priority limit order book. The market has four kinds of agents:

- noise traders who send market orders as a Poisson process
- informed traders who see a noisy signal of a hidden fundamental value, which follows a random walk with jumps
- noise liquidity providers who place resting limit orders
- one Avellaneda-Stoikov market maker that skews its quotes by inventory

The simulator records the book and measures some stylized facts: fat tails, volatility clustering, spread dynamics, and a decomposition of market-maker PnL into spread capture, adverse selection and inventory carry. It also sweeps the informed fraction and compares the results to Glosten-Milgrom intuition. Every run is seeded and reproducible bit for bit.

This is project 10 in the Trading thread. See `DESIGN.md` for the architecture and `DEVLOG.md` for the case-study write-up.

### Layout

```
src/marketsim/
  orderbook.py    limit order book (heaps of price levels, OrderedDict FIFO queues)
  engine.py       event scheduler, fundamental value process, Market (order routing, tape)
  agents.py       NoiseTaker, InformedTrader, NoiseLiquidity, MarketMaker
  config.py       SimConfig: every parameter in one frozen dataclass
  simulation.py   run(cfg) wires one run together and returns a SimResult
  metrics.py      kurtosis, ACF, Hill estimator, MM PnL decomposition, summarize()
experiments/
  informed_sweep.py      alpha sweep x 3 market-maker variants x 8 seeds
  stylized_facts.py      one long run plus ablations, stylized-fact plots
  inventory_ablation.py  ablation for the two inventory fixes
  bench.py               order book ops/s and simulation events/s
tests/                   pytest suite (32 tests)
results/                 raw outputs of the runs quoted in DEVLOG.md
```

### Build

You need [uv](https://docs.astral.sh/uv/). The project is pinned to Python 3.12.

```
uv sync
```

### Test

```
uv run pytest -q
```

### Run the experiments and benchmark

```
uv run python experiments/informed_sweep.py        # ~1-4 min on 14 cores; results/sweep/
uv run python experiments/stylized_facts.py        # 4 runs of 200k time units; results/stylized/
uv run python experiments/inventory_ablation.py    # results/inventory_ablation.csv
uv run python experiments/bench.py                 # results/bench.json
```

Each script prints a summary table. That output was saved as `results/*_stdout.txt`.

### Run one simulation from Python

```python
from marketsim import SimConfig, run
from marketsim.metrics import summarize

res = run(SimConfig(seed=1, informed_frac=0.2, mm_adaptive=True, t_end=20_000))
print(summarize(res))
```

### Headline results (v0)

All numbers come from files under `results/`. DEVLOG.md has the details.

- As the informed fraction goes from 0 to 0.6, adverse selection per fill for the fixed-spread market maker rises from 0.61 to 1.27 ticks. PnL per fill falls from 2.21 to 0.88 ticks. (`results/sweep/aggregate.csv`)
- The adaptive market maker adds its measured markout loss to its half-spread. Its own quoted spread widens from 6.85 to 8.16 ticks over the same range, which is the Glosten-Milgrom response. (`results/sweep/aggregate.csv`)
- Fat tails are present but mild. At the 10-unit horizon, excess kurtosis is 2.61 with jumps and 1.09 without. Volatility clustering is short-lived: the ACF of |r| is 0.19 at lag 1 and falls inside the iid band by about lag 5. (`results/stylized/summary.json`)
- Benchmark: 33k order book ops/s and 14k simulation events/s. Both were measured single-threaded while the machine had a load average near 300 on 14 cores, so treat them as lower bounds. (`results/bench.json`)

### Status

| Milestone | Status |
|---|---|
| v0: event scheduler, Python LOB, noise/informed/LP/AS market maker, stylized facts, informed-fraction sweep | done |
| RL market maker (learn the quoting policy, compare with AS) | planned |
| Latency effects (order entry and market data delays, stale-quote sniping) | planned |
| Multiple venues (fragmentation, routing, cross-venue arbitrage) | planned |
| Calibration to real data (spread, trade rate and return moments from public tick data) | planned |
| C++ matching core shared with project 06, with Python bindings | planned |

### What v0 does not do

- There is no C++ core yet. The book is pure Python, which is fine for the experiment sizes here.
- Order sizes are fixed at one unit for takers and the market maker. There are no hidden or iceberg orders, and there is no latency: an agent's reaction lands at the same timestamp, after the event that caused it.
- There is no real-data calibration, so nothing needs to be downloaded. When calibration is added, data will be fetched by a script into `data/` (which is git-ignored) and never committed.
- Volatility clustering and fat tails are measured, not engineered. The model has no mechanism for long memory, and DEVLOG.md says so.
