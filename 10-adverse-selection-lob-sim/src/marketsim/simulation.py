"""Wire up one run: scheduler, fundamental, book, agents, recorder."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .agents import InformedTrader, MarketMaker, NoiseLiquidity, NoiseTaker
from .config import SimConfig
from .engine import Fundamental, Market, Scheduler

SAMPLE_COLS = ("t", "fundamental", "best_bid", "best_ask", "mid", "spread", "last_price",
               "mm_inventory", "mm_wealth", "mm_quote_spread")
FILL_COLS = ("t", "side", "price", "qty", "mid_before", "mid_after_h", "fundamental")


@dataclass
class SimResult:
    config: SimConfig
    samples: dict[str, np.ndarray]  # regular snapshots every sample_dt
    mm_fills: dict[str, np.ndarray]  # one row per market-maker fill
    trades: dict[str, np.ndarray]  # full tape
    final_mid: float
    final_fundamental: float
    counters: dict[str, float] = field(default_factory=dict)
    jump_times: np.ndarray = field(default_factory=lambda: np.zeros(0))


class _Recorder:
    def __init__(self, sched: Scheduler, market: Market, fund: Fundamental, mm: MarketMaker | None, dt: float) -> None:
        self.sched, self.market, self.fund, self.mm, self.dt = sched, market, fund, mm, dt
        self.rows: list[tuple] = []
        sched.at(0.0, self.snap)

    def snap(self) -> None:
        book = self.market.book
        b, a = book.best_bid(), book.best_ask()
        nan = float("nan")
        mid = self.market.mid()
        mm = self.mm
        self.rows.append((
            self.sched.t, self.fund.value,
            nan if b is None else b, nan if a is None else a, mid,
            nan if (a is None or b is None) else a - b,
            self.market.last_price,
            0 if mm is None else mm.inventory,
            0.0 if mm is None else mm.wealth(mid),
            nan if mm is None else mm.quoted_spread(),
        ))
        self.sched.after(self.dt, self.snap)


def run(cfg: SimConfig, check_invariants_every: int = 0) -> SimResult:
    """Run one simulation. ``check_invariants_every`` > 0 validates the book
    every that many events (slow; used by tests)."""
    ss = np.random.SeedSequence(cfg.seed)
    r_fund, r_noise, r_inf, r_lp, r_mm = (np.random.default_rng(s) for s in ss.spawn(5))

    sched = Scheduler()
    market = Market(sched, cfg.p0)
    fund = Fundamental(sched, r_fund, cfg.p0, cfg.fund_sigma, cfg.jump_rate, cfg.jump_sd, cfg.fund_dt)
    market.fundamental = fund
    # Registration order fixes agent ids: mm=0 if enabled.
    mm = MarketMaker(market, sched, r_mm, cfg, fund) if cfg.mm_enabled else None
    lp = NoiseLiquidity(market, sched, r_lp, cfg.lp_rate, cfg.lp_offset_p, cfg.lp_lifetime)
    noise = NoiseTaker(market, sched, r_noise, cfg.taker_rate * (1 - cfg.informed_frac),
                       k=cfg.noise_k, ref_tau=cfg.noise_ref_tau)
    informed = InformedTrader(market, sched, r_inf, fund, cfg.taker_rate * cfg.informed_frac,
                              cfg.informed_signal_sd, cfg.informed_edge)
    rec = _Recorder(sched, market, fund, mm, cfg.sample_dt)

    wall = time.perf_counter()
    if check_invariants_every > 0:
        # Step in small slices and validate the book between them.
        step = cfg.sample_dt
        t = 0.0
        while t < cfg.t_end:
            t = min(t + step, cfg.t_end)
            sched.run(t)
            market.book.check_invariants()
    else:
        sched.run(cfg.t_end)
    wall = time.perf_counter() - wall

    samples = {c: np.array(v, dtype=float) for c, v in zip(SAMPLE_COLS, zip(*rec.rows))}
    fills_arr = np.array(mm.fills, dtype=float).reshape(-1, len(FILL_COLS)) if mm else np.zeros((0, len(FILL_COLS)))
    final_mid = market.mid()
    # Markouts that would land after t_end are resolved at the final mid.
    unresolved = np.isnan(fills_arr[:, 5])
    fills_arr[unresolved, 5] = final_mid
    mm_fills = {c: fills_arr[:, i] for i, c in enumerate(FILL_COLS)}
    tr = market.trades
    trades = {
        "t": np.array([x.t for x in tr]), "price": np.array([x.price for x in tr], dtype=float),
        "qty": np.array([x.qty for x in tr], dtype=float),
        "taker_side": np.array([x.taker_side for x in tr], dtype=float),
        "maker_agent": np.array([x.maker_agent for x in tr]), "taker_agent": np.array([x.taker_agent for x in tr]),
    }
    agents = {"noise": noise, "informed": informed, "noise_lp": lp}
    if mm is not None:
        agents["mm"] = mm
    counters = {
        "events": sched.n_events, "wall_s": wall, "events_per_s": sched.n_events / wall if wall > 0 else float("nan"),
        "n_trades": len(tr), "n_jumps": fund.n_jumps,
        "informed_wakes": informed.n_wakes, "informed_orders": informed.n_orders, "noise_orders": noise.n_orders,
        "noise_arrivals": noise.n_arrivals,
        "sum_cash": sum(a.cash for a in market.agents), "sum_inventory": sum(a.inventory for a in market.agents),
    }
    for name, a in agents.items():
        counters[f"{name}_inventory"] = a.inventory
        counters[f"{name}_pnl_mid"] = a.wealth(final_mid)
        counters[f"{name}_pnl_fundamental"] = a.wealth(fund.value)
        counters[f"{name}_trades"] = a.n_trades
        counters[f"{name}_edge_vs_V"] = a.edge_vs_fundamental
    return SimResult(cfg, samples, mm_fills, trades, final_mid, fund.value, counters,
                     np.array(fund.jump_times))
