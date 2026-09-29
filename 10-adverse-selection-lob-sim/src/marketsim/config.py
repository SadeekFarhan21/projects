"""Simulation parameters. All prices are in integer ticks, all times in
abstract simulation time units (think of one unit as roughly one second)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class SimConfig:
    seed: int = 0
    t_end: float = 20_000.0
    sample_dt: float = 1.0  # how often the recorder snapshots the book
    p0: int = 10_000  # initial fundamental and fair value, in ticks

    # Fundamental value: Gaussian random walk plus compound Poisson jumps,
    # updated every fund_dt.
    fund_dt: float = 1.0
    fund_sigma: float = 0.3  # diffusion, ticks per sqrt(time unit)
    jump_rate: float = 0.004  # jumps per time unit
    jump_sd: float = 12.0  # jump size standard deviation, ticks

    # Liquidity takers. Total arrival rate is split between informed and noise
    # traders by informed_frac (the Glosten-Milgrom "alpha").
    taker_rate: float = 1.0
    informed_frac: float = 0.1
    # Noise takers are mildly price-elastic, as Avellaneda-Stoikov assumes:
    # a noise buyer who finds the ask d ticks above a slow public reference
    # price (EWMA of the mid, time constant noise_ref_tau) trades with
    # probability exp(-noise_k * max(d, 0)); symmetric for sellers.
    # noise_k = 0 gives the classic price-insensitive noise trader.
    noise_k: float = 0.1
    noise_ref_tau: float = 50.0
    informed_signal_sd: float = 0.5  # noise on the informed trader's view of V
    informed_edge: float = 0.0  # min expected edge (ticks) before trading

    # Noise liquidity providers: join or sit behind the best quote on a random
    # side, then cancel after an exponential lifetime.
    lp_rate: float = 0.5
    lp_offset_p: float = 0.5  # geometric parameter for ticks behind the best
    lp_lifetime: float = 30.0

    # Avellaneda-Stoikov market maker.
    mm_enabled: bool = True
    mm_gamma: float = 0.1  # risk aversion
    mm_sigma: float = 1.0  # assumed volatility of fair value, ticks/sqrt(unit)
    mm_k: float = 0.5  # order-arrival decay in the AS intensity A*exp(-k*delta)
    mm_horizon: float = 10.0  # the (T - t) term, held constant (rolling horizon)
    mm_size: int = 1
    mm_requote_dt: float = 1.0
    # Fair value update per trade: fair += w * taker_side * base_half_spread.
    # "gm" mode: w = mm_learning * belief about the informed fraction, the
    # first-order Bayesian (Glosten-Milgrom) update; the MM knows alpha, as in
    # GM. "fixed" mode: w = mm_learning regardless of alpha (the naive MM).
    mm_learning: float = 1.0
    mm_learning_mode: str = "gm"
    mm_alpha_belief: float | None = None  # None means "use informed_frac"
    # Inventory as information: each requote tick, move fair value by
    # -mm_inventory_learning * inventory * skew_per_unit. Persistent inventory
    # means net order flow has been one-sided, which is itself a signal.
    mm_inventory_learning: float = 0.02
    # "direction" (default) or "price". "price" is the original buggy update
    # fair += w * (trade_price - fair), kept only so the DEVLOG bug is
    # reproducible (experiments/stylized_facts.py runs it as an ablation).
    mm_learning_signal: str = "direction"
    mm_max_inventory: int = 50
    # Adaptive variant: widen the half-spread by the measured adverse
    # selection per fill (EWMA of markouts at horizon mm_markout_h).
    mm_adaptive: bool = False
    mm_markout_h: float = 10.0
    mm_as_ewma: float = 0.02

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)
