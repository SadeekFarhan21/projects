"""Trading agents. Each agent owns its own RNG stream and schedules its own
wake-ups on the shared scheduler."""

from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .engine import Fundamental, Market, Scheduler, Trade
from .orderbook import BUY, SELL


class Agent:
    """Base class: keeps cash and inventory from fills (every agent does)."""

    name = "agent"

    def __init__(self, market: Market, sched: Scheduler, rng: np.random.Generator) -> None:
        self.market, self.sched, self.rng = market, sched, rng
        self.id = market.register(self)
        self.cash = 0.0  # in ticks
        self.inventory = 0
        self.n_trades = 0
        self.edge_vs_fundamental = 0.0  # sum of side * qty * (V_at_trade - price)

    def on_fill(self, tr: Trade, maker: bool, order_id: int | None) -> None:
        side = -tr.taker_side if maker else tr.taker_side  # +1 means we bought
        self.inventory += side * tr.qty
        self.cash -= side * tr.qty * tr.price
        self.n_trades += 1
        self.edge_vs_fundamental += side * tr.qty * (tr.fundamental - tr.price)

    def wealth(self, mark: float) -> float:
        return self.cash + self.inventory * mark


class NoiseTaker(Agent):
    """Uninformed market orders arriving as a Poisson process, random side.

    With ``k > 0`` they are price-elastic: the order goes through with
    probability exp(-k * d), where d is how far the quote they would hit is
    beyond a slow EWMA reference of the mid. This is the fill-intensity model
    A * exp(-k * delta) behind Avellaneda-Stoikov, and it is what lets the
    market maker's inventory skew actually pull inventory back to zero.
    """

    name = "noise"

    def __init__(self, market, sched, rng, rate: float, size: int = 1,
                 k: float = 0.0, ref_tau: float = 50.0) -> None:
        super().__init__(market, sched, rng)
        self.rate, self.size, self.k, self.ref_tau = rate, size, k, ref_tau
        self.ref = market.mid()
        self._ref_t = sched.t
        self.n_arrivals = 0
        self.n_orders = 0
        if rate > 0:
            sched.after(rng.exponential(1.0 / rate), self.wake)

    def _update_ref(self) -> float:
        # Continuous-time EWMA sampled at our own arrivals.
        t = self.sched.t
        w = 1.0 - math.exp(-(t - self._ref_t) / self.ref_tau)
        self.ref += w * (self.market.mid() - self.ref)
        self._ref_t = t
        return self.ref

    def wake(self) -> None:
        self.n_arrivals += 1
        side = BUY if self.rng.random() < 0.5 else SELL
        ref = self._update_ref()
        go = True
        if self.k > 0:
            book = self.market.book
            quote = book.best_ask() if side == BUY else book.best_bid()
            if quote is not None:
                d = (quote - ref) if side == BUY else (ref - quote)
                go = self.rng.random() < math.exp(-self.k * max(d, 0.0))
        if go:
            self.market.market(self.id, side, self.size)
            self.n_orders += 1
        self.sched.after(self.rng.exponential(1.0 / self.rate), self.wake)


class InformedTrader(Agent):
    """Sees a noisy signal of the fundamental V and takes liquidity only when
    the quote is on the wrong side of the signal (by more than ``edge``)."""

    name = "informed"

    def __init__(self, market, sched, rng, fundamental: Fundamental, rate: float,
                 signal_sd: float, edge: float, size: int = 1) -> None:
        super().__init__(market, sched, rng)
        self.fund, self.rate, self.signal_sd, self.edge, self.size = fundamental, rate, signal_sd, edge, size
        self.n_wakes = 0
        self.n_orders = 0
        if rate > 0:
            sched.after(rng.exponential(1.0 / rate), self.wake)

    def wake(self) -> None:
        self.n_wakes += 1
        signal = self.fund.value + self.signal_sd * self.rng.standard_normal()
        book = self.market.book
        ask, bid = book.best_ask(), book.best_bid()
        if ask is not None and signal > ask + self.edge:
            self.market.market(self.id, BUY, self.size)
            self.n_orders += 1
        elif bid is not None and signal < bid - self.edge:
            self.market.market(self.id, SELL, self.size)
            self.n_orders += 1
        self.sched.after(self.rng.exponential(1.0 / self.rate), self.wake)


class NoiseLiquidity(Agent):
    """Posts passive limit orders that join or sit behind the best quote, then
    cancels them after an exponential lifetime. Gives the book depth without
    setting the inside spread (the market maker does that)."""

    name = "noise_lp"

    def __init__(self, market, sched, rng, rate: float, offset_p: float, lifetime: float) -> None:
        super().__init__(market, sched, rng)
        self.rate, self.offset_p, self.lifetime = rate, offset_p, lifetime
        if rate > 0:
            sched.after(rng.exponential(1.0 / rate), self.wake)

    def wake(self) -> None:
        rng, book = self.rng, self.market.book
        side = BUY if rng.random() < 0.5 else SELL
        behind = int(rng.geometric(self.offset_p)) - 1  # 0 means join the best
        mid = self.market.mid()
        if side == BUY:
            best = book.best_bid()
            ref = best if best is not None else math.floor(mid) - 1
            price = ref - behind
            ask = book.best_ask()
            if ask is not None and price >= ask:
                price = ask - 1  # stay passive
        else:
            best = book.best_ask()
            ref = best if best is not None else math.ceil(mid) + 1
            price = ref + behind
            bid = book.best_bid()
            if bid is not None and price <= bid:
                price = bid + 1
        oid = self.market.limit(self.id, side, int(price), 1)
        if oid is not None:
            self.sched.after(rng.exponential(self.lifetime), self.market.cancel, oid)
        self.sched.after(rng.exponential(1.0 / self.rate), self.wake)


class MarketMaker(Agent):
    """Avellaneda-Stoikov style quoting around a learned fair value.

    reservation r = fair - q * gamma * sigma^2 * tau
    half spread  h = gamma * sigma^2 * tau / 2 + ln(1 + gamma / k) / gamma
                     (+ measured adverse selection per unit, if adaptive)

    tau is held constant (a rolling horizon), so quotes are stationary in time.
    The MM does not see V. It moves its fair value by a fixed step in the
    direction of every trade on the tape (a Kyle-lambda / first-order
    Glosten-Milgrom update), which is how informed flow gets into prices.
    """

    name = "mm"

    def __init__(self, market: Market, sched: Scheduler, rng, cfg: SimConfig, fundamental: Fundamental) -> None:
        super().__init__(market, sched, rng)
        self.cfg = cfg
        self.fund = fundamental  # used ONLY to log V at fill time for analysis
        self.fair = float(cfg.p0)
        self.bid_oid: int | None = None
        self.ask_oid: int | None = None
        self.as_cost = 0.0  # EWMA adverse selection per unit, ticks
        # One row per fill: t, side, price, qty, mid_before, mid_after_h, V_at_fill
        self.fills: list[list[float]] = []
        self._requote_pending = False
        if cfg.mm_learning_mode == "gm":
            belief = cfg.informed_frac if cfg.mm_alpha_belief is None else cfg.mm_alpha_belief
            self.learning = cfg.mm_learning * belief
        elif cfg.mm_learning_mode == "fixed":
            self.learning = cfg.mm_learning
        else:
            raise ValueError(f"unknown mm_learning_mode {cfg.mm_learning_mode!r}")
        var_tau = cfg.mm_gamma * cfg.mm_sigma ** 2 * cfg.mm_horizon
        self.skew_per_unit = var_tau
        self.base_half_spread = var_tau / 2 + math.log(1 + cfg.mm_gamma / cfg.mm_k) / cfg.mm_gamma
        market.trade_listeners.append(self.on_trade)
        sched.at(0.0, self._timer)

    # quoting ------------------------------------------------------------------
    def half_spread(self) -> float:
        h = self.base_half_spread
        if self.cfg.mm_adaptive:
            h += max(0.0, self.as_cost)
        return h

    def quotes(self) -> tuple[int, int]:
        r = self.fair - self.inventory * self.skew_per_unit
        h = self.half_spread()
        bid, ask = math.floor(r - h), math.ceil(r + h)
        if ask <= bid:
            ask = bid + 1
        return bid, ask

    def quoted_spread(self) -> float:
        """Our own ask - bid, or NaN if we are not quoting both sides."""
        b, a = self._side_price(self.bid_oid), self._side_price(self.ask_oid)
        return float("nan") if b is None or a is None else float(a - b)

    def _side_price(self, oid: int | None) -> int | None:
        o = self.market.book.orders.get(oid) if oid is not None else None
        return o.price if o is not None else None

    def requote(self) -> None:
        self._requote_pending = False
        cfg, book, mkt = self.cfg, self.market.book, self.market
        bid, ask = self.quotes()
        cur_bid, cur_ask = self._side_price(self.bid_oid), self._side_price(self.ask_oid)
        want_bid = self.inventory < cfg.mm_max_inventory
        want_ask = self.inventory > -cfg.mm_max_inventory
        # Pull quotes whose price changed (or whose side hit the inventory
        # limit); keep unchanged ones so they keep time priority.
        if cur_bid is not None and (cur_bid != bid or not want_bid):
            mkt.cancel(self.bid_oid)
            cur_bid = None
        if cur_ask is not None and (cur_ask != ask or not want_ask):
            mkt.cancel(self.ask_oid)
            cur_ask = None
        # Post-only: never cross the other side of the book.
        other_ask, other_bid = book.best_ask(), book.best_bid()
        if other_ask is not None and bid >= other_ask:
            bid = other_ask - 1
        if other_bid is not None and ask <= other_bid:
            ask = other_bid + 1
        if cur_bid is None:
            self.bid_oid = mkt.limit(self.id, BUY, bid, cfg.mm_size) if want_bid else None
        if cur_ask is None:
            self.ask_oid = mkt.limit(self.id, SELL, ask, cfg.mm_size) if want_ask else None

    def _schedule_requote(self) -> None:
        if not self._requote_pending:
            self._requote_pending = True
            self.sched.after(0.0, self.requote)

    def _timer(self) -> None:
        if self.cfg.mm_inventory_learning:
            self.fair -= self.cfg.mm_inventory_learning * self.inventory * self.skew_per_unit
        self.requote()
        self.sched.after(self.cfg.mm_requote_dt, self._timer)

    # events -------------------------------------------------------------------
    def on_trade(self, tr: Trade) -> None:
        # Learn from trade DIRECTION, not trade price. An earlier version used
        # fair += w * (price - fair); since our own quotes are shaded by the
        # inventory skew, that fed the skew back into fair value and caused
        # runaway 50-100 tick "crashes" with no news (see DEVLOG, Problems).
        if self.cfg.mm_learning_signal == "direction":
            self.fair += self.learning * tr.taker_side * self.base_half_spread
        else:  # "price": the original, buggy update
            self.fair += self.learning * (tr.price - self.fair)
        self._schedule_requote()

    def on_fill(self, tr: Trade, maker: bool, order_id: int | None) -> None:
        super().on_fill(tr, maker, order_id)
        side = -tr.taker_side if maker else tr.taker_side
        idx = len(self.fills)
        self.fills.append([tr.t, side, tr.price, tr.qty, tr.mid_before, math.nan, self.fund.value])
        self.sched.after(self.cfg.mm_markout_h, self._markout, idx)
        self._schedule_requote()

    def _markout(self, idx: int) -> None:
        row = self.fills[idx]
        row[5] = self.market.mid()
        # Adverse selection per unit: how far the mid moved against the fill.
        as_per_unit = -row[1] * (row[5] - row[4])
        self.as_cost += self.cfg.mm_as_ewma * (as_per_unit - self.as_cost)
