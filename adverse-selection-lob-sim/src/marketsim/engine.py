"""Discrete-event scheduler and the market that agents talk to.

Everything that happens is an event on one heap keyed by (time, seq). ``seq``
is a global counter, so two events at the same time run in the order they were
scheduled. That makes runs deterministic given the seed, and it lets an agent
react to a fill "immediately" by scheduling at the current time without
re-entering the matching code.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .orderbook import Fill, OrderBook


class Scheduler:
    def __init__(self) -> None:
        self.t = 0.0
        self._heap: list[tuple[float, int, Callable[..., Any], tuple]] = []
        self._seq = 0
        self.n_events = 0

    def at(self, t: float, fn: Callable[..., Any], *args: Any) -> None:
        if t < self.t:
            raise ValueError(f"cannot schedule in the past: {t} < {self.t}")
        heapq.heappush(self._heap, (t, self._seq, fn, args))
        self._seq += 1

    def after(self, dt: float, fn: Callable[..., Any], *args: Any) -> None:
        self.at(self.t + dt, fn, *args)

    def run(self, t_end: float) -> None:
        heap = self._heap
        while heap and heap[0][0] <= t_end:
            t, _, fn, args = heapq.heappop(heap)
            self.t = t
            self.n_events += 1
            fn(*args)
        self.t = t_end


class Fundamental:
    """V_t: Gaussian random walk with compound Poisson jumps, stepped every dt."""

    def __init__(self, sched: Scheduler, rng: np.random.Generator, v0: float,
                 sigma: float, jump_rate: float, jump_sd: float, dt: float) -> None:
        self.sched, self.rng = sched, rng
        self.value = float(v0)
        self.sigma, self.jump_rate, self.jump_sd, self.dt = sigma, jump_rate, jump_sd, dt
        self.n_jumps = 0
        self.jump_times: list[float] = []
        sched.after(dt, self._step)

    def _step(self) -> None:
        rng = self.rng
        self.value += self.sigma * math.sqrt(self.dt) * rng.standard_normal()
        n = rng.poisson(self.jump_rate * self.dt)
        if n:
            self.value += self.jump_sd * rng.standard_normal(n).sum()
            self.n_jumps += int(n)
            self.jump_times.append(self.sched.t)
        self.sched.after(self.dt, self._step)


@dataclass(slots=True)
class Trade:
    t: float
    price: int
    qty: int
    taker_side: int
    maker_agent: int
    taker_agent: int
    mid_before: float  # book mid just before the aggressing order arrived
    fundamental: float  # true value at trade time (analysis only; agents never read it)


class Market:
    """Owns the book and routes fills to agents. Agents only use this API."""

    def __init__(self, sched: Scheduler, p0: float) -> None:
        self.sched = sched
        self.book = OrderBook()
        self.agents: list[Any] = []
        self.trades: list[Trade] = []
        self.trade_listeners: list[Callable[[Trade], None]] = []
        self._last_mid = float(p0)
        self.last_price = float(p0)
        # Set by the simulation so trades can be stamped with V for analysis.
        self.fundamental: Fundamental | None = None

    def register(self, agent: Any) -> int:
        self.agents.append(agent)
        return len(self.agents) - 1

    def mid(self) -> float:
        """Book mid, or the last two-sided mid if one side is empty."""
        m = self.book.mid()
        if m is not None:
            self._last_mid = m
        return self._last_mid

    # Order entry -------------------------------------------------------------
    def limit(self, agent_id: int, side: int, price: int, qty: int) -> int | None:
        mid_before = self.mid()
        oid, fills = self.book.submit_limit(agent_id, side, price, qty)
        self._dispatch(fills, mid_before)
        return oid

    def market(self, agent_id: int, side: int, qty: int) -> list[Fill]:
        mid_before = self.mid()
        fills = self.book.submit_market(agent_id, side, qty)
        self._dispatch(fills, mid_before)
        return fills

    def cancel(self, order_id: int) -> bool:
        return self.book.cancel(order_id)

    def _dispatch(self, fills: list[Fill], mid_before: float) -> None:
        t = self.sched.t
        v = self.fundamental.value if self.fundamental is not None else float("nan")
        for f in fills:
            tr = Trade(t, f.price, f.qty, f.taker_side, f.maker_agent, f.taker_agent, mid_before, v)
            self.trades.append(tr)
            self.last_price = f.price
            self.agents[f.maker_agent].on_fill(tr, maker=True, order_id=f.maker_order_id)
            self.agents[f.taker_agent].on_fill(tr, maker=False, order_id=None)
            for listener in self.trade_listeners:
                listener(tr)
        if fills:
            self.mid()  # refresh the cached mid after the book changed
