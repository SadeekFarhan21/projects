"""Price-time priority limit order book.

Prices are integer ticks. Each side keeps:
  * ``levels``: dict price -> OrderedDict[order_id, Order]. Insertion order
    is time priority; the head is O(1) to read and any order is O(1) to
    remove. (A plain dict is not enough: repeatedly deleting from the front
    of a dict leaves dummy slots that iteration must skip, which made the
    head lookup O(n) on deep levels. See DEVLOG, Problems.)
  * a heap of prices for O(log n) best-price lookup. Empty levels are removed
    from ``levels`` immediately but their heap entries are deleted lazily: the
    best-price accessor pops heap tops whose price no longer has a level.

Trades always execute at the resting (maker) order's price.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass

BUY = 1
SELL = -1


@dataclass(slots=True)
class Order:
    order_id: int
    agent_id: int
    side: int  # BUY or SELL
    price: int
    qty: int  # remaining quantity, always > 0 while resting


@dataclass(slots=True, frozen=True)
class Fill:
    maker_order_id: int
    maker_agent: int
    taker_agent: int
    taker_side: int  # side of the aggressor; the maker is on the other side
    price: int
    qty: int


class _Side:
    """One side of the book. ``sign`` is +1 for bids, -1 for asks.

    The heap stores ``-sign * price`` so that heap[0] is always the best
    price: highest bid (stored negated) or lowest ask.
    """

    __slots__ = ("sign", "levels", "heap", "level_qty")

    def __init__(self, sign: int) -> None:
        self.sign = sign
        self.levels: dict[int, OrderedDict[int, Order]] = {}
        self.level_qty: dict[int, int] = {}
        self.heap: list[int] = []

    def best(self) -> int | None:
        heap, levels = self.heap, self.levels
        while heap:
            price = -self.sign * heap[0]
            if price in levels:
                return price
            heapq.heappop(heap)  # stale entry for an emptied level
        return None

    def add(self, order: Order) -> None:
        level = self.levels.get(order.price)
        if level is None:
            level = OrderedDict()
            self.levels[order.price] = level
            self.level_qty[order.price] = 0
            heapq.heappush(self.heap, -self.sign * order.price)
        level[order.order_id] = order
        self.level_qty[order.price] += order.qty

    def remove(self, order: Order) -> None:
        level = self.levels[order.price]
        del level[order.order_id]
        self.level_qty[order.price] -= order.qty
        if not level:
            del self.levels[order.price]
            del self.level_qty[order.price]


class OrderBook:
    def __init__(self) -> None:
        self.bids = _Side(BUY)
        self.asks = _Side(SELL)
        self.orders: dict[int, Order] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ queries
    def best_bid(self) -> int | None:
        return self.bids.best()

    def best_ask(self) -> int | None:
        return self.asks.best()

    def mid(self) -> float | None:
        b, a = self.bids.best(), self.asks.best()
        if b is None or a is None:
            return None
        return 0.5 * (b + a)

    def spread(self) -> int | None:
        b, a = self.bids.best(), self.asks.best()
        if b is None or a is None:
            return None
        return a - b

    def depth(self, side: int, n_levels: int = 5) -> list[tuple[int, int]]:
        """Top ``n_levels`` (price, total qty) on one side, best first."""
        s = self.bids if side == BUY else self.asks
        prices = sorted(s.levels, reverse=(side == BUY))[:n_levels]
        return [(p, s.level_qty[p]) for p in prices]

    def __len__(self) -> int:
        return len(self.orders)

    # --------------------------------------------------------------- mutations
    def _match(self, agent_id: int, side: int, limit: int | None, qty: int) -> tuple[int, list[Fill]]:
        """Match an incoming order against the opposite side.

        ``limit`` None means a market order (no price limit). Returns the
        unfilled remainder and the list of fills.
        """
        opp = self.asks if side == BUY else self.bids
        fills: list[Fill] = []
        while qty > 0:
            best = opp.best()
            if best is None:
                break
            if limit is not None and (best > limit if side == BUY else best < limit):
                break
            level = opp.levels[best]
            # Walk the level FIFO by always taking the current head.
            while qty > 0 and level:
                oid = next(iter(level))
                maker = level[oid]
                traded = min(qty, maker.qty)
                fills.append(Fill(oid, maker.agent_id, agent_id, side, best, traded))
                qty -= traded
                if traded == maker.qty:
                    opp.remove(maker)  # deletes the level dict when it empties
                    del self.orders[oid]
                else:
                    maker.qty -= traded
                    opp.level_qty[best] -= traded
        return qty, fills

    def submit_limit(self, agent_id: int, side: int, price: int, qty: int) -> tuple[int | None, list[Fill]]:
        """Submit a limit order. Returns (resting order id or None, fills)."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if side not in (BUY, SELL):
            raise ValueError("side must be BUY or SELL")
        remaining, fills = self._match(agent_id, side, int(price), qty)
        if remaining == 0:
            return None, fills
        oid = self._next_id
        self._next_id += 1
        order = Order(oid, agent_id, side, int(price), remaining)
        (self.bids if side == BUY else self.asks).add(order)
        self.orders[oid] = order
        return oid, fills

    def submit_market(self, agent_id: int, side: int, qty: int) -> list[Fill]:
        """Immediate-or-cancel at any price; any unfilled remainder is dropped."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        _, fills = self._match(agent_id, side, None, qty)
        return fills

    def cancel(self, order_id: int) -> bool:
        order = self.orders.pop(order_id, None)
        if order is None:
            return False  # already filled or cancelled; not an error
        (self.bids if order.side == BUY else self.asks).remove(order)
        return True

    # -------------------------------------------------------------- invariants
    def check_invariants(self) -> None:
        """Raise AssertionError if the book is internally inconsistent."""
        b, a = self.best_bid(), self.best_ask()
        assert b is None or a is None or b < a, f"crossed book {b} >= {a}"
        seen = 0
        for s in (self.bids, self.asks):
            assert set(s.levels) == set(s.level_qty)
            for price, level in s.levels.items():
                assert level, "empty level left in book"
                total = 0
                for oid, o in level.items():
                    assert o.qty > 0 and o.price == price and o.side == s.sign
                    assert self.orders.get(oid) is o
                    total += o.qty
                    seen += 1
                assert total == s.level_qty[price]
                assert -s.sign * price in s.heap, "level missing from heap"
        assert seen == len(self.orders)
