import numpy as np
import pytest

from marketsim.orderbook import BUY, SELL, OrderBook


def test_empty_book():
    b = OrderBook()
    assert b.best_bid() is None and b.best_ask() is None
    assert b.mid() is None and b.spread() is None
    assert b.submit_market(1, BUY, 5) == []


def test_resting_orders_and_quotes():
    b = OrderBook()
    b.submit_limit(1, BUY, 99, 3)
    b.submit_limit(1, BUY, 100, 2)
    b.submit_limit(2, SELL, 102, 4)
    assert b.best_bid() == 100 and b.best_ask() == 102
    assert b.mid() == 101.0 and b.spread() == 2
    assert b.depth(BUY) == [(100, 2), (99, 3)]
    b.check_invariants()


def test_price_time_priority_and_maker_price():
    b = OrderBook()
    o1, _ = b.submit_limit(1, SELL, 101, 2)
    o2, _ = b.submit_limit(2, SELL, 101, 2)
    o3, _ = b.submit_limit(3, SELL, 100, 1)
    oid, fills = b.submit_limit(9, BUY, 105, 4)
    assert oid is None  # fully filled, nothing rests
    # Best price first (100), then FIFO at 101; executions at maker prices.
    assert [(f.maker_order_id, f.price, f.qty) for f in fills] == [(o3, 100, 1), (o1, 101, 2), (o2, 101, 1)]
    assert b.orders[o2].qty == 1
    b.check_invariants()


def test_limit_remainder_rests_at_limit():
    b = OrderBook()
    b.submit_limit(1, SELL, 101, 2)
    oid, fills = b.submit_limit(2, BUY, 101, 5)
    assert sum(f.qty for f in fills) == 2
    assert b.orders[oid].qty == 3 and b.best_bid() == 101 and b.best_ask() is None
    b.check_invariants()


def test_market_order_walks_book_and_drops_remainder():
    b = OrderBook()
    b.submit_limit(1, BUY, 100, 1)
    b.submit_limit(1, BUY, 98, 1)
    fills = b.submit_market(2, SELL, 5)
    assert [(f.price, f.qty) for f in fills] == [(100, 1), (98, 1)]
    assert len(b) == 0 and b.best_bid() is None


def test_cancel():
    b = OrderBook()
    oid, _ = b.submit_limit(1, BUY, 100, 1)
    b.submit_limit(1, BUY, 99, 1)
    assert b.cancel(oid) is True
    assert b.cancel(oid) is False  # second cancel is a no-op
    assert b.best_bid() == 99
    b.check_invariants()


def test_level_reuse_after_empty():
    # A level that empties and is re-created must still be found via the heap.
    b = OrderBook()
    oid, _ = b.submit_limit(1, BUY, 100, 1)
    b.cancel(oid)
    b.submit_limit(1, BUY, 100, 1)
    b.submit_limit(1, BUY, 99, 1)
    assert b.best_bid() == 100
    b.check_invariants()


def test_rejects_bad_input():
    b = OrderBook()
    with pytest.raises(ValueError):
        b.submit_limit(1, BUY, 100, 0)
    with pytest.raises(ValueError):
        b.submit_limit(1, 0, 100, 1)
    with pytest.raises(ValueError):
        b.submit_market(1, BUY, 0)


class NaiveBook:
    """Reference implementation: a flat list scanned on every operation."""

    def __init__(self):
        self.orders = []  # [oid, agent, side, price, qty, seq]
        self.next_id = 1

    def _best_opp(self, side):
        opp = [o for o in self.orders if o[2] == -side]
        if not opp:
            return None
        key = (lambda o: (o[3], o[5])) if side == BUY else (lambda o: (-o[3], o[5]))
        return min(opp, key=key)

    def submit(self, agent, side, price, qty):
        fills = []
        while qty > 0:
            m = self._best_opp(side)
            if m is None or (price is not None and (m[3] > price if side == BUY else m[3] < price)):
                break
            t = min(qty, m[4])
            fills.append((m[0], m[3], t))
            qty -= t
            m[4] -= t
            if m[4] == 0:
                self.orders.remove(m)
        oid = None
        if qty > 0 and price is not None:
            oid = self.next_id
            self.next_id += 1
            self.orders.append([oid, agent, side, price, qty, oid])
        return oid, fills

    def cancel(self, oid):
        for o in self.orders:
            if o[0] == oid:
                self.orders.remove(o)
                return True
        return False


@pytest.mark.parametrize("seed", range(5))
def test_fuzz_against_naive_reference(seed):
    rng = np.random.default_rng(seed)
    fast, ref = OrderBook(), NaiveBook()
    live = []
    for _ in range(3000):
        u = rng.random()
        if u < 0.55:
            side = BUY if rng.random() < 0.5 else SELL
            price = int(100 + rng.integers(-6, 7))
            qty = int(rng.integers(1, 5))
            oid_f, fills_f = fast.submit_limit(0, side, price, qty)
            oid_r, fills_r = ref.submit(0, side, price, qty)
            if oid_f is not None:
                live.append(oid_f)
        elif u < 0.7:
            side = BUY if rng.random() < 0.5 else SELL
            qty = int(rng.integers(1, 6))
            fills_f = fast.submit_market(0, side, qty)
            _, fills_r = ref.submit(0, side, None, qty)
            oid_f = oid_r = None
        else:
            if not live:
                continue
            oid = live.pop(int(rng.integers(len(live))))
            assert fast.cancel(oid) == ref.cancel(oid)
            continue
        assert [(f.maker_order_id, f.price, f.qty) for f in fills_f] == fills_r
        assert (oid_f is None) == (oid_r is None)
        fast.check_invariants()
    # Final books agree level by level.
    for side in (BUY, SELL):
        agg = {}
        for o in ref.orders:
            if o[2] == side:
                agg[o[3]] = agg.get(o[3], 0) + o[4]
        want = sorted(agg.items(), reverse=(side == BUY))
        assert fast.depth(side, 100) == want
