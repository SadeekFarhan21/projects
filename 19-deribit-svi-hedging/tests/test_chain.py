import numpy as np
import polars as pl
import pytest

from optvol import black76 as b
from optvol import chain

EXP = 1788249600000000  # 2026-09-01 08:00 UTC in microseconds


def _row(sym, is_call, K, bid, ask, F, t_us, exp=EXP):
    return dict(symbol=sym, coin="BTC", is_call=is_call, strike_price=K, bid_price=bid,
                ask_price=ask, expiration=exp, t_us=t_us, underlying_price=F, F_und=F)


def test_clean_drops_bad_quotes():
    t = EXP - 3600 * 10**6
    rows = [
        _row("A", True, 1.0, 0.01, 0.02, 1.0, t),  # good
        _row("B", True, 1.0, 0.0, 0.02, 1.0, t),  # zero bid
        _row("C", True, 1.0, 0.03, 0.02, 1.0, t),  # crossed
        _row("D", True, 1.0, 0.02, 0.02, 1.0, t),  # locked
        _row("E", True, 1.0, None, 0.02, 1.0, t),  # one sided
    ]
    df = pl.DataFrame(rows, schema_overrides={"bid_price": pl.Float64})
    c, counts = chain.clean(df)
    assert c["symbol"].to_list() == ["A"]
    assert counts == {"one_sided_or_empty": 1, "zero_bid": 1, "crossed_or_locked": 2, "kept": 1}
    assert c["T"][0] == pytest.approx(3600 / (365 * 86400))
    assert c["mid_coin"][0] == pytest.approx(0.015)


def test_forward_regression_recovers_forward():
    rng = np.random.default_rng(3)
    F, T, s = 78000.0, 10 / 365, 0.45
    K = np.arange(60000, 100001, 2000, dtype=float)
    C = b.price(F, K, T, s, True) / F  # coin premiums, Deribit convention
    P = b.price(F, K, T, s, False) / F
    noise = 2e-4
    y = (C + rng.normal(0, noise, K.size)) - (P + rng.normal(0, noise, K.size))
    y[3] += 0.05  # one bad pair, removed by the robust pass
    Fh, a, n, sd = chain.fit_forward_one(K, y, np.ones_like(K))
    assert abs(Fh / F - 1) < 5e-4
    assert abs(a - 1) < 2e-3
    assert n == K.size - 1


def test_parity_regression_exact_on_clean_prices():
    F, T, s = 2500.0, 0.2, 0.7
    K = np.linspace(1800, 3400, 17)
    y = (b.price(F, K, T, s, True) - b.price(F, K, T, s, False)) / F
    Fh, a, n, sd = chain.fit_forward_one(K, y, np.ones_like(K))
    assert Fh == pytest.approx(F, rel=1e-10)
    assert a == pytest.approx(1.0, abs=1e-10)


def test_rejects_non_0800_expiry(tmp_path):
    hdr = ("snap_ts,exchange,symbol,timestamp,local_timestamp,type,strike_price,expiration,"
           "open_interest,last_price,bid_price,bid_amount,bid_iv,ask_price,ask_amount,ask_iv,"
           "mark_price,mark_iv,underlying_index,underlying_price,delta,gamma,vega,theta,rho")
    bad_exp = EXP + 3600 * 10**6  # 09:00 UTC
    line = (f"1,deribit,BTC-1SEP26-80000-C,1,1,call,80000,{bad_exp},0,,0.01,1,50,0.02,1,55,"
            "0.015,52,BTC-1SEP26,79000,0.4,0,0,0,0")
    p = tmp_path / "x.csv"
    p.write_text(hdr + "\n" + line + "\n")
    with pytest.raises(ValueError):
        chain.load_minute_file(str(p))


def test_forward_rejected_when_parity_is_implausible():
    # 3 pairs with an intercept far from 1 (seen at the 08:00 settlement minute)
    t = EXP - 86400 * 10**6
    rows = []
    for K, C, P in ((2400.0, 0.09, 0.05), (2500.0, 0.06, 0.07), (2600.0, 0.03, 0.12)):
        rows.append(_row(f"C{K}", True, K, C - 0.001, C + 0.001, 2480.0, t))
        rows.append(_row(f"P{K}", False, K, P - 0.001, P + 0.001, 2480.0, t))
    df = pl.DataFrame(rows)
    c, _ = chain.clean(df)
    fwd = chain.fit_forwards(c)
    assert not fwd["accepted"][0]
    out = chain.attach_iv(c, fwd)
    assert (out["F"] == 2480.0).all()  # fell back to the Deribit underlying
