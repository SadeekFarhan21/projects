import numpy as np
from hypothesis import given, settings, strategies as st

from optvol import black76 as b

F0 = 100.0
cases = st.tuples(
    st.floats(-0.5, 0.5),  # log moneyness
    st.floats(0.01, 2.0),  # T
    st.floats(0.1, 1.5),  # sigma
    st.booleans(),
    st.floats(0.8, 1.0),  # discount factor
)


def close(fd, an, rel=1e-6, abs_floor=1e-9):
    return abs(fd - an) <= rel * abs(an) + abs_floor


@settings(max_examples=300, deadline=None)
@given(cases)
def test_put_call_parity(c):
    x, T, s, _, D = c
    K = F0 * np.exp(x)
    C = b.price(F0, K, T, s, True, D)
    P = b.price(F0, K, T, s, False, D)
    assert abs((C - P) - D * (F0 - K)) < 1e-11 * F0


@settings(max_examples=300, deadline=None)
@given(cases)
def test_greeks_match_central_differences(c):
    x, T, s, is_call, D = c
    K = F0 * np.exp(x)
    g = b.greeks(F0, K, T, s, is_call, D)
    hF, hs, hT = 1e-6 * F0, 1e-5, 1e-4 * T

    def P(F=F0, T_=T, s_=s):
        return b.price(F, K, T_, s_, is_call, D)

    def G(name, F=F0, s_=s):
        return b.greeks(F, K, T, s_, is_call, D)[name]

    fd = {
        "delta": (P(F=F0 + hF) - P(F=F0 - hF)) / (2 * hF),
        "gamma": (G("delta", F=F0 + hF) - G("delta", F=F0 - hF)) / (2 * hF),
        "vega": (P(s_=s + hs) - P(s_=s - hs)) / (2 * hs),
        "theta": -(P(T_=T + hT) - P(T_=T - hT)) / (2 * hT),
        "vanna": (G("delta", s_=s + hs) - G("delta", s_=s - hs)) / (2 * hs),
        "volga": (G("vega", s_=s + hs) - G("vega", s_=s - hs)) / (2 * hs),
    }
    # absolute floors scale with the natural size of each Greek
    floors = {"delta": 1e-9, "gamma": 1e-11, "vega": 1e-7, "theta": 1e-7, "vanna": 1e-9, "volga": 1e-6}
    for k, v in fd.items():
        assert close(v, float(g[k]), abs_floor=floors[k]), (k, v, float(g[k]))


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    n = 1000
    K = F0 * np.exp(rng.uniform(-0.5, 0.5, n))
    T = rng.uniform(0.01, 2, n)
    s = rng.uniform(0.1, 1.5, n)
    cp = rng.random(n) < 0.5
    v = b.price(F0, K, T, s, cp)
    for i in range(0, n, 97):
        assert v[i] == b.price_scalar(F0, K[i], T[i], s[i], 1.0, cp[i])


def test_deribit_coin_convention():
    # Row from the 2026-09-01 tardis sample: BTC-4SEP26-80000-P, underlying
    # 78588.67, mark_iv 38.48, mark price 0.0255 BTC, 3 days 8 hours to expiry.
    T = (1788508800 - 1788220800) / (365 * 86400)
    usd = b.price(78588.67, 80000.0, T, 0.3848, False)
    coin = b.usd_to_coin(usd, 78588.67)
    assert abs(coin - 0.0255) < 5e-5  # mark is rounded to 4 decimals


def test_expired_option_is_intrinsic():
    assert b.price(100.0, 90.0, 0.0, 0.5, True) == 10.0
    assert b.price(100.0, 90.0, 0.0, 0.5, False) == 0.0
