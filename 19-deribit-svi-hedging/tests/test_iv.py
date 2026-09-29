import numpy as np
from hypothesis import given, settings, strategies as st

from optvol import black76 as b
from optvol import iv


@settings(max_examples=500, deadline=None)
@given(
    st.floats(-1.0, 1.0),
    st.floats(1.0 / (365 * 24), 3.0),
    st.floats(0.02, 3.0),
    st.booleans(),
)
def test_round_trip_price(x, T, s, is_call):
    F = 100.0
    K = F * np.exp(x)
    p = b.price(F, K, T, s, is_call)
    sig, stt, _ = iv.implied_vol(p, F, K, T, is_call, return_info=True)
    if stt == iv.AT_INTRINSIC:
        # time value is below double precision relative to intrinsic
        tv = p - max((F - K) if is_call else (K - F), 0.0)
        assert tv < 1e-12 * F
        return
    assert stt in (iv.OK, iv.BRENT)
    p2 = b.price(F, K, T, sig, is_call)
    assert abs(p2 - p) < 1e-10 * F


def test_round_trip_vectorized_large():
    rng = np.random.default_rng(7)
    n = 200_000
    F = 100.0
    K = F * np.exp(rng.uniform(-1, 1, n))
    T = rng.uniform(1 / 365 / 24, 3, n)
    s = rng.uniform(0.02, 3, n)
    cp = rng.random(n) < 0.5
    p = b.price(F, K, T, s, cp)
    sig, stt, it = iv.implied_vol(p, F, K, T, cp, return_info=True)
    ok = np.isin(stt, [iv.OK, iv.BRENT])
    at_intr = stt == iv.AT_INTRINSIC
    intrinsic = np.where(cp, np.maximum(F - K, 0), np.maximum(K - F, 0))
    # AT_INTRINSIC only where the time value is lost to double rounding
    assert np.all(p[at_intr] - intrinsic[at_intr] < 1e-12 * F)
    assert (ok | at_intr).all()
    p2 = b.price(F, K[ok], T[ok], sig[ok], cp[ok])
    assert np.max(np.abs(p2 - p[ok])) < 1e-10 * F
    # where vega is not negligible the vol itself is recovered
    vega = b.greeks(F, K, T, s, cp)["vega"]
    good = ok & (vega > 1e-3)
    assert np.max(np.abs(sig[good] - s[good])) < 1e-8


def test_bounds_are_flagged():
    F, K, T = 100.0, 90.0, 0.5
    sig, stt, _ = iv.implied_vol(
        np.array([5.0, 101.0, 10.0, 0.0, 95.0]),
        F, K, T,
        np.array([True, True, True, False, False]),
        return_info=True,
    )
    assert stt[0] == iv.BELOW_INTRINSIC and np.isnan(sig[0])  # call below F - K
    assert stt[1] == iv.ABOVE_UPPER and np.isnan(sig[1])  # call above F
    assert stt[2] == iv.AT_INTRINSIC and sig[2] == 0.0  # exactly intrinsic
    assert stt[3] == iv.AT_INTRINSIC  # OTM put worth zero
    assert stt[4] == iv.ABOVE_UPPER  # put above K
    _, stt2, _ = iv.implied_vol(1.0, -1.0, K, T, True, return_info=True)
    assert stt2 == iv.BAD_INPUT


def test_itm_and_otm_give_same_vol():
    F, K, T, s = 100.0, 80.0, 0.25, 0.6
    c = b.price(F, K, T, s, True)
    p = b.price(F, K, T, s, False)
    assert abs(iv.implied_vol(c, F, K, T, True) - s) < 1e-10
    assert abs(iv.implied_vol(p, F, K, T, False) - s) < 1e-10
