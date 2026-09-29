import numpy as np
import pandas as pd
import pytest


def make_panel(n_sym: int = 30, n_days: int = 400, reversal: float = 0.0,
               seed: int = 0) -> pd.DataFrame:
    """Synthetic long panel in the same schema as data.load_panel().
    `reversal` plants r_t = -reversal * r_{t-1} + noise in idiosyncratic returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")
    syms = ["BTCUSDT"] + [f"C{i:02d}USDT" for i in range(n_sym - 1)]
    mkt = rng.normal(0, 0.03, n_days)
    rows = []
    for j, s in enumerate(syms):
        eps = rng.normal(0, 0.04, n_days)
        idio = np.zeros(n_days)
        for t in range(1, n_days):
            idio[t] = -reversal * idio[t - 1] + eps[t]
        r = mkt + idio
        close = 100 * np.exp(np.cumsum(r))
        vol = np.full(n_days, 5e6 * (1 + j))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": s, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": vol / close,
            "quote_volume": vol}))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"], ignore_index=True)


@pytest.fixture
def panel():
    return make_panel()
