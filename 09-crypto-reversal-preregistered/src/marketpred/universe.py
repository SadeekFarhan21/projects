"""Point-in-time universe: which symbols are tradable on each day.

Every quantity here at date t uses only rows with date <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Base assets that are not independent crypto risk: stablecoins, fiat,
# gold tokens, wrapped or staked duplicates of another asset.
EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "PAX", "USDP", "DAI", "FDUSD", "UST", "USTC",
    "SUSD", "USDS", "USDSB", "AEUR", "EURI", "XUSD", "USD1", "BFUSD", "PYUSD",
    "RLUSD", "USDE", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "BIDR", "IDRT",
    "NGN", "UAH", "ZAR", "PLN", "RON", "ARS", "JPY", "MXN", "COP", "CZK",
    "PAXG", "XAUT", "WBTC", "WBETH", "BETH", "BNSOL", "STETH", "WETH",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

MIN_HISTORY = 60
TOP_N = 50
MIN_DVOL = 1_000_000.0
DVOL_WINDOW = 30
MIN_NAMES = 20


def base_asset(symbol: str) -> str:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return base.split("#")[0]  # strip relisting episode suffix


def is_excluded(symbol: str, all_bases: set[str]) -> bool:
    base = base_asset(symbol)
    if base in EXCLUDE_BASES:
        return True
    # Leveraged tokens (BTCUP, ETHBEAR...). Only flag when the stripped stem is
    # itself a listed base, so JUP or SUPER are not caught by accident.
    for suf in LEVERAGED_SUFFIXES:
        if base.endswith(suf) and base[: -len(suf)] in all_bases:
            return True
    return False


def eligibility(close: pd.DataFrame, dvol: pd.DataFrame, top_n: int = TOP_N,
                min_history: int = MIN_HISTORY, min_dvol: float = MIN_DVOL,
                min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Boolean date x symbol mask of universe membership.

    close, dvol: wide frames (date x symbol), NaN where no bar.
    """
    all_bases = {base_asset(s) for s in close.columns}
    allowed = np.array([not is_excluded(s, all_bases) for s in close.columns])
    has_bar = close.notna()
    history = has_bar.cumsum()  # bars seen up to and including t
    med_dvol = dvol.rolling(DVOL_WINDOW, min_periods=20).median()
    ok = has_bar & (history >= min_history) & (med_dvol >= min_dvol)
    ok &= allowed[None, :]
    score = med_dvol.where(ok)
    rank = score.rank(axis=1, ascending=False, method="first")
    mask = ok & (rank <= top_n)
    thin = mask.sum(axis=1) < min_names
    mask.loc[thin, :] = False
    return mask
