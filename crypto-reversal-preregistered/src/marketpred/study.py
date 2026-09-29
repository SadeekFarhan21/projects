"""Shared helpers for the experiment scripts: evaluate one model under both
portfolio constructions and all cost levels."""
from __future__ import annotations

import math

import pandas as pd

from .metrics import ANN, daily_ic, summarize
from .portfolio import CONSTRUCTIONS, backtest

BASE_BPS = 15.0
COST_GRID = [0.0, 5.0, 10.0, 15.0, 25.0]


def evaluate(pred: pd.Series, fwd_ret: pd.Series, lag: int = 0) -> tuple[pd.Series, dict]:
    """Returns (ic series, {construction: (Backtest, summary@BASE_BPS)})."""
    fwd = fwd_ret.reindex(pred.index)
    ic = daily_ic(pred, fwd)
    out = {}
    for cname, wfn in CONSTRUCTIONS.items():
        bt = backtest(wfn(pred), fwd, lag=lag)
        s = summarize(bt.net(BASE_BPS), bt.daily["gross"], bt.daily["turnover"], ic)
        s["sr_per_period_net"] = s["sharpe_net"] / math.sqrt(ANN)
        for c in COST_GRID:
            s[f"sharpe_net_{int(c)}bps"] = summarize(bt.net(c), bt.daily["gross"], bt.daily["turnover"])["sharpe_net"]
        out[cname] = (bt, s)
    return ic, out
