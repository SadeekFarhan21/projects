"""Time each pipeline stage on the development data (median of N repeats).
Writes results/benchmark.json. Uses dev-period data only."""
from __future__ import annotations

import json
import platform
import statistics
import time

import pandas as pd

from marketpred.data import PROJECT_ROOT, load_panel
from marketpred.features import build_dataset
from marketpred.metrics import daily_ic
from marketpred.models import registry
from marketpred.portfolio import backtest, weights_quintile, weights_rank
from marketpred.walkforward import DEV_END

REPEATS = 3


def timeit(fn, repeats=REPEATS):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts, out


def main():
    res = {"machine": platform.platform(), "python": platform.python_version(), "repeats": REPEATS}
    t, ts, panel = timeit(lambda: load_panel(), 1)
    panel = panel[panel.date <= DEV_END]
    res["load_panel_s"] = t
    t, ts, ds = timeit(lambda: build_dataset(panel, end=DEV_END))
    res["build_dataset_s"] = {"median": t, "all": ts, "member_rows": len(ds.long),
                              "symbols": ds.mask.shape[1], "days": ds.mask.shape[0]}
    long = ds.long
    train = long[long.index.get_level_values("date") < pd.Timestamp("2024-01-01")]
    test = long[long.index.get_level_values("date") >= pd.Timestamp("2024-01-01")]
    res["fit_rows"], res["predict_rows"] = len(train), len(test)
    for name, f in registry().items():
        if name.startswith(("rev", "mom")):
            continue
        t_fit, _, m = timeit(lambda: f().fit(train))
        t_pred, _, pred = timeit(lambda: m.predict(test))
        res[f"model_{name}"] = {"fit_s": t_fit, "predict_s": t_pred}
    pred = registry()["ridge_a10"]().fit(train).predict(long)
    fwd = long["fwd_ret"]
    res["daily_ic_s"] = timeit(lambda: daily_ic(pred, fwd))[0]
    res["weights_rank_s"] = timeit(lambda: weights_rank(pred))[0]
    res["weights_quintile_s"] = timeit(lambda: weights_quintile(pred))[0]
    w = weights_rank(pred)
    res["backtest_s"] = timeit(lambda: backtest(w, fwd))[0]
    out = PROJECT_ROOT / "results" / "benchmark.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
