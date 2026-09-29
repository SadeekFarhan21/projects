"""End-to-end research pipeline: config -> PIT data -> universe -> features -> model
-> portfolio -> backtest -> tracked run.

Timing convention used everywhere: row t means "after the close of day t". Features at
row t use bars <= t. Targets decided at row t are executed at the close of row t + delay.
The training label for row t is the return from close t + delay to close t + delay + horizon,
so its information interval ends at row t + delay + horizon, which is what CV purges on.
"""

from __future__ import annotations

import copy
import tomllib
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from qrp.backtest.engine import CostModel, run_backtest
from qrp.backtest.metrics import summarize
from qrp.cv import purged_splits
from qrp.data.store import BarStore
from qrp.features.ops import op_xs_rank, rolling_mean, rolling_std
from qrp.features.spec import FeatureSpec, audit_causality, compute_features
from qrp.panel import Panel, simple_returns, split_relisted
from qrp.tracker import Run

DEFAULTS = {
    "run": {"name": "run", "runs_root": "runs"},
    "data": {"root": "data", "start": None, "end": None, "as_of": None, "split_gap_days": 3},
    "universe": {"top_k": 100, "min_history": 60, "liquidity_window": 30},
    "features": [],
    "model": {"kind": "ridge", "alpha": 10.0, "horizon": 5, "signal": None, "sign": 1.0},
    "cv": {"kind": "walk_forward", "n_splits": 8, "min_train_days": 365, "embargo": 5,
           "purge": True, "train_window": None},
    "portfolio": {"gross": 1.0, "rebalance_every": 1},
    "costs": {"fee_bps": 10.0, "slippage_bps": 5.0, "impact_coef": 0.0, "aum": 1e6},
    "backtest": {"delay": 1, "engine": "numba"},
    "checks": {"leakage_audit": True, "fail_on_leak": True},
}


# ---------------------------------------------------------------- config
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    return s


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _merge(DEFAULTS, tomllib.loads(Path(path).read_text()))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(val)
    return cfg


def _as_date(x):
    if x is None or isinstance(x, date):
        return x
    return date.fromisoformat(str(x))


# ---------------------------------------------------------------- stages
def load_panel(cfg: dict) -> tuple[Panel, str]:
    d = cfg["data"]
    store = BarStore(d["root"])
    as_of = datetime.fromisoformat(d["as_of"]) if d.get("as_of") else None
    bars = store.load(_as_date(d.get("start")), _as_date(d.get("end")), d.get("symbols"), as_of)
    if bars.height == 0:
        raise SystemExit(f"no bars in {d['root']} (run `qrp ingest` first)")
    if d.get("split_gap_days"):
        bars = split_relisted(bars, int(d["split_gap_days"]))
    return Panel.from_long(bars), store.snapshot_id(as_of)


def universe_mask(panel: Panel, top_k: int, min_history: int, liquidity_window: int
                  ) -> np.ndarray:
    """Point-in-time universe: has a bar today, enough history, and top_k by trailing
    mean dollar volume. Uses only rows <= t, so it is itself a causal feature."""
    close = panel["close"]
    has_bar = np.isfinite(close)
    history = np.cumsum(has_bar, axis=0)
    liq = rolling_mean(np.where(has_bar, panel["quote_volume"], 0.0), liquidity_window)
    ok = has_bar & (history >= min_history) & np.isfinite(liq)
    liq = np.where(ok, liq, -np.inf)
    # rank descending per row; keep the top_k eligible names
    order = np.argsort(-liq, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(close.shape[0])[:, None]
    ranks[rows, order] = np.arange(close.shape[1])[None, :]
    return ok & (ranks < top_k)


def forward_label(close: np.ndarray, delay: int, horizon: int) -> np.ndarray:
    """y[t] = close[t+delay+horizon] / close[t+delay] - 1. Forward looking by design:
    labels are only ever used as training targets and for evaluation, never as features."""
    T = close.shape[0]
    y = np.full_like(close, np.nan)
    a, b = delay, delay + horizon
    if b < T:
        y[: T - b] = close[b:] / close[a: T - b + a] - 1.0
    return y


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalized intercept. Returns [intercept, coefs...]."""
    Xc = np.column_stack([np.ones(len(X)), X])
    reg = alpha * np.eye(Xc.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ y)


def rank_weights(score: np.ndarray, mask: np.ndarray, gross: float) -> np.ndarray:
    """Dollar-neutral weights proportional to the demeaned cross-sectional rank."""
    s = np.where(mask, score, np.nan)
    rk = op_xs_rank(s)                       # in [-0.5, 0.5], NaN outside mask
    cnt = np.isfinite(rk).sum(axis=1, keepdims=True)
    mean = np.where(cnt > 0, np.nansum(rk, axis=1, keepdims=True) / np.maximum(cnt, 1), 0.0)
    rk = rk - mean
    rk = np.nan_to_num(rk, nan=0.0)
    denom = np.abs(rk).sum(axis=1, keepdims=True)
    return np.where(denom > 0, rk / np.where(denom > 0, denom, 1.0) * gross, 0.0)


def information_coefficient(score: np.ndarray, label: np.ndarray, mask: np.ndarray,
                            rows: np.ndarray) -> float:
    """Mean cross-sectional rank correlation between score and realized label."""
    ics = []
    for t in rows:
        m = mask[t] & np.isfinite(score[t]) & np.isfinite(label[t])
        if m.sum() < 10:
            continue
        a = np.argsort(np.argsort(score[t, m])).astype(float)
        b = np.argsort(np.argsort(label[t, m])).astype(float)
        ics.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(ics)) if ics else float("nan")


def run_research(cfg: dict, panel: Panel | None = None, snapshot: str = "synthetic",
                 track: bool = True) -> dict:
    """Run the pipeline. Returns a dict with metrics and arrays (and the Run if tracked)."""
    if panel is None:
        panel, snapshot = load_panel(cfg)
    T, N = panel.shape
    specs = [FeatureSpec.from_dict(f) for f in cfg["features"]]
    run = Run(cfg["run"]["runs_root"], cfg["run"]["name"], cfg) if track else None
    try:
        # 1. leakage audit on the exact feature spec this run uses
        audit = []
        if cfg["checks"]["leakage_audit"]:
            audit = audit_causality(panel, specs)
            leaks = [r for r in audit if not r.causal]
            if leaks and cfg["checks"]["fail_on_leak"]:
                from qrp.features.spec import LookAheadError
                raise LookAheadError("look-ahead in: " + ", ".join(r.feature for r in leaks))

        # 2. universe, features, labels
        u = cfg["universe"]
        U = universe_mask(panel, u["top_k"], u["min_history"], u["liquidity_window"])
        feats = compute_features(panel, specs)
        m, bt = cfg["model"], cfg["backtest"]
        delay, horizon = int(bt["delay"]), int(m["horizon"])
        label = forward_label(panel["close"], delay, horizon)
        with np.errstate(invalid="ignore"):
            label_rank = op_xs_rank(np.where(U, label, np.nan))

        # 3. walk-forward CV: fit on train rows, predict test rows
        c = cfg["cv"]
        splits = purged_splits(T, c["n_splits"], label_horizon=delay + horizon,
                               embargo=c["embargo"], kind=c["kind"],
                               min_train=c["min_train_days"], train_window=c["train_window"],
                               purge=c["purge"])
        names = [s.name for s in specs if s.name in (m.get("features") or [s.name for s in specs])]
        Xall = np.stack([feats[n] for n in names], axis=-1)            # (T, N, F)
        finite_x = np.isfinite(Xall).all(axis=-1) & U
        score = np.full((T, N), np.nan)
        test_rows = np.zeros(T, dtype=bool)
        coefs = []
        for sp in splits:
            test_rows[sp.test] = True
            if m["kind"] == "signal":
                score[sp.test] = m["sign"] * feats[m["signal"]][sp.test]
                continue
            if m["kind"] == "random":  # negative control: should show IC ~ 0, Sharpe < 0 net
                rng = np.random.default_rng(int(m.get("seed", 0)) + int(sp.test[0]))
                score[sp.test] = rng.standard_normal((len(sp.test), N))
                continue
            tr = np.zeros(T, dtype=bool)
            tr[sp.train] = True
            sel = finite_x & tr[:, None] & np.isfinite(label_rank)
            if sel.sum() < 100:
                continue
            beta = fit_ridge(Xall[sel], label_rank[sel], m["alpha"])
            coefs.append(beta.tolist())
            te = finite_x & test_rows[:, None] & np.isin(np.arange(T), sp.test)[:, None]
            pred = np.full((T, N), np.nan)
            pred[te] = beta[0] + Xall[te] @ beta[1:]
            score[sp.test] = pred[sp.test]

        # 4. portfolio and backtest
        p = cfg["portfolio"]
        W = rank_weights(score, U & test_rows[:, None], p["gross"])
        k = int(p["rebalance_every"])
        if k > 1:
            hold = (np.arange(T) - np.argmax(test_rows)) // k * k + np.argmax(test_rows)
            W = W[np.clip(hold, 0, T - 1)] * test_rows[:, None]
        costs = CostModel(**cfg["costs"])
        sigma = adv = None
        if costs.impact_coef > 0:
            sigma = rolling_std(simple_returns(panel["close"]), 20)
            adv = rolling_mean(panel["quote_volume"], 20)
        res = run_backtest(W, panel["close"], costs, delay=delay, engine=bt["engine"],
                           sigma=sigma, adv=adv)

        # 5. evaluate on out-of-sample rows only (from the first executed test trade)
        first = int(np.argmax(test_rows)) + delay
        oos = np.zeros(T, dtype=bool)
        oos[first:] = True
        metrics = summarize(res.returns, res.turnover, res.cost, res.equity, mask=oos)
        metrics["ic"] = information_coefficient(score, label, U, np.where(test_rows)[0])
        metrics["n_splits"] = len(splits)
        metrics["n_symbols"] = N
        metrics["avg_universe"] = float(U[oos].sum(axis=1).mean())
        metrics["leak_free"] = all(r.causal for r in audit) if audit else None
        out = {"metrics": metrics, "result": res, "weights": W, "score": score, "oos": oos,
               "dates": panel.dates, "audit": audit, "coefs": coefs}

        if run is not None:
            run.log_meta(data_snapshot=snapshot, T=T, N=N,
                         start=str(panel.dates[0]), end=str(panel.dates[-1]))
            run.log_metrics(metrics)
            pl.DataFrame({"date": panel.dates, "equity": res.equity, "returns": res.returns,
                          "gross_pnl": res.gross_pnl, "cost": res.cost,
                          "turnover": res.turnover, "oos": oos}
                         ).write_parquet(run.artifact_path("equity.parquet"))
            (run.artifact_path("leakage_audit.json")).write_text(
                "[" + ",".join(f'{{"feature":"{r.feature}","causal":{str(r.causal).lower()}}}'
                               for r in audit) + "]")
            run.finish("ok")
            out["run"] = run
        return out
    except BaseException:
        if run is not None:
            run.finish("failed")
        raise
