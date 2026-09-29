"""Vectorized target-weight backtester with drift, linear costs and square-root impact.

Timeline for one strategy (all prices are closes, equity is in units of starting capital):

    close t:  mark to market         E-[t] = A[t-1] + sum_i D[t-1,i] * r[t,i]
              pre-trade weights      p[t]  = D[t-1] * (1 + r[t]) / E-[t]
              target weights         h[t]  = W[t - delay]   (h = p where the asset has no bar)
              trade                  dw    = h[t] - p[t]
              cost (fraction of E-)  k[t]  = sum_i |dw_i| * (lin + impact * sigma_i * sqrt(|dw_i| * E- * aum / adv_i))
              post-trade equity      A[t]  = E-[t] * (1 - k[t])
              dollar positions       D[t]  = h[t] * E-[t]   (cost is paid from cash)

Accounting identities this guarantees (and tests check):
    A[t] = A[t-1] + gross_pnl[t] - cost[t]
    gross_pnl[t] = sum_i D[t-1,i] * r[t,i]
    A[t] = cash[t] + sum_i D[t,i]   (share-level reference engine tracks cash explicitly)

The recursion is sequential in time (p[t] depends on k[t-1]) so the kernels loop over t
and vectorize over strategies and assets. Three kernels implement the same math:
    reference  pure Python in share/cash space, obviously correct, slow
    numpy      loop over t, numpy over (S, N)
    numba      compiled triple loop, parallel over strategies
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


@dataclass
class CostModel:
    fee_bps: float = 10.0        # exchange fee per unit traded
    slippage_bps: float = 5.0    # half spread / fixed slippage per unit traded
    impact_coef: float = 0.0     # square-root impact coefficient (0 disables)
    aum: float = 1e6             # starting capital in quote currency, only used by impact

    @property
    def linear(self) -> float:
        return (self.fee_bps + self.slippage_bps) * 1e-4


@dataclass
class BacktestResult:
    equity: np.ndarray       # (S, T) post-trade equity A[t], A[-1] := 1
    returns: np.ndarray      # (S, T) A[t]/A[t-1] - 1
    gross_pnl: np.ndarray    # (S, T)
    cost: np.ndarray         # (S, T) in equity units
    turnover: np.ndarray     # (S, T) sum |dw|, fraction of pre-trade equity
    gross_exposure: np.ndarray  # (S, T) sum |h|
    net_exposure: np.ndarray    # (S, T) sum h
    positions: np.ndarray | None = None  # (S, T, N) dollar positions if requested

    def squeeze(self) -> "BacktestResult":
        """Drop the strategy axis when S == 1."""
        if self.equity.shape[0] != 1:
            return self
        kw = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return BacktestResult(**kw)


def prepare_market(close: np.ndarray):
    """Returns (r, tradeable). r uses forward-filled prices so a missing bar earns 0."""
    tradeable = np.isfinite(close)
    # forward fill along time: index of the last finite row at or before t
    idx = np.where(tradeable, np.arange(len(close))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    ff = np.take_along_axis(close, idx, axis=0)
    r = np.zeros_like(close)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = ff[1:] / ff[:-1] - 1.0
    r[~np.isfinite(r)] = 0.0
    return r, tradeable


def _lagged_targets(W: np.ndarray, delay: int) -> np.ndarray:
    H = np.zeros_like(W)
    if delay == 0:
        H[:] = W
    else:
        H[:, delay:] = W[:, :-delay]
    return np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ kernels
def _kernel_numpy(H, r, tradeable, lin, impact, aum, sigma, adv, keep_pos):
    S, T, N = H.shape
    out = {k: np.zeros((S, T)) for k in ("equity", "gross_pnl", "cost", "turnover", "gexp", "nexp")}
    pos = np.zeros((S, T, N)) if keep_pos else None
    A = np.ones(S)
    D = np.zeros((S, N))
    for t in range(T):
        gpnl = D @ r[t]
        Em = A + gpnl
        alive = Em > 0
        Em_safe = np.where(alive, Em, 1.0)
        p = D * (1.0 + r[t]) / Em_safe[:, None]
        h = np.where(tradeable[t], H[:, t], p)
        h[~alive] = 0.0
        dw = np.abs(h - p)
        if impact > 0:
            unit = lin + impact * sigma[t] * np.sqrt(dw * Em_safe[:, None] * aum / adv[t])
        else:
            unit = lin
        k = (dw * unit).sum(axis=1)
        A = np.where(alive, Em * (1.0 - k), 0.0)
        D = h * Em_safe[:, None]
        D[~alive] = 0.0
        out["equity"][:, t] = A
        out["gross_pnl"][:, t] = gpnl
        out["cost"][:, t] = np.where(alive, Em * k, 0.0)
        out["turnover"][:, t] = dw.sum(axis=1)
        out["gexp"][:, t] = np.abs(h).sum(axis=1)
        out["nexp"][:, t] = h.sum(axis=1)
        if keep_pos:
            pos[:, t] = D
    return out, pos


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True, fastmath=False)
    def _kernel_numba(H, r, tradeable, lin, impact, aum, sigma, adv, keep_pos,
                      eq, gp, cs, to, ge, ne, pos):
        S, T, N = H.shape
        for s in numba.prange(S):
            A = 1.0
            D = np.zeros(N)
            h = np.zeros(N)
            for t in range(T):
                gpnl = 0.0
                for i in range(N):
                    gpnl += D[i] * r[t, i]
                Em = A + gpnl
                alive = Em > 0
                Es = Em if alive else 1.0
                k = 0.0
                turn = 0.0
                g = 0.0
                n = 0.0
                for i in range(N):
                    p = D[i] * (1.0 + r[t, i]) / Es
                    hi = H[s, t, i] if tradeable[t, i] else p
                    if not alive:
                        hi = 0.0
                    dw = abs(hi - p)
                    unit = lin
                    if impact > 0.0:
                        unit = lin + impact * sigma[t, i] * np.sqrt(dw * Es * aum / adv[t, i])
                    k += dw * unit
                    turn += dw
                    g += abs(hi)
                    n += hi
                    h[i] = hi
                if alive:
                    A = Em * (1.0 - k)
                    cs[s, t] = Em * k
                else:
                    A = 0.0
                    cs[s, t] = 0.0
                for i in range(N):
                    D[i] = h[i] * Es if alive else 0.0
                    if keep_pos:
                        pos[s, t, i] = D[i]
                eq[s, t] = A
                gp[s, t] = gpnl
                to[s, t] = turn
                ge[s, t] = g
                ne[s, t] = n


def _kernel_reference(H, close, lin, impact, aum, sigma, adv, keep_pos):
    """Share-and-cash accounting, one asset at a time. Slow on purpose: it is the oracle."""
    S, T, N = H.shape
    out = {k: np.zeros((S, T)) for k in ("equity", "gross_pnl", "cost", "turnover", "gexp", "nexp")}
    pos = np.zeros((S, T, N)) if keep_pos else None
    for s in range(S):
        cash, shares = 1.0, [0.0] * N
        last_px = [float("nan")] * N
        prev_mtm = 1.0
        for t in range(T):
            for i in range(N):
                if np.isfinite(close[t, i]):
                    last_px[i] = float(close[t, i])
            mtm = cash + sum(shares[i] * last_px[i] for i in range(N) if shares[i] != 0.0)
            gross = mtm - prev_mtm
            alive = mtm > 0
            cost = turn = g = n = 0.0
            for i in range(N):
                px = last_px[i]
                cur = shares[i] * px if shares[i] != 0.0 else 0.0
                if not alive:
                    tgt = 0.0
                elif np.isfinite(close[t, i]):
                    tgt = H[s, t, i] * mtm
                else:
                    tgt = cur  # no bar today: cannot trade, hold what we have
                dw = abs(tgt - cur) / mtm if alive else 0.0
                if dw > 0:
                    unit = lin
                    if impact > 0:
                        unit += impact * sigma[t, i] * np.sqrt(dw * mtm * aum / adv[t, i])
                    cost += dw * unit * mtm
                    cash -= tgt - cur
                    shares[i] = tgt / px
                turn += dw
                w = tgt / mtm if alive else 0.0
                g += abs(w)
                n += w
                if keep_pos:
                    pos[s, t, i] = shares[i] * px if shares[i] != 0.0 else 0.0
            cash -= cost
            equity = cash + sum(shares[i] * last_px[i] for i in range(N) if shares[i] != 0.0)
            if not alive:
                equity, cash, shares = 0.0, 0.0, [0.0] * N
            out["equity"][s, t] = equity
            out["gross_pnl"][s, t] = gross
            out["cost"][s, t] = cost
            out["turnover"][s, t] = turn
            out["gexp"][s, t] = g
            out["nexp"][s, t] = n
            prev_mtm = equity
    return out, pos


# ------------------------------------------------------------------ public API
def run_backtest(weights: np.ndarray, close: np.ndarray, costs: CostModel | None = None,
                 delay: int = 1, engine: str = "numba", sigma: np.ndarray | None = None,
                 adv: np.ndarray | None = None, keep_positions: bool = False,
                 market: tuple[np.ndarray, np.ndarray] | None = None) -> BacktestResult:
    """Simulate target weights.

    weights: (T, N) or (S, T, N). Row t is the target decided with information up to
             the close of day t. It is executed at the close of day t + delay.
    close:   (T, N) closes, NaN where the asset has no bar (not listed, halted, delisted).
    sigma, adv: (T, N) daily return vol and dollar volume, only needed when
             costs.impact_coef > 0. They must themselves be causal (trailing windows).
    market:  optional prepare_market(close) result. The forward fill costs more than the
             numba kernel for a single strategy, so sweeps over one panel should reuse it.
    """
    costs = costs or CostModel()
    if delay < 0:
        raise ValueError("delay must be >= 0 (negative delay trades before the signal exists)")
    W = weights[None] if weights.ndim == 2 else weights
    S, T, N = W.shape
    if close.shape != (T, N):
        raise ValueError(f"close shape {close.shape} != weights (T, N) {(T, N)}")
    H = np.ascontiguousarray(_lagged_targets(np.asarray(W, dtype=np.float64), delay))
    impact = float(costs.impact_coef)
    if impact > 0:
        if sigma is None or adv is None:
            raise ValueError("impact_coef > 0 needs sigma and adv")
        sigma = np.nan_to_num(sigma, nan=0.0)
        adv = np.where(np.isfinite(adv) & (adv > 0), adv, np.inf)
    else:
        sigma = np.zeros((T, N))
        adv = np.ones((T, N))

    if engine == "reference":
        out, pos = _kernel_reference(H, close, costs.linear, impact, costs.aum, sigma, adv,
                                     keep_positions)
    else:
        r, tradeable = market if market is not None else prepare_market(close)
        if engine == "numpy" or (engine == "numba" and not HAVE_NUMBA):
            out, pos = _kernel_numpy(H, r, tradeable, costs.linear, impact, costs.aum, sigma,
                                     adv, keep_positions)
        elif engine == "numba":
            out = {k: np.zeros((S, T)) for k in ("equity", "gross_pnl", "cost", "turnover",
                                                 "gexp", "nexp")}
            pos = np.zeros((S, T, N) if keep_positions else (1, 1, 1))
            _kernel_numba(H, r, tradeable, costs.linear, impact, costs.aum,
                          np.ascontiguousarray(sigma), np.ascontiguousarray(adv), keep_positions,
                          out["equity"], out["gross_pnl"], out["cost"], out["turnover"],
                          out["gexp"], out["nexp"], pos)
            pos = pos if keep_positions else None
        else:
            raise ValueError(f"unknown engine {engine!r}")

    eq = out["equity"]
    prev = np.concatenate([np.ones((S, 1)), eq[:, :-1]], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(prev > 0, eq / prev - 1.0, 0.0)
    res = BacktestResult(eq, ret, out["gross_pnl"], out["cost"], out["turnover"], out["gexp"],
                         out["nexp"], pos)
    return res.squeeze() if weights.ndim == 2 else res
