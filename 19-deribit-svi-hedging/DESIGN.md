# Design

### Data flow

```
 tardis.dev options_chain CSV.gz (one day, ~100 GB uncompressed, 236M rows)
        |  curl | gunzip
        v
 tools/minute_snap (C++20)      keep BTC-/ETH- rows; per symbol, last row before
        |                       each minute boundary; prefix snap_ts
        v
 data/raw/deribit_options_chain_YYYYMMDD_1m.csv.gz   (2.3M rows, 147 MB)
        |
        v
 chain.load_minute_file         polars; parse coin/expiry; assert 08:00 UTC expiries
        |
        +--------------------------------------------+
        | snapshot view (per hour)                   | panel view (per minute, one expiry)
        v                                            v
 chain.snapshot(t)   as-of, max age 10 min    hedging.build_panel   as-of join on a minute grid,
 chain.clean         drop one-sided, zero-bid,                      spread spike filter,
                     crossed, locked; mids, T                       USD mids, own IVs,
 chain.fit_forwards  weighted robust LS of                          ITM -> OTM twin IV
                     (C - P) on K per expiry        |
 chain.attach_iv     iv.implied_vol on bid/mid/ask  v
        |                                    hedging.fit_surface_path  SVI each minute, warm start
        v                                    hedging.hedge_ratios      own IV / sticky strike /
 surface.fit_snapshot                                                  sticky moneyness deltas
   svi.fit_svi per expiry                    hedging.simulate          short 1 option, hedge every
   svi.butterfly_ok (g on 2001 pts)                                    N min, per-minute attribution
   svi.calendar_ok (adjacent expiries)       hedging.bootstrap_std     (cluster) bootstrap of std
   svi.fit_ssvi (all expiries at once)              |
        |                                           v
        v                                    experiments/04_hedging.py
 experiments/02, 03
                        black76 (numba kernels) and iv (numba kernel + Brent) underneath everything
```

### Key data structures

- Day frame (polars): one row per symbol per minute with the tardis columns (bid/ask price and IV, mark price and IV, underlying, Deribit greeks) plus `snap_ts`, `coin`, `is_call`.
- Clean chain (polars): the snapshot rows that survive cleaning, with `T` (exact, from `expiration - t` in microseconds over 365 days), coin and USD mids and spreads, `F` (parity forward or Deribit underlying), `k = ln(K/F)`, `iv_bid/iv_mid/iv_ask` with status codes, and Black-76 vega and delta at mid IV.
- `ForwardFit` rows: per (coin, expiry) `F`, intercept, number of pairs, residual std, `accepted`.
- `SVIParams(a, b, rho, m, s)` and `SSVIParams(rho, eta, gamma, theta[])`. The fit works in `(amin, b, rho, m, s)` with `a = amin - b s sqrt(1 - rho^2)`, so the minimum total variance is a simple box bound.
- `Panel`: numpy arrays for one (coin, expiry): minute grid `t_us (n_t)`, hedge price `F (n_t)`, `K, is_call (n_o)`, and `mid, half_spread, iv (n_t x n_o)`, NaN where no clean quote. All hedging code is array code on these.
- `HedgeResult`: per option totals of P&L and of each attribution term.

### Invariants

- Every Deribit expiry is 08:00 UTC; the loader raises if any row says otherwise. T is never computed from a date alone.
- Prices are compared to Black-76 in one currency. Coin premiums are multiplied by the forward they are priced against (`coin_to_usd`), never by a different underlying.
- The IV solver never returns a number for a price outside `[intrinsic, upper]`: it returns NaN with a status (`BELOW_INTRINSIC`, `ABOVE_UPPER`, `AT_INTRINSIC`, `BAD_INPUT`), and statuses flow into the chain as columns.
- The solver always inverts the out-of-the-money price (ITM prices are converted by parity first), so it never solves for a tiny time value buried under a large intrinsic value.
- A parity forward is used only if it comes from at least 5 strike pairs within 25% log moneyness and its intercept is within 0.01 of 1 (under Deribit's coin convention `C - P = 1 - K/F` in coin). Otherwise the chain falls back to Deribit's underlying for that expiry.
- The hedge instrument and the option are priced off the same `F` series, and at expiry both settle to the same settlement value.
- The attribution adds up exactly: `total = theta + gamma + vega + hedge_err + residual + tail` per position (checked in `tests/test_hedging.py`).

### Trade-offs

Chosen:

- A C++ stream reducer before any Python. The day file is 100 GB uncompressed; Python or awk could not keep up with the download, and keeping only the last row per symbol per minute cuts it 100x while keeping minute resolution for hedging.
- numba for Black-76 and the IV solver instead of vectorized numpy. The solver has data-dependent branches and iteration counts per option; a per-option loop compiled by numba with `prange` is simpler and faster than masked numpy iterations, and hits 2M IVs/s on one core.
- Newton on `log(price)` in total vol `v = sigma sqrt(T)` with a bracket kept at every step. The log makes far OTM options (price near 0) well conditioned; the bracket makes the method globally safe. Brent on the original price is the fallback for the 0.025% that do not converge in 40 steps.
- Corrado-Miller as the initial guess, with the inflection point `sqrt(2 |ln K/F|)` when its discriminant is negative. Mean 6.2 Newton steps.
- Raw SVI fit on total variance with weights vega over spread, capped at 10x the slice median. Weighting by vega converts total variance error back toward price error; dividing by spread trusts tight quotes more; the cap stops one ATM quote dominating.
- Multi-start (6 starts) least squares with bounds for SVI; warm start and 2 starts when fitting minute after minute.
- SSVI with the power-law phi parameterized so that the static no-arbitrage conditions (`theta` increasing, `eta (1 + |rho|) <= 2`, `0 < gamma <= 1/2`) hold by construction. It is arbitrage free every time and gives a much worse fit, which is the comparison the spec asked for.
- Hedging done on out-of-the-money options only, each ITM option taking its OTM twin's IV. An ITM option is its twin plus a forward; the forward is hedged away, so ITM positions add only their noisier quotes.

Rejected:

- Jaeckel's "Let's be rational" solver. It is the right answer for machine precision in two iterations, but it is long and subtle to port; the safeguarded Newton already reaches 1e-15 price error and 2M/s, far beyond the target.
- Using Deribit's `underlying_price` as the forward everywhere. It is a good default (median gap to the parity forward 1.7 bp) but the spec asked for a parity regression, and the regression exposes when the chain disagrees with itself.
- Fitting SVI to IVs directly. Total variance is the natural space for the no-arbitrage conditions and makes the calendar check a direct comparison.
- Repairing butterfly violations inside the fit (penalizing negative `g`). The v0 goal was to measure how often an honest fit violates it; a constrained refit is a later step.
- Parity-implied marking of ITM options. It would fix ITM mid noise but hides what the market actually quoted; selecting OTM options achieves the same without inventing prices.
- Inverse (coin-margined) P&L accounting. Correct for Deribit but it mixes a currency effect into the hedging comparison; USD linear P&L isolates the delta hedging question.
