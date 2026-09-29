"""Tournament harness: the Bayesian bot against each scripted baseline.

Each matchup plays ``--games`` games as ``games / 4`` deals, each deal replayed
four times with the lineup rotated one seat (duplicate format), so every bot
sees every seat and every hand. Intervals are bootstrap 95 percent intervals
that resample whole deals.

Usage:
    uv run python scripts/tournament.py --games 10000 --threads 2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import figgie_arena as fa
from figgie_arena.stats import bootstrap_ci

MATCHUPS = {
    "bayes_vs_random": ["bayes", "random", "random", "random"],
    "bayes_vs_passive": ["bayes", "passive", "passive", "passive"],
    "bayes_vs_taker": ["bayes", "taker", "taker", "taker"],
    "mixed": ["bayes", "passive", "taker", "random"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = fa.Config()
    n = cfg.n_players
    deals = args.games // n

    pnl_rows, brier_rows, summary = [], [], {"config": {
        "games_per_matchup": deals * n, "deals": deals, "seed": args.seed, "ticks": cfg.ticks,
        "n_players": n, "start_cash": cfg.start_cash, "pot": cfg.pot, "threads": args.threads,
        "checkpoints": args.checkpoints}, "matchups": {}}
    raw = {}
    for name, lineup in MATCHUPS.items():
        t0 = time.perf_counter()
        r = fa.run_games(lineup, deals, args.seed, cfg, args.checkpoints, True, True, args.threads)
        sec = time.perf_counter() - t0
        pnl, brier, deal = r["pnl"], r["brier"], r["deal"]
        assert np.allclose(pnl.sum(axis=1), 0.0), "PnL must be zero sum"
        m = {"seconds": round(sec, 2), "games": int(pnl.shape[0]),
             "invariant_violations": int(r["invariant_violations"]),
             "mean_trades_per_game": float(r["n_trades"].mean()), "bots": []}
        # Aggregate identical bots in a lineup into one entry (per-game mean over copies).
        for bot in dict.fromkeys(lineup):
            cols = [i for i, b in enumerate(lineup) if b == bot]
            per_game = pnl[:, cols].mean(axis=1)
            mean, lo, hi = bootstrap_ci(per_game, deal, seed=args.seed)
            win = float((r["goal_cards_end"][:, cols].mean(axis=1)).mean())
            entry = {"bot": bot, "seats": len(cols), "pnl_mean": mean, "pnl_lo": lo, "pnl_hi": hi,
                     "pnl_sd": float(pnl[:, cols].std()), "mean_goal_cards_end": win}
            for c in range(args.checkpoints):
                b = brier[:, cols, c].mean(axis=1)
                bm, blo, bhi = bootstrap_ci(b, deal, seed=args.seed + c)
                entry[f"brier_cp{c + 1}"] = bm
                brier_rows.append({"matchup": name, "bot": bot, "checkpoint": c + 1,
                                   "tick": int(np.ceil(cfg.ticks * (c + 1) / args.checkpoints)),
                                   "brier": bm, "lo": blo, "hi": bhi})
            m["bots"].append(entry)
            pnl_rows.append({"matchup": name, **{k: v for k, v in entry.items() if not k.startswith("brier")}})
        summary["matchups"][name] = m
        raw[name] = pnl
        print(f"{name:18s} {sec:6.1f}s  " + "  ".join(
            f"{e['bot']}={e['pnl_mean']:+.2f} [{e['pnl_lo']:+.2f},{e['pnl_hi']:+.2f}] brier={e[f'brier_cp{args.checkpoints}']:.3f}"
            for e in m["bots"]))

    with open(args.out / "tournament_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for fname, rows in (("tournament_pnl.csv", pnl_rows), ("tournament_brier.csv", brier_rows)):
        with open(args.out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    np.savez_compressed(args.out / "tournament_pnl_per_game.npz", **raw)


if __name__ == "__main__":
    main()
