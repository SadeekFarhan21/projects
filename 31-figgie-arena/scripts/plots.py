"""Plots from results/ (tournament and env benchmark). Run after the experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
COLORS = {"bayes": "#2b6cb0", "passive": "#38a169", "taker": "#dd6b20", "random": "#718096"}


def pnl_plot() -> None:
    s = json.loads((R / "tournament_summary.json").read_text())["matchups"]
    names = ["bayes_vs_random", "bayes_vs_taker", "bayes_vs_passive"]
    labels = ["vs 3 random", "vs 3 takers", "vs 3 passive quoters"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, n in enumerate(names):
        b = next(e for e in s[n]["bots"] if e["bot"] == "bayes")
        ax.barh(i, b["pnl_mean"], color=COLORS["bayes"], height=0.55)
        ax.errorbar(b["pnl_mean"], i, xerr=[[b["pnl_mean"] - b["pnl_lo"]], [b["pnl_hi"] - b["pnl_mean"]]],
                    color="black", capsize=4, lw=1)
        ax.text(b["pnl_hi"] + 4, i, f"{b['pnl_mean']:+.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)), labels)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Bayesian bot mean PnL per game (chips, 95% bootstrap interval)")
    ax.set_title("Bayesian bot against each scripted baseline")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(R / "tournament_pnl.png", dpi=150)


def brier_plot() -> None:
    rows = list(csv.DictReader(open(R / "tournament_brier.csv")))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for bot in ["bayes", "passive", "taker", "random"]:
        pts = [(int(r["tick"]), float(r["brier"])) for r in rows if r["matchup"] == "mixed" and r["bot"] == bot]
        ax.plot(*zip(*pts), marker="o", color=COLORS[bot], label=bot)
    ax.set_xlabel("Tick (of 200)")
    ax.set_ylabel("Brier score of goal suit belief")
    ax.set_title("Goal suit belief quality in the mixed lineup (lower is better)")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(R / "tournament_brier.png", dpi=150)


def env_plot() -> None:
    rows = list(csv.DictReader(open(R / "env_bench.csv")))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for opp, c in [("random", "#718096"), ("scripted", "#38a169"), ("bayes", "#2b6cb0")]:
        pts = [(int(r["num_envs"]), float(r["env_steps_per_sec"])) for r in rows if r["opponents"] == opp]
        ax.plot(*zip(*pts), marker="o", color=c, label=f"{opp} opponents")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Environments per VecEnv (one thread)")
    ax.set_ylabel("Env steps per second")
    ax.set_title("VecEnv throughput through the Python binding")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(R / "env_bench.png", dpi=150)


if __name__ == "__main__":
    pnl_plot()
    brier_plot()
    env_plot()
