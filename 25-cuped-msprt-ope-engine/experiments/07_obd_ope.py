"""Open Bandit Dataset: estimate the Bernoulli TS policy value from uniform-random logs.

Evaluation policy: ZOZOTOWN's production Bernoulli Thompson Sampling with its Beta prior,
turned into a slate distribution P(item a at slot k) by Monte Carlo. Logged data: the
uniform-random policy (pscore = 1/80). Ground truth: the on-policy click rate in the BTS logs.
Every estimator is also run through obp on the identical arrays and the gap is recorded.

    uv run python scripts/fetch_obd.py
    uv run python experiments/07_obd_ope.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from _common import PALETTE, RESULTS, ROOT, plt, save, style, write_json
from expope.ope import (
    RewardModelConfig,
    bootstrap_ci,
    estimate_all,
    estimator_terms,
    fit_predict_q_hat,
    row_terms,
)
from expope.ope.obd import bts_action_dist, load_bts_prior, load_obd


def item_context(data_dir: Path) -> np.ndarray:
    ic = pl.read_csv(data_dir / "random" / "all" / "item_context.csv").sort("item_id")
    cols = [ic["item_feature_0"].to_numpy()]
    for c in ("item_feature_1", "item_feature_2", "item_feature_3"):
        levels = {v: i for i, v in enumerate(sorted(ic[c].unique().to_list()))}
        cols.append(np.array([levels[v] for v in ic[c].to_list()], dtype=float))
    return np.stack(cols, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "obd")
    ap.add_argument("--n-sim", type=int, default=100_000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rnd = load_obd(args.data_dir, "random")
    bts = load_obd(args.data_dir, "bts")
    alpha, beta = load_bts_prior(args.data_dir / "prior_bts.yaml")
    t0 = time.perf_counter()
    dist = bts_action_dist(alpha, beta, n_sim=args.n_sim, seed=args.seed)
    t_dist = time.perf_counter() - t0
    action_dist = np.broadcast_to(dist, (rnd.n, rnd.n_actions, rnd.len_list)).copy()

    # On-policy ground truth and its sampling noise.
    truth = float(bts.reward.mean())
    truth_ci = bootstrap_ci(bts.reward, None, n_boot=args.n_boot, seed=1)

    t0 = time.perf_counter()
    q_hat = fit_predict_q_hat(
        rnd.context, rnd.action, rnd.reward, rnd.n_actions, rnd.len_list, rnd.position,
        action_context=item_context(args.data_dir),
        cfg=RewardModelConfig(n_estimators=100, learning_rate=0.05, num_leaves=7, min_child_samples=100, n_jobs=2, seed=0),
    )
    t_model = time.perf_counter() - t0

    t = row_terms(rnd.reward, rnd.action, rnd.pscore, action_dist, rnd.position, q_hat)
    ests = estimate_all(t, tau=args.tau)
    rows = []
    for name, v in ests.items():
        num, den = estimator_terms(name, t, args.tau)
        ci = bootstrap_ci(num, den, n_boot=args.n_boot, seed=7)
        rows.append(dict(estimator=name, estimate=v, lower=ci["lower"], upper=ci["upper"], boot_se=ci["se"],
                         on_policy=truth, rel_error=abs(v - truth) / truth,
                         within_truth_ci=truth_ci["lower"] <= v <= truth_ci["upper"]))
    tau_grid = {str(tau): estimate_all(t, tau=tau)["switch_dr"] for tau in (1.0, 2.0, 5.0, 10.0, 20.0)}
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "obd_ope.csv")
    print(df)

    # Cross-check against obp on identical inputs.
    xcheck = {}
    try:
        from obp.ope import (
            DirectMethod,
            DoublyRobust,
            InverseProbabilityWeighting,
            SelfNormalizedInverseProbabilityWeighting,
            SwitchDoublyRobust,
        )
        from obp.policy import BernoulliTS

        kw = dict(reward=rnd.reward, action=rnd.action, pscore=rnd.pscore, action_dist=action_dist, position=rnd.position)
        ref = {
            "ips": InverseProbabilityWeighting().estimate_policy_value(**kw),
            "snips": SelfNormalizedInverseProbabilityWeighting().estimate_policy_value(**kw),
            "dm": DirectMethod().estimate_policy_value(action_dist=action_dist, estimated_rewards_by_reg_model=q_hat, position=rnd.position),
            "dr": DoublyRobust().estimate_policy_value(**kw, estimated_rewards_by_reg_model=q_hat),
            "switch_dr": SwitchDoublyRobust(tau=args.tau).estimate_policy_value(**kw, estimated_rewards_by_reg_model=q_hat),
        }
        xcheck["estimators"] = {k: dict(ours=ests[k], obp=float(ref[k]), abs_diff=abs(ests[k] - float(ref[k]))) for k in ref}
        xcheck["max_abs_diff"] = max(v["abs_diff"] for v in xcheck["estimators"].values())
        # BTS slate distribution: both are Monte Carlo, so compare within MC noise, not to 1e-9.
        t0 = time.perf_counter()
        obp_dist = BernoulliTS(n_actions=80, len_list=3, is_zozotown_prior=True, campaign="all", random_state=args.seed
                               ).compute_batch_action_dist(n_rounds=1, n_sim=args.n_sim)[0]
        xcheck["bts_action_dist"] = dict(
            max_abs_diff=float(np.abs(obp_dist - dist).max()),
            total_variation_per_slot=[float(0.5 * np.abs(obp_dist[:, k] - dist[:, k]).sum()) for k in range(3)],
            obp_seconds=time.perf_counter() - t0,
            ours_seconds=t_dist,
        )
        ests_obp_dist = estimate_all(row_terms(rnd.reward, rnd.action, rnd.pscore,
                                               np.broadcast_to(obp_dist, action_dist.shape).copy(), rnd.position, q_hat), tau=args.tau)
        xcheck["estimates_with_obp_bts_dist"] = ests_obp_dist
        print("obp max abs diff", xcheck["max_abs_diff"])
    except ImportError:
        xcheck["skipped"] = "obp not installed (uv sync --group crosscheck)"

    w = t.w
    write_json(
        "obd_ope_summary.json",
        dict(
            config={k: str(v) for k, v in vars(args).items()},
            n_random=rnd.n, n_bts=bts.n, clicks_random=int(rnd.reward.sum()), clicks_bts=int(bts.reward.sum()),
            random_policy_ctr=float(rnd.reward.mean()),
            on_policy_bts_ctr=truth, on_policy_ci=truth_ci,
            weights=dict(mean=float(w.mean()), max=float(w.max()), share_above_tau=float((w > args.tau).mean())),
            reward_model_seconds=t_model,
            q_hat_mean=float(q_hat.mean()),
            estimates=rows, switch_dr_tau_grid=tau_grid, obp_crosscheck=xcheck,
        ),
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(df.height)
    ax.axhspan(truth_ci["lower"], truth_ci["upper"], color=PALETTE[2], alpha=0.15, label="On policy BTS click rate, 95 percent CI")
    ax.axhline(truth, color=PALETTE[2], lw=1.5)
    ax.axhline(rnd.reward.mean(), color="#777", lw=1, ls=":", label="Random policy click rate")
    ax.errorbar(x, df["estimate"], yerr=[df["estimate"] - df["lower"], df["upper"] - df["estimate"]], fmt="o", color=PALETTE[0], capsize=4)
    ax.set_xticks(x, [e.replace("_", " ") for e in df["estimator"]])
    style(ax, "Open Bandit Dataset, BTS value from random logs", "", "Click rate")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    save(fig, "obd_ope.png")


if __name__ == "__main__":
    main()
