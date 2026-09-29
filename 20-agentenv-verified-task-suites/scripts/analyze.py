"""Turn model-run traces into the headline tables and plots.

uv run python scripts/analyze.py --run results/runs/qwen1.5b_t07_n4_seed0.jsonl \
    [--repro results/runs/qwen1.5b_t07_n4_seed0_repeat.jsonl]

Writes, next to results/:
  results/model_summary.csv          pass@k with bootstrap intervals by family and difficulty
  results/failure_taxonomy.csv       failure labels by family
  results/reproducibility.json       same model and seed, two runs compared (when --repro)
  results/pass_at_k.png, results/failure_taxonomy.png
"""
import argparse
import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agentenv import PROJECT_ROOT  # noqa: E402
from agentenv.metrics import (bootstrap_ci, episodes_frame, failure_table, load_traces,  # noqa: E402
                              summarize)

RES = PROJECT_ROOT / "results"
COLORS = {"sql": "#3b6ea5", "bugfix": "#c0633a"}


def plot_pass_at_k(summ, ks, path):
    fams = [f for f in ("sql", "bugfix") if f in set(summ.family)]
    diffs = ["easy", "medium", "hard", "all"]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / len(ks)
    for ax, fam in zip(axes, fams):
        for j, k in enumerate(ks):
            vals, lo, hi = [], [], []
            for d in diffs:
                r = summ[(summ.family == fam) & (summ.difficulty == d)].iloc[0]
                vals.append(r[f"pass@{k}"])
                lo.append(r[f"pass@{k}"] - r[f"pass@{k}_lo"])
                hi.append(r[f"pass@{k}_hi"] - r[f"pass@{k}"])
            x = np.arange(len(diffs)) + (j - (len(ks) - 1) / 2) * width
            ax.bar(x, vals, width * 0.92, yerr=[lo, hi], capsize=2, label=f"pass@{k}",
                   color=COLORS[fam], alpha=0.45 + 0.55 * (j + 1) / len(ks), error_kw={"lw": 0.8})
        ax.set_xticks(range(len(diffs)), diffs)
        ax.set_title(f"{fam} family")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Pass rate (95% bootstrap CI)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Qwen2.5 1.5B Instruct 4-bit, pass@k by difficulty", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_failures(ft, path):
    fams = [f for f in ("sql", "bugfix") if f in set(ft.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.4 * len(fams), 3.6))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        g = ft[ft.family == fam].sort_values("count")
        labels = [s.replace("_", " ") for s in g.failure]
        ax.barh(labels, g["count"], color=COLORS[fam])
        for y, c in enumerate(g["count"]):
            ax.text(c, y, f" {c}", va="center", fontsize=8)
        ax.set_title(f"{fam} failed episodes by cause", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Episodes")
        ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reproducibility(df_a, df_b):
    key = ["task_id", "sample"]
    m = df_a[key + ["family", "passed", "total_tokens"]].merge(
        df_b[key + ["passed", "total_tokens"]], on=key, suffixes=("_a", "_b"))
    out = {"episodes_compared": int(len(m)),
           "identical_outcome_rate": float((m.passed_a == m.passed_b).mean()),
           "identical_token_count_rate": float((m.total_tokens_a == m.total_tokens_b).mean())}
    for fam, g in [("all", m), *m.groupby("family")]:
        per_a = g.groupby("task_id").passed_a.mean().to_numpy(dtype=float)
        per_b = g.groupby("task_id").passed_b.mean().to_numpy(dtype=float)
        ci_a, ci_b = bootstrap_ci(per_a), bootstrap_ci(per_b)
        out[fam] = {"pass@1_a": float(per_a.mean()), "pass@1_b": float(per_b.mean()),
                    "ci_a": ci_a, "ci_b": ci_b,
                    "b_inside_ci_a": bool(ci_a[0] <= per_b.mean() <= ci_a[1]),
                    "a_inside_ci_b": bool(ci_b[0] <= per_a.mean() <= ci_b[1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repro", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    traces = load_traces([args.run])
    df = episodes_frame(traces)
    n = int(df.groupby("task_id").size().min())
    ks = [k for k in (1, 2, 4, 8) if k <= n]
    summ = summarize(df, ks=ks)
    meta = json.loads(Path(args.run.replace(".jsonl", ".meta.json")).read_text())
    summ["episodes_per_hour_run"] = meta["episodes_per_hour"]
    tag = args.tag
    summ.to_csv(RES / f"model_summary{tag}.csv", index=False)
    ft = failure_table(df)
    ft.to_csv(RES / f"failure_taxonomy{tag}.csv", index=False)
    plot_pass_at_k(summ, ks, RES / f"pass_at_k{tag}.png")
    plot_failures(ft, RES / f"failure_taxonomy{tag}.png")

    # a DuckDB view over the raw JSONL, for per-step stats without loading into Python
    con = duckdb.connect()
    step_stats = con.execute(f"""
        SELECT family, avg(n_steps) AS mean_steps, avg(completion_tokens) AS mean_completion_tokens,
               avg(prompt_tokens) AS mean_prompt_tokens, avg(model_latency_s) AS mean_model_s,
               avg(wall_s) AS mean_wall_s, avg(CASE WHEN submitted THEN 1 ELSE 0 END) AS submit_rate
        FROM read_json_auto('{args.run}', maximum_object_size=67108864) GROUP BY family ORDER BY family
    """).df()
    step_stats.to_csv(RES / f"episode_stats{tag}.csv", index=False)

    headline = {"run": args.run, "meta": meta, "ks": ks,
                "summary": summ.to_dict(orient="records"),
                "episode_stats": step_stats.to_dict(orient="records")}
    if args.repro:
        headline["reproducibility"] = reproducibility(df, episodes_frame(load_traces([args.repro])))
        (RES / f"reproducibility{tag}.json").write_text(json.dumps(headline["reproducibility"], indent=2))
    (RES / f"headline{tag}.json").write_text(json.dumps(headline, indent=2, default=str))
    cols = ["family", "difficulty", "n_tasks", "episodes"] + [c for c in summ.columns if c.startswith("pass@")] \
        + ["tokens_per_solved", "wall_s_per_solved"]
    print(summ[cols].round(3).to_string(index=False))
    print(ft.to_string(index=False))
    print(step_stats.round(2).to_string(index=False))
    if args.repro:
        print(json.dumps(headline["reproducibility"], indent=1))


if __name__ == "__main__":
    main()
