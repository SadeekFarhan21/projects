"""Post-hoc figure from saved results only (does not re-run the holdout):
dev Sharpe vs holdout Sharpe for all 18 trials."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from marketpred.data import PROJECT_ROOT

R = PROJECT_ROOT / "results"
dev = pd.read_csv(R / "dev" / "summary.csv").set_index("trial")["sharpe_net"]
hold = pd.read_csv(R / "holdout" / "all_trials_posthoc.csv").set_index("trial")["sharpe_net"]
df = pd.DataFrame({"dev": dev, "holdout": hold})
fig, ax = plt.subplots(figsize=(7, 5.5))
colors = {"rev": "tab:blue", "mom": "tab:green", "rid": "tab:purple", "gbm": "tab:red"}
labels = {"rev": "reversal factor", "mom": "momentum factor", "rid": "ridge", "gbm": "gradient boosting"}
for k, c in colors.items():
    d = df[df.index.str.startswith(k)]
    ax.scatter(d["dev"], d["holdout"], color=c, label=labels[k], s=30)
sel = "mom_20d|quintile"
ax.annotate("dev-selected", df.loc[sel], textcoords="offset points", xytext=(8, -12), fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("development net Sharpe (15 bps)")
ax.set_ylabel("holdout net Sharpe (15 bps)")
ax.set_title("18 trials: development vs holdout (post-hoc view)")
ax.legend(frameon=False, loc="upper left")
fig.text(0.01, 0.01, "Source: Binance spot daily klines (data.binance.vision), own calculations.", fontsize=7, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(R / "holdout" / "dev_vs_holdout_sharpe.png", dpi=150)
df.to_csv(R / "holdout" / "dev_vs_holdout_sharpe.csv")
print(df.round(3).sort_values("dev", ascending=False).to_string())
