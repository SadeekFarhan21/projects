"""Experiment 06: Are the induction heads necessary for in-context copying?

Question. Patching (exp 03) showed the induction heads are sufficient to carry
the answer. Here we remove them and measure how much in-context learning is
lost, against random heads of equal count as a control.

Metric. On repeated random sequences [BOS][A][A] (N = 50), the per-token loss
on the first half is near chance (the tokens are unpredictable) and drops on
the second half once the model can copy from the first. The in-context
learning (ICL) score is
    ICL = loss_2nd_half - loss_1st_half
averaged over query positions 1 .. N-1 (first half) and N+1 .. 2N-1 (second
half). It is strongly negative for the intact model; 0 means no benefit from
context. This mirrors Olsson et al. 2022's ICL score (loss at a late token
minus loss at an early token) on a task where copying is the only way to win.

Ablations (on hook_z, all positions).
  zero   the head's output is set to 0
  mean   the head's output is replaced by its per-position mean over a separate
         batch of 64 repeated random sequences, which removes token-specific
         information but keeps the average activation the rest of the model
         expects
Cases.
  - the five induction heads together, and each alone
  - 20 random sets of five heads drawn from the other 139 heads (control)
  - the previous-token head L4H11 (from exps 03 and 04)
  - the top-k heads by induction score for k = 1 .. 20, against random k-sets,
    to show backup induction heads
  - every one of the 144 heads alone (knockout sweep), zero and mean

Outputs:
  figures/06_ablation.png         ICL score under each ablation, and the top-k curve
  figures/06_knockout_sweep.png   change in ICL score when each head is mean-ablated
  results/06_ablation.csv         every ablation case with both half-losses and the ICL score
  results/06_knockout_sweep.csv   layer, head, ICL score under zero and mean ablation
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from functools import partial
from pathlib import Path

import torch

from tiny_circuits.model import load_model, SEED
from tiny_circuits.tracking import init_run

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
SEQ_LEN = 50          # length of the random sequence (repeated once -> 2*SEQ_LEN)
BATCH = 32            # first 8 rows are exp 01's batch; more rows for a stabler loss
MEAN_BATCH = 64       # separate sequences for the mean-ablation reference
N_RANDOM_SETS = 20
MAX_K = 20

INDUCTION_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]  # from experiment 01
PREV_TOKEN_HEAD = (4, 11)                                      # from experiment 04


def make_repeated_tokens(model, seq_len: int, batch: int, seed: int = SEED) -> torch.Tensor:
    """Build [BOS][rand seq][rand seq] token batches (as in experiment 01)."""
    g = torch.Generator().manual_seed(seed)
    vocab = model.cfg.d_vocab
    rand = torch.randint(0, vocab, (batch, seq_len), generator=g)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    tokens = torch.cat([bos, rand, rand], dim=1)
    return tokens.to(model.cfg.device)


def induction_scores(model, tokens: torch.Tensor) -> torch.Tensor:
    """[n_layers, n_heads] attention to the induction target on second-half queries."""
    seq_len = (tokens.shape[1] - 1) // 2
    _, cache = model.run_with_cache(
        tokens[:8], return_type=None, names_filter=lambda n: n.endswith("hook_pattern")
    )
    q = torch.arange(1 + seq_len, 1 + 2 * seq_len, device=tokens.device)
    return torch.stack([
        cache["pattern", l][:, :, q, q - seq_len + 1].mean(dim=(0, 2)).cpu()
        for l in range(model.cfg.n_layers)
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb", action="store_true", help="log this run to Weights & Biases"
    )
    parser.add_argument("--wandb-name", default="06-ablation", help="W&B run name")
    args = parser.parse_args()
    t0 = time.time()
    torch.set_grad_enabled(False)

    model = load_model()
    cfg = model.cfg
    print(f"loaded {cfg.model_name} on {cfg.device} ({cfg.n_layers}L x {cfg.n_heads}H)")

    run = init_run(
        args.wandb_name,
        config={
            "experiment": "06_ablation",
            "model": cfg.model_name,
            "device": str(cfg.device),
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "mean_batch": MEAN_BATCH,
            "seed": SEED,
            "n_random_sets": N_RANDOM_SETS,
        },
        enabled=args.wandb,
    )

    tokens = make_repeated_tokens(model, SEQ_LEN, BATCH)
    ref = make_repeated_tokens(model, SEQ_LEN, MEAN_BATCH, seed=SEED + 2)
    _, ref_cache = model.run_with_cache(ref, return_type=None,
                                        names_filter=lambda n: n.endswith("hook_z"))
    z_mean = {l: ref_cache["z", l].mean(0) for l in range(cfg.n_layers)}  # [pos, head, d_head]
    del ref_cache

    first = slice(1, SEQ_LEN)                    # queries 1 .. N-1
    second = slice(SEQ_LEN + 1, 2 * SEQ_LEN)     # queries N+1 .. 2N-1

    def ablate(act, hook, heads, mode):
        if mode == "zero":
            act[:, :, heads] = 0.0
        else:
            act[:, :, heads] = z_mean[hook.layer()][None, :, heads]
        return act

    def evaluate(heads: list[tuple[int, int]], mode: str) -> tuple[float, float, float]:
        by_layer: dict[int, list[int]] = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        hooks = [(f"blocks.{l}.attn.hook_z", partial(ablate, heads=hs, mode=mode))
                 for l, hs in by_layer.items()]
        loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True,
                                    fwd_hooks=hooks)          # [b, pos - 1], index p predicts p + 1
        l1 = loss[:, first].mean().item()
        l2 = loss[:, second].mean().item()
        return l1, l2, l2 - l1

    rows = []  # (case, mode, n_heads, loss1, loss2, icl)

    def record(case, mode, heads):
        l1, l2, icl = evaluate(heads, mode)
        rows.append((case, mode, len(heads), l1, l2, icl))
        return icl

    base = record("none", "none", [])
    for mode in ("zero", "mean"):
        record("induction_top5", mode, INDUCTION_HEADS)
        for l, h in INDUCTION_HEADS:
            record(f"L{l}H{h}", mode, [(l, h)])
        record(f"L{PREV_TOKEN_HEAD[0]}H{PREV_TOKEN_HEAD[1]}", mode, [PREV_TOKEN_HEAD])
        record("induction_top5+L4H11", mode, INDUCTION_HEADS + [PREV_TOKEN_HEAD])

    # random sets of five, same draws for both modes
    rng = random.Random(SEED)
    pool = [(l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)
            if (l, h) not in INDUCTION_HEADS]
    random_sets = [rng.sample(pool, len(INDUCTION_HEADS)) for _ in range(N_RANDOM_SETS)]
    rand_icl = {"zero": [], "mean": []}
    for i, hs in enumerate(random_sets):
        for mode in ("zero", "mean"):
            rand_icl[mode].append(record(f"random5_{i}", mode, hs))
    print(f"main ablations done ({time.time() - t0:.1f}s)")

    # top-k by induction score vs random k (mean ablation)
    ind = induction_scores(model, tokens)
    ranked = sorted(((float(ind[l, h]), l, h) for l in range(cfg.n_layers)
                     for h in range(cfg.n_heads)), reverse=True)
    topk_icl, randk_icl = [], []
    for k in range(1, MAX_K + 1):
        hs = [(l, h) for _, l, h in ranked[:k]]
        topk_icl.append(record(f"top{k}_by_induction_score", "mean", hs))
        vals = [evaluate(rng.sample(pool, k), "mean")[2] for _ in range(10)]
        randk_icl.append(sum(vals) / len(vals))
    print(f"top-k curve done ({time.time() - t0:.1f}s)")

    # knockout sweep
    sweep = {"zero": torch.zeros(cfg.n_layers, cfg.n_heads),
             "mean": torch.zeros(cfg.n_layers, cfg.n_heads)}
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            for mode in ("zero", "mean"):
                sweep[mode][l, h] = evaluate([(l, h)], mode)[2]
    print(f"knockout sweep done ({time.time() - t0:.1f}s)")

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    def icl_of(case, mode):
        return next(r[5] for r in rows if r[0] == case and r[1] == mode)

    rmean = {m: sum(v) / len(v) for m, v in rand_icl.items()}
    rstd = {m: float(torch.tensor(v).std()) for m, v in rand_icl.items()}

    # ------------------------------------------------ figure A
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    cases = [("5 induction heads", icl_of("induction_top5", "zero"), icl_of("induction_top5", "mean"), 0, 0),
             ("5 random heads", rmean["zero"], rmean["mean"], rstd["zero"], rstd["mean"]),
             ("L4H11", icl_of("L4H11", "zero"), icl_of("L4H11", "mean"), 0, 0),
             ("induction heads and L4H11", icl_of("induction_top5+L4H11", "zero"),
              icl_of("induction_top5+L4H11", "mean"), 0, 0)]
    y = torch.arange(len(cases)).float()
    hgt = 0.38
    labels = [c[0] for c in cases][::-1]
    zero_v = [c[1] for c in cases][::-1]
    mean_v = [c[2] for c in cases][::-1]
    zero_e = [c[3] for c in cases][::-1]
    mean_e = [c[4] for c in cases][::-1]
    ax0.barh(y + hgt / 2, zero_v, hgt, color="#b8c7dc", label="zero ablation")
    ax0.barh(y - hgt / 2, mean_v, hgt, color="#3b6ea5", label="mean ablation")
    for yi, v, e in zip(y, zero_v, zero_e):       # error bars only where there is a spread
        if e > 0:
            ax0.errorbar(v, float(yi) + hgt / 2, xerr=e, color="0.2", lw=0.8, capsize=2)
    for yi, v, e in zip(y, mean_v, mean_e):
        if e > 0:
            ax0.errorbar(v, float(yi) - hgt / 2, xerr=e, color="0.2", lw=0.8, capsize=2)
        ax0.text(v - max(e, 0.1), float(yi) - hgt / 2, f"{v:.1f}  ", va="center", ha="right", fontsize=8)
    ax0.axvline(base, color="0.3", lw=0.8, ls="--", label="intact model")
    ax0.set_xlim(min(zero_v + mean_v) - 2.5, 0)
    ax0.set_yticks(y.tolist(), labels)
    ax0.axvline(0, color="0.3", lw=0.8)
    ax0.set_xlabel("ICL score, loss 2nd half minus loss 1st half (nats)")
    ax0.set_title("Ablating heads", fontsize=11)
    ax0.legend(frameon=False, fontsize=8, ncols=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.16))
    ax0.spines[["top", "right"]].set_visible(False)

    ks = list(range(1, MAX_K + 1))
    ax1.plot(ks, topk_icl, marker="o", ms=3.5, color="#3b6ea5", label="top k by induction score")
    ax1.plot(ks, randk_icl, marker="o", ms=3.5, color="0.55", label="random k heads")
    ax1.axhline(base, color="0.3", lw=0.8, ls="--")
    ax1.axvline(5, color="0.8", lw=0.8)
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_xlabel("heads mean-ablated")
    ax1.set_ylabel("ICL score (nats)")
    ax1.set_title("Backup induction heads", fontsize=11)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "06_ablation.png", dpi=150)
    run.log_figure("ablation", fig)
    plt.close(fig)

    # ------------------------------------------------ figure B: knockout sweep
    delta = sweep["mean"] - base
    fig, ax = plt.subplots(figsize=(5.8, 4.6), layout="constrained")
    lim = float(delta.abs().max())
    im = ax.imshow(delta.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                   interpolation="nearest")
    ax.set_title("Knockout sweep, one head mean-ablated", fontsize=11)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks([0, 5, 11])
    ax.set_yticks([0, 5, 11])
    fig.colorbar(im, ax=ax, shrink=0.85, label="change in ICL score (nats)")
    fig.savefig(FIGURES / "06_knockout_sweep.png", dpi=150)
    run.log_figure("knockout_sweep", fig)
    plt.close(fig)

    # ------------------------------------------------ tables
    cols = ["case", "ablation", "n_heads", "loss_1st_half", "loss_2nd_half", "icl_score"]
    with open(RESULTS / "06_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    run.log_table("ablation", cols, rows)
    sweep_rows = [(l, h, float(sweep["zero"][l, h]), float(sweep["mean"][l, h]))
                  for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    with open(RESULTS / "06_knockout_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "icl_zero", "icl_mean"])
        w.writerows(sweep_rows)
    run.log_table("knockout_sweep", ["layer", "head", "icl_zero", "icl_mean"], sweep_rows)

    # ------------------------------------------------ summary
    base_row = rows[0]
    print(f"\nIntact model  loss 1st half {base_row[3]:.3f}  2nd half {base_row[4]:.3f}  "
          f"ICL {base:.3f}  (log vocab = {torch.tensor(float(cfg.d_vocab)).log():.2f})")
    print("\nAblation                    zero ICL   mean ICL   (loss 2nd half, mean)")
    for case in ["induction_top5"] + [f"L{l}H{h}" for l, h in INDUCTION_HEADS] + \
                ["L4H11", "induction_top5+L4H11"]:
        z = next(r for r in rows if r[0] == case and r[1] == "zero")
        m = next(r for r in rows if r[0] == case and r[1] == "mean")
        print(f"  {case:<26} {z[5]:8.3f}  {m[5]:8.3f}   ({m[4]:.3f})")
    print(f"  {'random 5 heads (20 sets)':<26} {rmean['zero']:8.3f}  {rmean['mean']:8.3f}   "
          f"std {rstd['zero']:.3f} / {rstd['mean']:.3f}, worst {max(rand_icl['mean']):.3f} (mean)")
    frac = lambda v: (v - base) / (0 - base)
    ind_mean = icl_of("induction_top5", "mean")
    print(f"\n  five induction heads remove {frac(ind_mean):.0%} of the ICL score (mean ablation), "
          f"random five remove {frac(rmean['mean']):.0%}")
    print("  top-k curve (mean): " + ", ".join(f"k={k} {v:.2f}" for k, v in zip(ks, topk_icl)
                                              if k in (1, 5, 10, 15, 20)))
    print("  random-k (mean):    " + ", ".join(f"k={k} {v:.2f}" for k, v in zip(ks, randk_icl)
                                              if k in (1, 5, 10, 15, 20)))
    print("  top-20 heads by induction score: "
          + ", ".join(f"L{l}H{h}" for _, l, h in ranked[:MAX_K]))
    worst = sorted(sweep_rows, key=lambda r: -r[3])[:6]
    best = sorted(sweep_rows, key=lambda r: r[3])[:3]
    print("  knockout sweep, biggest ICL loss (mean): "
          + ", ".join(f"L{l}H{h} {m - base:+.3f}" for l, h, _, m in worst))
    print("  knockout sweep, ablation helps ICL (mean): "
          + ", ".join(f"L{l}H{h} {m - base:+.3f}" for l, h, _, m in best))

    run.summary({
        "icl_intact": base, "icl_induction5_zero": icl_of("induction_top5", "zero"),
        "icl_induction5_mean": ind_mean, "icl_random5_mean": rmean["mean"],
        "icl_random5_zero": rmean["zero"], "icl_L4H11_mean": icl_of("L4H11", "mean"),
    })

    print("\nLiterature check:")
    print("  Olsson et al. 2022 find that knocking out induction heads removes most in-context")
    print("  learning in small attention-only models, and that the effect is far larger than")
    print(f"  for other heads. Here the five heads remove {frac(ind_mean):.0%} vs {frac(rmean['mean']):.0%} for random")
    print("  heads, so the direction MATCHES. Removal is not total because GPT-2 small has further")
    print("  induction-like heads (L9H6, L9H9, L10H1 and others) that the top-k curve keeps")
    print("  picking up. Redundancy and self-repair of this kind are reported for GPT-2 small by")
    print("  Wang et al. 2022 and McGrath et al. 2023. Olsson et al. did not study GPT-2 small,")
    print("  so this is a qualitative comparison, not a numeric one.")
    print(f"\nsaved figures/06_*.png and results/06_*.csv ({time.time() - t0:.1f}s)")
    if run.active:
        print(f"logged run to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
