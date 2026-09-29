"""Experiment 04: Which upstream heads feed the induction heads?

Question. An induction head needs to know, at each earlier key position j,
which token came *before* j, so that a query for token t can match the key
whose previous token was t. In the canonical two-layer story that information
is written by a previous-token head. Here we find GPT-2 small's previous-token
heads and test, from the weights alone, whether they compose with the induction
heads through keys.

1. Previous-token score per head. Mean attention from position i to i - 1,
   over all non-BOS query positions. Measured on the repeated random sequences
   of experiment 01 and, as a robustness check, on a short passage of English
   (so the score does not depend on the random-token setting).

2. Composition scores (Elhage et al. 2021). For an upstream head A with OV
   matrix W_OV^A and a downstream head B with QK matrix W_QK^B (TransformerLens
   row-vector convention, x W_OV and x_q W_QK x_k^T):
     K-composition  ||W_OV^A (W_QK^B)^T||_F / (||W_OV^A||_F ||W_QK^B||_F)
     Q-composition  ||W_OV^A W_QK^B||_F     / (same norms)
     V-composition  ||W_OV^A W_OV^B||_F     / (||W_OV^A||_F ||W_OV^B||_F)
   computed for every head in layers 0 to 4 against each of the five induction
   heads. The baseline is the same score for random matrices with the same
   low-rank shapes (Gaussian factors, 200 draws), as in ARENA's tutorial.

Outputs:
  figures/04_prev_token_heads.png    prev-token heatmap and prev-token vs K-comp scatter
  results/04_prev_token_scores.csv   layer, head, score on random tokens, score on text
  results/04_composition_scores.csv  upstream head, induction head, K, Q, V composition
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from tiny_circuits.model import load_model, SEED
from tiny_circuits.tracking import init_run

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
SEQ_LEN = 50          # length of the random sequence (repeated once -> 2*SEQ_LEN)
BATCH = 8             # same batch as experiment 01
UPSTREAM_LAYERS = range(0, 5)
INDUCTION_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]  # from experiment 01
N_BASELINE = 200

TEXT = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk. He "
    "cleaned the lens, trimmed the wick, and wrote the weather in a small leather "
    "book. Ships passing the rocky point would see the beam sweep across the water "
    "and turn away from the reef. In winter the storms came in from the north, and "
    "the tower shook, but the light never went out. Years later his daughter took "
    "over the job, and she kept the same book on the same shelf by the window."
)


def make_repeated_tokens(model, seq_len: int, batch: int) -> torch.Tensor:
    """Build [BOS][rand seq][rand seq] token batches (identical to experiment 01)."""
    g = torch.Generator().manual_seed(SEED)
    vocab = model.cfg.d_vocab
    rand = torch.randint(0, vocab, (batch, seq_len), generator=g)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    tokens = torch.cat([bos, rand, rand], dim=1)
    return tokens.to(model.cfg.device)


def prev_token_scores(model, tokens: torch.Tensor) -> torch.Tensor:
    """[n_layers, n_heads] mean attention from i to i - 1, over queries i >= 2.

    Query 1 is skipped because its previous token is BOS, which many heads
    attend to by default, so it would inflate every head's score.
    """
    _, cache = model.run_with_cache(
        tokens, return_type=None, names_filter=lambda n: n.endswith("hook_pattern")
    )
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        pattern = cache["pattern", layer]            # [batch, head, q, k]
        diag = pattern.diagonal(offset=-1, dim1=-2, dim2=-1)[..., 1:]
        scores[layer] = diag.mean(dim=(0, -1)).cpu()
    return scores


def comp_score(a, b) -> float:
    """Elhage et al. composition score ||AB||_F / (||A||_F ||B||_F).

    a and b are FactoredMatrix objects, so the product is never materialized
    as a dense d_model x d_model matrix more than necessary.
    """
    return float((a @ b).norm() / (a.norm() * b.norm()))


def random_baseline(model, n: int) -> float:
    """Composition score between random low-rank matrices of the OV/QK shape."""
    from transformer_lens import FactoredMatrix

    g = torch.Generator().manual_seed(SEED)
    d_model, d_head = model.cfg.d_model, model.cfg.d_head
    vals = []
    for _ in range(n):
        a = FactoredMatrix(torch.randn(d_model, d_head, generator=g),
                           torch.randn(d_head, d_model, generator=g))
        b = FactoredMatrix(torch.randn(d_model, d_head, generator=g),
                           torch.randn(d_head, d_model, generator=g))
        vals.append(comp_score(a, b))
    return float(torch.tensor(vals).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb", action="store_true", help="log this run to Weights & Biases"
    )
    parser.add_argument("--wandb-name", default="04-prev-token-heads", help="W&B run name")
    args = parser.parse_args()
    t0 = time.time()

    model = load_model()
    cfg = model.cfg
    print(f"loaded {cfg.model_name} on {cfg.device} ({cfg.n_layers}L x {cfg.n_heads}H)")

    run = init_run(
        args.wandb_name,
        config={
            "experiment": "04_prev_token_heads",
            "model": cfg.model_name,
            "device": str(cfg.device),
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "seed": SEED,
            "n_baseline": N_BASELINE,
        },
        enabled=args.wandb,
    )

    # ------------------------------------------------ previous-token scores
    with torch.no_grad():
        rand_scores = prev_token_scores(model, make_repeated_tokens(model, SEQ_LEN, BATCH))
        text_tokens = model.to_tokens(TEXT)          # prepends BOS
        text_scores = prev_token_scores(model, text_tokens)

    prev_rows = [(l, h, float(rand_scores[l, h]), float(text_scores[l, h]))
                 for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    prev_rows.sort(key=lambda r: -r[2])
    upstream_rows = [r for r in prev_rows if r[0] in UPSTREAM_LAYERS]

    # ------------------------------------------------ composition scores
    with torch.no_grad():
        OV = model.OV   # FactoredMatrix [n_layers, n_heads, d_model, d_model]
        QK = model.QK
        comp_rows = []
        for ul in UPSTREAM_LAYERS:
            for uh in range(cfg.n_heads):
                ov_a = OV[ul, uh]
                for dl, dh in INDUCTION_HEADS:
                    k = comp_score(ov_a, QK[dl, dh].T)
                    q = comp_score(ov_a, QK[dl, dh])
                    v = comp_score(ov_a, OV[dl, dh])
                    comp_rows.append((ul, uh, f"L{dl}H{dh}", k, q, v))
        baseline = random_baseline(model, N_BASELINE)
    print(f"composition scores done ({time.time() - t0:.1f}s)")

    # mean over the five induction heads, per upstream head
    mean_comp: dict[tuple[int, int], tuple[float, float, float]] = {}
    for ul in UPSTREAM_LAYERS:
        for uh in range(cfg.n_heads):
            rs = [r for r in comp_rows if r[0] == ul and r[1] == uh]
            mean_comp[(ul, uh)] = tuple(sum(r[i] for r in rs) / len(rs) for i in (3, 4, 5))
    by_k = sorted(mean_comp.items(), key=lambda kv: -kv[1][0])

    # rank correlation between prev-token score and mean K-comp in layers 0-4
    xs = torch.tensor([rand_scores[l, h] for (l, h) in mean_comp])
    ys = torch.tensor([v[0] for v in mean_comp.values()])
    rank = lambda t: t.argsort().argsort().float()
    spearman = float(torch.corrcoef(torch.stack([rank(xs), rank(ys)]))[0, 1])

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # ------------------------------------------------ figure
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    im = ax0.imshow(rand_scores.numpy(), cmap="viridis", origin="lower", vmin=0, vmax=1,
                    interpolation="nearest")
    ax0.set_title("Previous-token score per head", fontsize=11)
    ax0.set_xlabel("head")
    ax0.set_ylabel("layer")
    ax0.set_xticks([0, 5, 11])
    ax0.set_yticks([0, 5, 11])
    fig.colorbar(im, ax=ax0, shrink=0.85, ticks=[0, 0.5, 1], label="attention to previous token")

    ax1.scatter(xs.numpy(), ys.numpy(), s=18, color="#3b6ea5", alpha=0.8)
    ax1.axhline(baseline, color="0.5", lw=0.8, ls="--")
    ax1.text(0.99, baseline, "random baseline", ha="right", va="bottom", fontsize=8,
             color="0.4", transform=ax1.get_yaxis_transform())
    labelled = {prev_rows[0][:2], by_k[0][0], by_k[1][0]}
    for (l, h) in labelled:
        ax1.annotate(f"L{l}H{h}", (float(rand_scores[l, h]), mean_comp[(l, h)][0]),
                     xytext=(-6, 4), textcoords="offset points", fontsize=9, ha="right")
    ax1.set_title("Layers 0 to 4, K-composition into induction heads", fontsize=11)
    ax1.set_xlabel("previous-token score")
    ax1.set_ylabel("mean K-composition score")
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_yticks([0.03, 0.06, 0.09])
    ax1.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "04_prev_token_heads.png", dpi=150)
    run.log_figure("prev_token_heads", fig)

    # ------------------------------------------------ tables
    with open(RESULTS / "04_prev_token_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "prev_score_random", "prev_score_text"])
        w.writerows(prev_rows)
    with open(RESULTS / "04_composition_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["up_layer", "up_head", "induction_head", "k_comp", "q_comp", "v_comp"])
        w.writerows(comp_rows)
        w.writerow(["random", "baseline", "", baseline, baseline, baseline])
    run.log_table("prev_token_scores", ["layer", "head", "random", "text"], prev_rows)
    run.log_table("composition", ["up_layer", "up_head", "induction_head", "k", "q", "v"], comp_rows)

    top = prev_rows[0]
    run.summary({
        "top_prev_token_head": f"L{top[0]}H{top[1]}",
        "top_prev_token_score": top[2],
        "top_k_comp_head": f"L{by_k[0][0][0]}H{by_k[0][0][1]}",
        "top_k_comp": by_k[0][1][0],
        "k_comp_baseline": baseline,
        "spearman_prev_vs_kcomp": spearman,
    })

    # ------------------------------------------------ summary
    print("\nTop previous-token heads (random tokens, English text):")
    for l, h, r, t in prev_rows[:8]:
        print(f"  L{l}H{h:<3} {r:.3f}  {t:.3f}")
    print("\nMean composition with the five induction heads (layers 0 to 4), top by K:")
    print(f"  {'head':>6} {'K':>6} {'Q':>6} {'V':>6}   prev-token")
    for (l, h), (k, q, v) in by_k[:8]:
        print(f"  L{l}H{h:<3} {k:6.3f} {q:6.3f} {v:6.3f}   {float(rand_scores[l, h]):.3f}")
    print(f"  random baseline {baseline:.3f}")
    k411 = mean_comp.get((4, 11), (float("nan"),) * 3)
    per_ind = [r for r in comp_rows if r[0] == 4 and r[1] == 11]
    print("  L4H11 K-comp per induction head: "
          + ", ".join(f"{r[2]} {r[3]:.3f}" for r in per_ind))
    print(f"  Spearman(prev-token score, mean K-comp) over layers 0 to 4 = {spearman:.2f}")

    print("\nLiterature check:")
    print(f"  Top previous-token head is L{top[0]}H{top[1]} ({top[2]:.2f} on random tokens, "
          f"{top[3]:.2f} on text).")
    print("  Wang et al. 2022 (IOI) and ARENA name 4.11 as GPT-2 small's main previous-token")
    print("  head, with 2.2 as a weaker one. The 'L0 previous-token head' of the two-layer story")
    print("  does not carry over: in GPT-2 small the role sits in layer 4.")
    print(f"  L4H11 mean K-comp {k411[0]:.3f} vs Q {k411[1]:.3f} and V {k411[2]:.3f}, baseline "
          f"{baseline:.3f}. Elhage et al. 2021 predict K > Q, V for the prev-token edge.")
    print("  ARENA and Elhage et al. note composition scores are a noisy signal in real")
    print("  models, so exp 03's path patching is the stronger evidence for this edge.")
    print(f"\nsaved figures/04_prev_token_heads.png and results/04_*.csv ({time.time() - t0:.1f}s)")
    if run.active:
        print(f"logged run to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
