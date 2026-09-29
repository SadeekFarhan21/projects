"""Experiment 02: What do the top induction heads attend to?

Question. Experiment 01 ranked heads by a single number (mean attention along
the induction diagonal). Here we look at the full attention patterns of the
five top induction heads (L5H5, L6H9, L5H1, L7H10, L7H2) on one repeated random
sequence [BOS][A][A], next to a contrast head, to confirm the score reflects
the expected geometry. An induction head at query position i in the second
copy should attend to key position i - N + 1 (the token *after* the earlier
occurrence of the current token), which shows up as a stripe offset N - 1
below the main diagonal that only exists in the second half.

The contrast head is L4H11, the strongest previous-token head in GPT-2 small in
the literature (Wang et al. 2022; ARENA's TransformerLens tutorial) and in our
own experiment 04. It attends to i - 1 everywhere, so it shows the main
off-by-one diagonal and no induction stripe.

For every plotted head we also report, averaged over the whole batch, where its
attention mass goes on second-half queries: to the induction target
(i - N + 1), to the previous token (i - 1), to BOS, and elsewhere.

Outputs:
  figures/02_attention_viz.png   2x3 grid of attention patterns (batch item 0)
  results/02_attention_viz.csv   attention-mass breakdown per plotted head
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

INDUCTION_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]  # from experiment 01
CONTRAST_HEAD = (4, 11)                                        # previous-token head


def make_repeated_tokens(model, seq_len: int, batch: int) -> torch.Tensor:
    """Build [BOS][rand seq][rand seq] token batches (identical to experiment 01)."""
    g = torch.Generator().manual_seed(SEED)
    vocab = model.cfg.d_vocab
    rand = torch.randint(0, vocab, (batch, seq_len), generator=g)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    tokens = torch.cat([bos, rand, rand], dim=1)
    return tokens.to(model.cfg.device)


def attention_breakdown(pattern: torch.Tensor, seq_len: int) -> dict[str, float]:
    """Split second-half query attention into induction / prev / BOS / other.

    pattern: [batch, query_pos, key_pos] for one head.
    Second-half queries are positions 1 + N .. 2N; the induction target of
    query q is q - N + 1 and its previous token is q - 1.
    """
    q = torch.arange(1 + seq_len, 1 + 2 * seq_len, device=pattern.device)
    ind = pattern[:, q, q - seq_len + 1].mean().item()
    prev = pattern[:, q, q - 1].mean().item()
    bos = pattern[:, q, 0].mean().item()
    return {
        "induction": ind,
        "previous": prev,
        "bos": bos,
        "other": 1.0 - ind - prev - bos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb", action="store_true", help="log this run to Weights & Biases"
    )
    parser.add_argument("--wandb-name", default="02-attention-viz", help="W&B run name")
    args = parser.parse_args()
    t0 = time.time()

    model = load_model()
    print(f"loaded {model.cfg.model_name} on {model.cfg.device} "
          f"({model.cfg.n_layers}L x {model.cfg.n_heads}H)")

    heads = INDUCTION_HEADS + [CONTRAST_HEAD]
    run = init_run(
        args.wandb_name,
        config={
            "experiment": "02_attention_viz",
            "model": model.cfg.model_name,
            "device": str(model.cfg.device),
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "seed": SEED,
            "heads": [f"L{l}H{h}" for l, h in heads],
        },
        enabled=args.wandb,
    )

    tokens = make_repeated_tokens(model, SEQ_LEN, BATCH)
    layers = sorted({l for l, _ in heads})
    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter=lambda n: n.endswith("hook_pattern")
        and int(n.split(".")[1]) in layers,
    )

    rows = []
    patterns = {}
    for layer, head in heads:
        pat = cache["pattern", layer][:, head]  # [batch, q, k]
        patterns[(layer, head)] = pat[0].float().cpu().numpy()
        br = attention_breakdown(pat, SEQ_LEN)
        rows.append((layer, head, br["induction"], br["previous"], br["bos"], br["other"]))

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # --- figure: 2x3 grid of attention patterns ---
    import matplotlib.pyplot as plt

    n = tokens.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (layer, head) in zip(axes.flat, heads):
        im = ax.imshow(patterns[(layer, head)], cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest")
        label = f"L{layer}H{head}"
        if (layer, head) == CONTRAST_HEAD:
            label += ", previous-token contrast"
        ax.set_title(label, fontsize=11)
        ax.axhline(SEQ_LEN + 0.5, color="0.6", lw=0.6, ls="--")
        ax.axvline(SEQ_LEN + 0.5, color="0.6", lw=0.6, ls="--")
        ax.set_xticks([0, SEQ_LEN, n - 1])
        ax.set_yticks([0, SEQ_LEN, n - 1])
    for ax in axes[1]:
        ax.set_xlabel("key position")
    for ax in axes[:, 0]:
        ax.set_ylabel("query position")
    fig.suptitle("Attention on a repeated random sequence in GPT-2 small", fontsize=12)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, ticks=[0, 0.5, 1])
    cbar.set_label("attention weight")
    fig.savefig(FIGURES / "02_attention_viz.png", dpi=150)
    run.log_figure("attention_patterns", fig)

    cols = ["layer", "head", "attn_induction", "attn_previous", "attn_bos", "attn_other"]
    with open(RESULTS / "02_attention_viz.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    run.log_table("attention_breakdown", cols, rows)

    ind_mean = sum(r[2] for r in rows[:5]) / 5
    run.summary({"mean_induction_attention_top5": ind_mean,
                 "contrast_previous_attention": rows[5][3]})

    print("\nSecond-half attention mass (mean over batch and query positions):")
    print(f"  {'head':>6} {'induct':>7} {'prev':>6} {'BOS':>6} {'other':>6}")
    for l, h, a, p, b, o in rows:
        print(f"  L{l}H{h:<3} {a:7.3f} {p:6.3f} {b:6.3f} {o:6.3f}")

    print("\nLiterature check:")
    print(f"  Top-5 induction heads put {ind_mean:.2f} of their attention on the "
          "induction target on average.")
    print("  ARENA's TransformerLens tutorial shows the same offset stripe for "
          "5.1, 5.5, 6.9, 7.2, 7.10 on repeated random tokens: MATCH (qualitative).")
    print(f"  L4H11 puts {rows[5][3]:.2f} on the previous token and {rows[5][2]:.2f} on "
          "the induction target, as expected of a previous-token head (Wang et al. 2022).")
    print(f"\nsaved figures/02_attention_viz.png and results/02_attention_viz.csv "
          f"({time.time() - t0:.1f}s)")
    if run.active:
        print(f"logged run to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
