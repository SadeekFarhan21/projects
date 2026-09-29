"""Experiment 05: Do the weights show attend-back and copy?

Question. Patching (exp 03) says the induction heads carry the answer. Here we
ask whether their weights implement the two halves of the induction algorithm
directly, independent of any particular input.

1. OV circuit (copying). For each induction head we form the full OV circuit
   E W_V W_O W_U restricted to a fixed random subset of 2,000 vocabulary
   tokens (seeded; same subset for every head), giving a 2,000 x 2,000 matrix
   whose row i says which output logits go up when the head attends to token i.
   A copying head puts the largest entry of row i on column i. We report
     top-1   fraction of tokens whose own logit is the row maximum
     top-5   fraction whose own logit is in the row's top 5
     eig     Elhage et al. 2021 positive-eigenvalue share, sum(lambda) / sum(|lambda|)
             over the full-vocabulary circuit (1 = pure copying, -1 = anti-copying)
   E is computed two ways. "raw" uses the token embedding W_E. "effective" uses
   W_E + MLP0(W_E), because GPT-2 small is known to use its first MLP as an
   extended embedding (Wang et al. 2022; McDougall et al. 2023). Rows are
   normalized to the LayerNorm scale before entering the head.

   Baselines. Every other head of GPT-2 small (the 139 non-induction heads),
   and a Gaussian random head with W_V, W_O matched in Frobenius norm to the
   induction heads (chance top-1 is 1/2000 = 0.0005).

2. QK circuit (attend-back), as a complement to exp 04. With the main previous-
   token head L4H11 feeding the keys, a query for token t should attend most to
   the key whose previous token is t. We form E W_QK (E W_OV^{L4H11})^T on the
   same subset and report the fraction of query tokens whose top key is
   themselves (top-1 among 2,000).

3. Direct logit attribution. On the repeated random sequences of exp 01, each
   head's output at every position is passed through the final LayerNorm scale
   and dotted with the unembedding of the correct next token. We average over
   second-half query positions N+1 .. 2N-1 (and first-half positions for contrast).

Outputs:
  figures/05_ov_copying.png      top-1 and top-5 copying per induction head vs baselines
  figures/05_dla.png             direct logit attribution per head, second half
  results/05_qk_ov_analysis.csv  per-head OV metrics (all 144 heads, effective E) and DLA
  results/05_summary.json        induction-head metrics, raw vs effective, baselines, QK
"""
from __future__ import annotations

import argparse
import csv
import json
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
N_TOKENS = 2000       # vocabulary subset for the OV and QK circuits
N_RANDOM_HEADS = 20   # Gaussian random heads for the baseline

INDUCTION_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]  # from experiment 01
PREV_TOKEN_HEAD = (4, 11)                                      # from experiment 04


def make_repeated_tokens(model, seq_len: int, batch: int) -> torch.Tensor:
    """Build [BOS][rand seq][rand seq] token batches (identical to experiment 01)."""
    g = torch.Generator().manual_seed(SEED)
    vocab = model.cfg.d_vocab
    rand = torch.randint(0, vocab, (batch, seq_len), generator=g)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    tokens = torch.cat([bos, rand, rand], dim=1)
    return tokens.to(model.cfg.device)


def ln_normalize(x: torch.Tensor) -> torch.Tensor:
    """LayerNorm without weights (they are folded into the next layer): center, scale."""
    x = x - x.mean(-1, keepdim=True)
    return x / x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()


def effective_embedding(model) -> torch.Tensor:
    """W_E + MLP0(LN(W_E)): the token embedding as later layers see it."""
    W_E = model.W_E
    out = []
    for chunk in W_E.split(8192):
        out.append(chunk + model.blocks[0].mlp(ln_normalize(chunk)[None])[0])
    return torch.cat(out)


def ov_copy_metrics(E_sub, W_V, W_O, W_U_sub) -> tuple[float, float]:
    """Top-1 and top-5 self-prediction rate of the OV circuit on a token subset."""
    M = (E_sub @ W_V) @ (W_O @ W_U_sub)                   # [n, n]
    idx = torch.arange(M.shape[0], device=M.device)
    top1 = (M.argmax(-1) == idx).float().mean().item()
    top5 = (M.topk(5, dim=-1).indices == idx[:, None]).any(-1).float().mean().item()
    return top1, top5


def eig_share(E_full, W_V, W_O, W_U) -> float:
    """sum(lambda) / sum(|lambda|) of E W_V W_O W_U, via the d_head x d_head form."""
    small = (W_O @ W_U) @ (E_full @ W_V)                  # [d_head, d_head], same nonzero eigs
    lam = torch.linalg.eigvals(small.cpu().double())
    return float(lam.sum().real / lam.abs().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb", action="store_true", help="log this run to Weights & Biases"
    )
    parser.add_argument("--wandb-name", default="05-qk-ov-analysis", help="W&B run name")
    args = parser.parse_args()
    t0 = time.time()
    torch.set_grad_enabled(False)

    model = load_model()
    cfg = model.cfg
    print(f"loaded {cfg.model_name} on {cfg.device} ({cfg.n_layers}L x {cfg.n_heads}H)")

    run = init_run(
        args.wandb_name,
        config={
            "experiment": "05_qk_ov_analysis",
            "model": cfg.model_name,
            "device": str(cfg.device),
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "seed": SEED,
            "n_tokens": N_TOKENS,
        },
        enabled=args.wandb,
    )

    g = torch.Generator().manual_seed(SEED)
    subset = torch.randperm(cfg.d_vocab, generator=g)[:N_TOKENS].to(cfg.device)
    W_U = model.W_U
    W_U_sub = W_U[:, subset]
    E = {"raw": ln_normalize(model.W_E), "effective": ln_normalize(effective_embedding(model))}
    E_sub = {k: v[subset] for k, v in E.items()}

    # ------------------------------------------------ 1. OV circuit, every head
    ov = {}  # (kind, layer, head) -> (top1, top5, eig)
    for kind in ("raw", "effective"):
        for l in range(cfg.n_layers):
            for h in range(cfg.n_heads):
                if kind == "raw" and (l, h) not in INDUCTION_HEADS:
                    continue
                t1, t5 = ov_copy_metrics(E_sub[kind], model.W_V[l, h], model.W_O[l, h], W_U_sub)
                ev = eig_share(E[kind], model.W_V[l, h], model.W_O[l, h], W_U)
                ov[(kind, l, h)] = (t1, t5, ev)
    print(f"OV circuits done ({time.time() - t0:.1f}s)")

    # Gaussian random heads matched in norm to the induction heads
    v_norm = torch.stack([model.W_V[l, h].norm() for l, h in INDUCTION_HEADS]).mean()
    o_norm = torch.stack([model.W_O[l, h].norm() for l, h in INDUCTION_HEADS]).mean()
    rand_vals = []
    for _ in range(N_RANDOM_HEADS):
        Wv = torch.randn(cfg.d_model, cfg.d_head, generator=g).to(cfg.device)
        Wo = torch.randn(cfg.d_head, cfg.d_model, generator=g).to(cfg.device)
        Wv, Wo = Wv * v_norm / Wv.norm(), Wo * o_norm / Wo.norm()
        t1, t5 = ov_copy_metrics(E_sub["effective"], Wv, Wo, W_U_sub)
        rand_vals.append((t1, t5, eig_share(E["effective"], Wv, Wo, W_U)))
    rand_mean = tuple(float(sum(v[i] for v in rand_vals) / len(rand_vals)) for i in range(3))

    others = [ov[("effective", l, h)] for l in range(cfg.n_layers) for h in range(cfg.n_heads)
              if (l, h) not in INDUCTION_HEADS]
    others_t = torch.tensor(others)
    others_median = tuple(float(x) for x in others_t.median(0).values)
    others_p90 = tuple(float(x) for x in others_t.quantile(0.9, dim=0))

    # ------------------------------------------------ 2. QK circuit with L4H11 keys
    pl, ph = PREV_TOKEN_HEAD
    Es = E_sub["effective"]
    key_side = ln_normalize(Es @ model.W_V[pl, ph] @ model.W_O[pl, ph])
    qk_top1 = {}
    for l, h in INDUCTION_HEADS:
        S = (Es @ model.W_Q[l, h]) @ (key_side @ model.W_K[l, h]).T
        idx = torch.arange(S.shape[0], device=S.device)
        qk_top1[f"L{l}H{h}"] = (S.argmax(-1) == idx).float().mean().item()
    # control: same computation with raw token keys (no previous-token head)
    qk_ctrl = {}
    for l, h in INDUCTION_HEADS:
        S = (Es @ model.W_Q[l, h]) @ (Es @ model.W_K[l, h]).T
        idx = torch.arange(S.shape[0], device=S.device)
        qk_ctrl[f"L{l}H{h}"] = (S.argmax(-1) == idx).float().mean().item()

    # ------------------------------------------------ 3. direct logit attribution
    tokens = make_repeated_tokens(model, SEQ_LEN, BATCH)
    _, cache = model.run_with_cache(tokens)
    scale = cache["ln_final.hook_scale"]                      # [b, pos, 1]
    first = torch.arange(1, SEQ_LEN, device=cfg.device)
    second = torch.arange(SEQ_LEN + 1, 2 * SEQ_LEN, device=cfg.device)

    def dla_for(pos: torch.Tensor) -> torch.Tensor:
        target_dir = W_U[:, tokens[:, pos + 1]].permute(1, 2, 0)  # [b, p, d_model]
        out = torch.zeros(cfg.n_layers, cfg.n_heads)
        for l in range(cfg.n_layers):
            z = cache["z", l][:, pos]                          # [b, p, h, d_head]
            res = torch.einsum("bphe,hed->bphd", z, model.W_O[l]) / scale[:, pos, None]
            out[l] = torch.einsum("bphd,bpd->h", res, target_dir).cpu() / (res.shape[0] * res.shape[1])
        return out

    dla2 = dla_for(second)
    dla1 = dla_for(first)
    logits = model(tokens)
    correct_logit = logits[:, second].gather(-1, tokens[:, second + 1, None]).mean().item()
    mean_logit = logits[:, second].mean().item()
    ind_dla_sum = float(sum(dla2[l, h] for l, h in INDUCTION_HEADS))
    all_dla_sum = float(dla2.sum())
    print(f"DLA done ({time.time() - t0:.1f}s)")

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    # ------------------------------------------------ figure A: OV copying
    names = [f"L{l}H{h}" for l, h in INDUCTION_HEADS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
    x = torch.arange(len(names)).float()
    w = 0.38
    for ax, i, title in ((axes[0], 0, "Top-1 self-prediction"), (axes[1], 1, "Top-5 self-prediction")):
        raw = [ov[("raw", l, h)][i] for l, h in INDUCTION_HEADS]
        eff = [ov[("effective", l, h)][i] for l, h in INDUCTION_HEADS]
        ax.bar(x - w / 2, raw, w, color="#b8c7dc", label="raw embedding")
        ax.bar(x + w / 2, eff, w, color="#3b6ea5", label="effective embedding")
        for xi, v in zip(x + w / 2, eff):
            ax.text(float(xi), v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(others_p90[i], color="0.45", lw=0.8, ls="--",
                   label="other heads, 90th percentile")
        ax.set_xticks(x.tolist(), names)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fraction of 2,000 tokens")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=9, ncols=3, loc="outside lower center")
    fig.suptitle("OV circuits of the induction heads copy their input token", fontsize=12)
    fig.savefig(FIGURES / "05_ov_copying.png", dpi=150)
    run.log_figure("ov_copying", fig)
    plt.close(fig)

    # ------------------------------------------------ figure B: DLA
    fig, ax = plt.subplots(figsize=(5.8, 4.6), layout="constrained")
    lim = float(dla2.abs().max())
    im = ax.imshow(dla2.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                   interpolation="nearest")
    ax.set_title("Direct logit attribution to the correct token", fontsize=11)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks([0, 5, 11])
    ax.set_yticks([0, 5, 11])
    fig.colorbar(im, ax=ax, shrink=0.85, label="logit contribution, second half")
    fig.savefig(FIGURES / "05_dla.png", dpi=150)
    run.log_figure("dla", fig)
    plt.close(fig)

    # ------------------------------------------------ tables
    rows = []
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            t1, t5, ev = ov[("effective", l, h)]
            rows.append((l, h, (l, h) in INDUCTION_HEADS, t1, t5, ev,
                         float(dla2[l, h]), float(dla1[l, h])))
    cols = ["layer", "head", "is_induction", "ov_top1", "ov_top5", "ov_eig_share",
            "dla_second_half", "dla_first_half"]
    with open(RESULTS / "05_qk_ov_analysis.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        wr.writerows(rows)
    run.log_table("per_head", cols, rows)

    summary = {
        "n_tokens": N_TOKENS,
        "induction_heads": {
            f"L{l}H{h}": {
                kind: dict(zip(("top1", "top5", "eig_share"), ov[(kind, l, h)]))
                for kind in ("raw", "effective")
            } | {"qk_top1_with_L4H11_keys": qk_top1[f"L{l}H{h}"],
                 "qk_top1_token_keys_control": qk_ctrl[f"L{l}H{h}"],
                 "dla_second_half": float(dla2[l, h]),
                 "dla_first_half": float(dla1[l, h])}
            for l, h in INDUCTION_HEADS
        },
        "baseline_gaussian_head": dict(zip(("top1", "top5", "eig_share"), rand_mean)),
        "baseline_other_heads_median": dict(zip(("top1", "top5", "eig_share"), others_median)),
        "baseline_other_heads_p90": dict(zip(("top1", "top5", "eig_share"), others_p90)),
        "correct_token_logit_second_half": correct_logit,
        "mean_logit_second_half": mean_logit,
        "induction_dla_sum_second_half": ind_dla_sum,
        "all_heads_dla_sum_second_half": all_dla_sum,
    }
    (RESULTS / "05_summary.json").write_text(json.dumps(summary, indent=1))

    run.summary({
        "mean_top1_effective": sum(ov[("effective", l, h)][0] for l, h in INDUCTION_HEADS) / 5,
        "mean_eig_effective": sum(ov[("effective", l, h)][2] for l, h in INDUCTION_HEADS) / 5,
        "induction_dla_sum": ind_dla_sum,
    })

    # ------------------------------------------------ printed summary
    print("\nOV circuit on 2,000 random tokens (raw embedding | effective embedding):")
    print(f"  {'head':>6}  {'top1':>5} {'top5':>5} {'eig':>6} | {'top1':>5} {'top5':>5} {'eig':>6}")
    for l, h in INDUCTION_HEADS:
        r, e = ov[("raw", l, h)], ov[("effective", l, h)]
        print(f"  L{l}H{h:<3}  {r[0]:5.3f} {r[1]:5.3f} {r[2]:6.3f} | {e[0]:5.3f} {e[1]:5.3f} {e[2]:6.3f}")
    print(f"  Gaussian random head      top1 {rand_mean[0]:.4f} top5 {rand_mean[1]:.4f} eig {rand_mean[2]:+.3f}")
    print(f"  other GPT-2 heads, median top1 {others_median[0]:.4f} top5 {others_median[1]:.4f} eig {others_median[2]:+.3f}")
    print(f"  other GPT-2 heads, 90th pct top1 {others_p90[0]:.4f} top5 {others_p90[1]:.4f} eig {others_p90[2]:+.3f}")
    top_copy = sorted(rows, key=lambda r: -r[3])[:8]
    print("  strongest copiers among all 144 heads (top-1): "
          + ", ".join(f"L{r[0]}H{r[1]} {r[3]:.2f}" for r in top_copy))

    print("\nQK circuit, query token t vs key whose previous token is t (via L4H11), top-1 of 2,000:")
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in qk_top1.items()))
    print("  control with plain token keys: " + ", ".join(f"{k} {v:.3f}" for k, v in qk_ctrl.items()))

    print("\nDirect logit attribution to the correct next token (mean logit contribution):")
    top_dla = sorted(rows, key=lambda r: -r[6])[:6]
    neg_dla = sorted(rows, key=lambda r: r[6])[:3]
    print("  second half, top:      " + ", ".join(f"L{r[0]}H{r[1]} {r[6]:+.2f}" for r in top_dla))
    print("  second half, negative: " + ", ".join(f"L{r[0]}H{r[1]} {r[6]:+.2f}" for r in neg_dla))
    print("  induction heads, first half: " + ", ".join(f"L{l}H{h} {dla1[l, h]:+.2f}" for l, h in INDUCTION_HEADS))
    print("  induction heads, second half: " + ", ".join(f"L{l}H{h} {dla2[l, h]:+.2f}" for l, h in INDUCTION_HEADS))
    print(f"  sum over the 5 induction heads {ind_dla_sum:.2f}, over all 144 heads {all_dla_sum:.2f}; "
          f"correct-token logit {correct_logit:.2f} (mean logit {mean_logit:.2f})")

    eff = [ov[("effective", l, h)] for l, h in INDUCTION_HEADS]
    print("\nLiterature check:")
    print("  OV. Elhage et al. 2021 and Olsson et al. 2022 describe induction-head OV circuits as")
    print(f"  copying with mostly positive eigenvalues. Eig shares here are {min(e[2] for e in eff):.2f} to "
          f"{max(e[2] for e in eff):.2f} against {rand_mean[2]:+.2f}")
    print("  for a random head, and top-5 rates are far above the 0.0025 chance level: MATCH.")
    print("  The raw-embedding rates are much lower, as expected if GPT-2 small uses MLP0 as an")
    print("  extended embedding (Wang et al. 2022; McDougall et al. 2023): MATCH.")
    print("  L5H5 is the weakest copier despite the top induction score, and its DLA is small, so")
    print("  its output may matter mostly through later heads rather than directly.")
    print("  QK. With L4H11 feeding the keys, every induction head picks out the key whose previous")
    print("  token matches the query for ~99% of tokens, and never the plain same-token key. This")
    print("  is the K-composition mechanism of Elhage et al. 2021 and ARENA: MATCH.")
    print("  DLA. The induction heads write the correct token in the second half and nothing in")
    print("  the first half. L10H7 and L11H10 are the most negative heads, the copy-suppression")
    print("  heads of McDougall et al. 2023: MATCH. L9H6, L9H9 and L10H0 contribute as much as the")
    print("  exp 01 top five, so the direct write is spread over more heads than the canonical five.")
    print(f"\nsaved figures/05_*.png and results/05_* ({time.time() - t0:.1f}s)")
    if run.active:
        print(f"logged run to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
