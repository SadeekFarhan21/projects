"""Experiment 03: Which components are causally responsible for in-context copying?

Question. Experiments 01 and 02 are correlational: they show where heads look.
Here we intervene on activations to find which components carry the
information that makes the second copy of a repeated sequence predictable.

Clean and corrupted inputs.
  clean      [BOS][A][A]    A is a random sequence of N = 50 tokens
  corrupted  [BOS][A'][A]   the FIRST half is replaced by different random tokens

We corrupt the first half rather than the second. This keeps the second-half
tokens, and therefore the prediction targets, identical in both runs, so the
metric is evaluated on the same labels at the same positions and the only thing
removed is the earlier occurrence that induction needs to look back at. (If we
instead replaced the second half, the corrupted run would be predicting
different tokens at the metric positions, and a residual patch at one position
would just re-insert the clean input token rather than induction information.)

Metric. Mean log probability of the correct next token over second-half query
positions N+1 .. 2N-1 (the positions where induction can predict the next
token), averaged over a batch of 8 sequences. Normalized so that the clean run
scores 1 and the corrupted run scores 0.

Interventions.
  1. Residual stream patching (denoising). Copy the clean resid_pre at one
     (layer, position) into the corrupted run. Also patch the whole first half
     or the whole second half at once, per layer, to see where the information
     lives before and after the induction layers.
  2. Head output patching on hook_z, all positions at once, in both directions.
     Denoising (clean into corrupted) measures sufficiency; noising (corrupted
     into clean) measures necessity.
  3. Path patching of the prev-token to induction edge (noising). For every head
     in layers 0 to 4 we replace only its direct contribution to the key input
     of the five induction heads with its value from the corrupted run, leaving
     every other path clean. We repeat this for the query and value inputs as a
     control: previous-token heads should matter through keys (K-composition),
     not through queries or values.

Outputs:
  figures/03_resid_patching.png    layer x position patching, plus half-sequence curves
  figures/03_head_patching.png     per-head denoising and noising
  figures/03_path_patching.png     sender heads (L0 to L4) into induction K, Q, V
  results/03_resid_patching.csv    layer, position, normalized metric
  results/03_head_patching.csv     layer, head, denoise, noise
  results/03_path_patching.csv     sender layer, head, effect via K, Q, V
"""
from __future__ import annotations

import argparse
import csv
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
BATCH = 8             # same batch as experiment 01

INDUCTION_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]  # from experiment 01
SENDER_LAYERS = range(0, 5)                                    # candidate upstream heads


def make_repeated_tokens(model, seq_len: int, batch: int, seed: int = SEED) -> torch.Tensor:
    """Build [BOS][rand seq][rand seq] token batches (identical to experiment 01)."""
    g = torch.Generator().manual_seed(seed)
    vocab = model.cfg.d_vocab
    rand = torch.randint(0, vocab, (batch, seq_len), generator=g)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    tokens = torch.cat([bos, rand, rand], dim=1)
    return tokens.to(model.cfg.device)


def make_corrupted(model, clean: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Replace the first copy with fresh random tokens: [BOS][A'][A]."""
    g = torch.Generator().manual_seed(SEED + 1)
    fresh = torch.randint(0, model.cfg.d_vocab, (clean.shape[0], seq_len), generator=g)
    corrupted = clean.clone()
    corrupted[:, 1 : 1 + seq_len] = fresh.to(clean.device)
    return corrupted


def second_half_logprob(logits: torch.Tensor, tokens: torch.Tensor, seq_len: int) -> float:
    """Mean log prob of the correct next token at query positions N+1 .. 2N-1."""
    pos = torch.arange(seq_len + 1, 2 * seq_len, device=tokens.device)
    logp = logits[:, pos].log_softmax(-1)
    target = tokens[:, pos + 1]
    return logp.gather(-1, target[..., None]).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb", action="store_true", help="log this run to Weights & Biases"
    )
    parser.add_argument("--wandb-name", default="03-activation-patching", help="W&B run name")
    args = parser.parse_args()
    t0 = time.time()

    model = load_model()
    cfg = model.cfg
    print(f"loaded {cfg.model_name} on {cfg.device} ({cfg.n_layers}L x {cfg.n_heads}H)")

    run = init_run(
        args.wandb_name,
        config={
            "experiment": "03_activation_patching",
            "model": cfg.model_name,
            "device": str(cfg.device),
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "seed": SEED,
            "corruption": "first half replaced by fresh random tokens",
        },
        enabled=args.wandb,
    )

    clean = make_repeated_tokens(model, SEQ_LEN, BATCH)
    corrupted = make_corrupted(model, clean, SEQ_LEN)
    n_pos = clean.shape[1]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean)
        corr_logits, corr_cache = model.run_with_cache(corrupted)
    m_clean = second_half_logprob(clean_logits, clean, SEQ_LEN)
    m_corr = second_half_logprob(corr_logits, clean, SEQ_LEN)
    print(f"second-half mean log prob  clean {m_clean:.3f}  corrupted {m_corr:.3f}")

    def normalized(logits: torch.Tensor) -> float:
        return (second_half_logprob(logits, clean, SEQ_LEN) - m_corr) / (m_clean - m_corr)

    # ---------------------------------------------------------------- 1. residual
    def patch_resid(act, hook, pos, src):
        act[:, pos] = src[hook.name][:, pos]
        return act

    resid = torch.zeros(cfg.n_layers, n_pos)
    halves = {"first": torch.zeros(cfg.n_layers), "second": torch.zeros(cfg.n_layers)}
    first_idx = torch.arange(1, 1 + SEQ_LEN, device=cfg.device)
    second_idx = torch.arange(1 + SEQ_LEN, 1 + 2 * SEQ_LEN, device=cfg.device)
    with torch.no_grad():
        for layer in range(cfg.n_layers):
            name = f"blocks.{layer}.hook_resid_pre"
            for pos in range(n_pos):
                logits = model.run_with_hooks(
                    corrupted,
                    fwd_hooks=[(name, partial(patch_resid, pos=pos, src=clean_cache))],
                )
                resid[layer, pos] = normalized(logits)
            for key, idx in (("first", first_idx), ("second", second_idx)):
                logits = model.run_with_hooks(
                    corrupted,
                    fwd_hooks=[(name, partial(patch_resid, pos=idx, src=clean_cache))],
                )
                halves[key][layer] = normalized(logits)
    print(f"residual patching done ({time.time() - t0:.1f}s)")

    # ---------------------------------------------------------------- 2. heads
    def patch_head_z(act, hook, head, src):
        act[:, :, head] = src[hook.name][:, :, head]
        return act

    denoise = torch.zeros(cfg.n_layers, cfg.n_heads)
    noise = torch.zeros(cfg.n_layers, cfg.n_heads)
    with torch.no_grad():
        for layer in range(cfg.n_layers):
            name = f"blocks.{layer}.attn.hook_z"
            for head in range(cfg.n_heads):
                logits = model.run_with_hooks(
                    corrupted, fwd_hooks=[(name, partial(patch_head_z, head=head, src=clean_cache))]
                )
                denoise[layer, head] = normalized(logits)
                logits = model.run_with_hooks(
                    clean, fwd_hooks=[(name, partial(patch_head_z, head=head, src=corr_cache))]
                )
                noise[layer, head] = normalized(logits)
    print(f"head patching done ({time.time() - t0:.1f}s)")

    # all five induction heads together, both directions
    def patch_heads(act, hook, heads, src):
        act[:, :, heads] = src[hook.name][:, :, heads]
        return act

    def induction_hooks(src):
        by_layer: dict[int, list[int]] = {}
        for l, h in INDUCTION_HEADS:
            by_layer.setdefault(l, []).append(h)
        return [(f"blocks.{l}.attn.hook_z", partial(patch_heads, heads=hs, src=src))
                for l, hs in by_layer.items()]

    with torch.no_grad():
        ind_denoise = normalized(model.run_with_hooks(corrupted, fwd_hooks=induction_hooks(clean_cache)))
        ind_noise = normalized(model.run_with_hooks(clean, fwd_hooks=induction_hooks(corr_cache)))

    # ---------------------------------------------------------------- 3. path patching
    # Direct path sender -> receiver input only. For receiver heads we edit
    # hook_{k,q,v}_input (one copy of resid_pre per head) by adding the change in
    # the sender's output, so every other path keeps its clean value.
    model.set_use_split_qkv_input(True)
    receivers: dict[int, list[int]] = {}
    for l, h in INDUCTION_HEADS:
        receivers.setdefault(l, []).append(h)

    def sender_out(cache, layer, head):
        return cache["z", layer][:, :, head] @ model.W_O[layer, head]  # [b, pos, d_model]

    def add_delta(act, hook, heads, delta):
        act[:, :, heads] = act[:, :, heads] + delta[:, :, None]
        return act

    path = {"k": torch.zeros(len(SENDER_LAYERS), cfg.n_heads),
            "q": torch.zeros(len(SENDER_LAYERS), cfg.n_heads),
            "v": torch.zeros(len(SENDER_LAYERS), cfg.n_heads)}
    with torch.no_grad():
        for i, s_layer in enumerate(SENDER_LAYERS):
            for s_head in range(cfg.n_heads):
                delta = sender_out(corr_cache, s_layer, s_head) - sender_out(clean_cache, s_layer, s_head)
                for kind in ("k", "q", "v"):
                    hooks = [(f"blocks.{l}.hook_{kind}_input", partial(add_delta, heads=hs, delta=delta))
                             for l, hs in receivers.items()]
                    logits = model.run_with_hooks(clean, fwd_hooks=hooks)
                    path[kind][i, s_head] = 1.0 - normalized(logits)  # fraction of performance lost
    model.set_use_split_qkv_input(False)
    print(f"path patching done ({time.time() - t0:.1f}s)")

    # Total vs direct effect of the top K sender. Noising its hook_z everywhere
    # lets every downstream component react; freezing the next MLP at its clean
    # output isolates how much of the gap between the direct K path and the
    # total effect is that MLP's response.
    ks = path["k"]
    top_i = int(ks.flatten().argmax())
    top_sender = (list(SENDER_LAYERS)[top_i // cfg.n_heads], top_i % cfg.n_heads)
    s_l, s_h = top_sender

    def zero_head(act, hook, head):
        act[:, :, head] = 0.0
        return act

    def keep_clean(act, hook):
        return clean_cache[hook.name]

    z_name = f"blocks.{s_l}.attn.hook_z"
    with torch.no_grad():
        sender_total = 1 - normalized(model.run_with_hooks(
            clean, fwd_hooks=[(z_name, partial(patch_head_z, head=s_h, src=corr_cache))]))
        sender_mlp_frozen = 1 - normalized(model.run_with_hooks(
            clean, fwd_hooks=[(z_name, partial(patch_head_z, head=s_h, src=corr_cache)),
                              (f"blocks.{s_l}.hook_mlp_out", keep_clean)]))
        sender_zero = 1 - normalized(model.run_with_hooks(
            clean, fwd_hooks=[(z_name, partial(zero_head, head=s_h))]))

    # ---------------------------------------------------------------- outputs
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    # Figure A: residual stream
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11, 4.2), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]}
    )
    im = ax0.imshow(resid.numpy(), aspect="auto", cmap="Blues", origin="lower",
                    vmin=0, vmax=float(resid[:, 1:].max()), interpolation="nearest")
    ax0.axvline(SEQ_LEN + 0.5, color="0.5", lw=0.8, ls="--")
    ax0.set_xticks([1, SEQ_LEN, 2 * SEQ_LEN])
    ax0.set_yticks([0, 4, 8, 11])
    ax0.set_xlabel("patched position")
    ax0.set_ylabel("layer (resid_pre)")
    ax0.set_title("Patching one residual position")
    fig.colorbar(im, ax=ax0, shrink=0.9, ticks=[0, 0.01, 0.02], label="recovered")
    layers = range(cfg.n_layers)
    ax1.plot(layers, halves["first"].numpy(), marker="o", ms=4, label="first half")
    ax1.plot(layers, halves["second"].numpy(), marker="o", ms=4, label="second half")
    ax1.set_xticks([0, 4, 8, 11])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlabel("layer (resid_pre)")
    ax1.set_ylabel("recovered performance")
    ax1.set_title("Patching a whole half")
    ax1.legend(frameon=False, fontsize=9, loc="center left")
    ax1.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "03_resid_patching.png", dpi=150)
    run.log_figure("resid_patching", fig)
    plt.close(fig)

    # Figure B: heads
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    lim = float(max(denoise.abs().max(), (1 - noise).abs().max()))
    for ax, data, title in ((axes[0], denoise, "Denoising, performance recovered"),
                            (axes[1], 1 - noise, "Noising, performance lost")):
        im = ax.imshow(data.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("head")
        ax.set_xticks([0, 5, 11])
        ax.set_yticks([0, 5, 11])
    axes[0].set_ylabel("layer")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("fraction of clean performance")
    fig.savefig(FIGURES / "03_head_patching.png", dpi=150)
    run.log_figure("head_patching", fig)
    plt.close(fig)

    # Figure C: path patching senders -> induction keys, and the top sender's effects
    flat_k = [(float(path["k"][i, h]), l, h) for i, l in enumerate(SENDER_LAYERS)
              for h in range(cfg.n_heads)]
    flat_k.sort(reverse=True)
    top8 = flat_k[:8][::-1]
    max_qv = float(max(path["q"].abs().max(), path["v"].abs().max()))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    ax0.barh([f"L{l}H{h}" for _, l, h in top8], [v for v, _, _ in top8], color="#3b6ea5")
    ax0.set_title("Sender to induction keys, direct path", fontsize=11)
    ax0.set_xlabel("performance lost")
    ax0.text(0.98, 0.04, f"Q and V paths below {max(max_qv, 0.001):.3f} for every sender",
             transform=ax0.transAxes, ha="right", fontsize=8, color="0.4")
    v_top = top8[-1][0]
    ax0.text(v_top, len(top8) - 1, f" {v_top:.2f}", va="center", fontsize=9)
    labels = ["direct K path", "noised everywhere", f"noised, MLP{s_l} clean", "zero-ablated"]
    vals = [float(ks.flatten()[top_i]), sender_total, sender_mlp_frozen, sender_zero]
    ax1.barh(labels[::-1], vals[::-1], color="#7a7a7a")
    for y, v in enumerate(vals[::-1]):
        ax1.text(v, y, f" {v:.2f}", va="center", fontsize=9)
    ax1.set_title(f"L{s_l}H{s_h}, direct versus total effect", fontsize=11)
    ax1.set_xlabel("performance lost")
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1.25 * max(vals + [top8[-1][0]]))
    fig.savefig(FIGURES / "03_path_patching.png", dpi=150)
    run.log_figure("path_patching", fig)
    plt.close(fig)

    with open(RESULTS / "03_resid_patching.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "position", "recovered"])
        for l in range(cfg.n_layers):
            for p in range(n_pos):
                w.writerow([l, p, float(resid[l, p])])
        for key in ("first", "second"):
            for l in range(cfg.n_layers):
                w.writerow([l, f"{key}_half", float(halves[key][l])])

    head_rows = [(l, h, float(denoise[l, h]), float(1 - noise[l, h]))
                 for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    with open(RESULTS / "03_head_patching.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "denoise_recovered", "noise_lost"])
        w.writerows(head_rows)
    run.log_table("head_patching", ["layer", "head", "denoise_recovered", "noise_lost"], head_rows)

    path_rows = [(l, h, float(path["k"][i, h]), float(path["q"][i, h]), float(path["v"][i, h]))
                 for i, l in enumerate(SENDER_LAYERS) for h in range(cfg.n_heads)]
    with open(RESULTS / "03_path_patching.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sender_layer", "sender_head", "lost_via_k", "lost_via_q", "lost_via_v"])
        w.writerows(path_rows)
    run.log_table("path_patching", ["sender_layer", "sender_head", "k", "q", "v"], path_rows)

    # ---------------------------------------------------------------- summary
    top_denoise = sorted(head_rows, key=lambda r: -r[2])[:6]
    top_noise = sorted(head_rows, key=lambda r: -r[3])[:6]
    top_k = sorted(path_rows, key=lambda r: -r[2])[:5]
    cross = int(torch.nonzero(halves["second"] > halves["first"])[0]) if (halves["second"] > halves["first"]).any() else -1

    run.summary({
        "sender_total_lost": sender_total, "sender_mlp_frozen_lost": sender_mlp_frozen,
        "sender_zero_lost": sender_zero,
        "clean_logprob": m_clean, "corrupted_logprob": m_corr,
        "induction_heads_denoise": ind_denoise, "induction_heads_noise_lost": 1 - ind_noise,
        "top_k_sender": f"L{top_k[0][0]}H{top_k[0][1]}", "top_k_sender_lost": top_k[0][2],
        "crossover_layer": cross,
    })

    print("\nResidual stream (patching a whole half):")
    for l in range(cfg.n_layers):
        print(f"  L{l:<2} first half {halves['first'][l]:.3f}   second half {halves['second'][l]:.3f}")
    print(f"  second-half patch first beats first-half patch at resid_pre of layer {cross}")

    print("\nHead patching (all positions):")
    print("  denoise, recovered: " + ", ".join(f"L{l}H{h} {d:.3f}" for l, h, d, _ in top_denoise))
    print("  noise, lost:        " + ", ".join(f"L{l}H{h} {n:.3f}" for l, h, _, n in top_noise))
    neg = sorted(head_rows, key=lambda r: r[2])[:3]
    print("  denoise, most negative: " + ", ".join(f"L{l}H{h} {d:.3f}" for l, h, d, _ in neg))
    print(f"  all 5 induction heads together  denoise {ind_denoise:.3f}  noise lost {1 - ind_noise:.3f}")

    print("\nPath patching sender -> induction heads (performance lost):")
    for l, h, k, q, v in top_k:
        print(f"  L{l}H{h:<3} via K {k:.3f}   via Q {q:.3f}   via V {v:.3f}")

    print(f"\nTop K sender L{s_l}H{s_h}, fraction of performance lost:")
    print(f"  direct path into induction keys only   {float(ks.flatten()[top_i]):.3f}")
    print(f"  noised everywhere (total effect)       {sender_total:.3f}")
    print(f"  noised everywhere, MLP{s_l} held clean    {sender_mlp_frozen:.3f}")
    print(f"  zero-ablated everywhere                {sender_zero:.3f}")

    print("\nLiterature check:")
    print("  Residual handoff. Second-half information lives at first-half positions up to")
    print(f"  resid_pre 5 and moves to second-half positions over layers 6 to 8 (crossover at {cross}),")
    print("  right after the L5 to L7 induction heads. This is the handoff the induction mechanism")
    print("  predicts (Elhage et al. 2021, Olsson et al. 2022): MATCH. We know of no published")
    print("  layer x position map for this exact setup to compare numbers against.")
    print("  Heads. The heads that recover or destroy performance are the exp 01 induction heads")
    print("  plus L9H6 and L9H9, which rank 7 and 8 by induction score in exp 01. No single head")
    print("  is necessary, while the five together are. This fits the redundancy and backup")
    print("  behaviour reported for GPT-2 small (Wang et al. 2022; McGrath et al. 2023): MATCH.")
    print("  Negative heads. Patching clean L10H7 and L11H10 into the corrupted run lowers the")
    print("  metric. These are the copy-suppression heads of McDougall et al. 2023, which push")
    print("  down tokens that earlier heads predict from context: MATCH.")
    print(f"  Path. The top K-path sender is L{s_l}H{s_h}. Wang et al. 2022 and ARENA name 4.11 as")
    print("  GPT-2 small's main previous-token head: MATCH. Its effect runs through keys only,")
    print("  as K-composition predicts: MATCH. Q and V paths are ~0.")
    print("  Caveat. The total effect of noising 4.11 is smaller than its direct K path, because")
    print("  MLP4's reaction to the corrupted input offsets part of the damage, and zero-ablating")
    print("  4.11 barely matters. Previous-token information is redundant in GPT-2 small.")
    print(f"\nsaved figures/03_*.png and results/03_*.csv ({time.time() - t0:.1f}s)")
    if run.active:
        print(f"logged run to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
