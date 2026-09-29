---
layout: post
title: "Erasing Artists From Stable Diffusion at Right Angles"
code: https://github.com/SadeekFarhan21/projects/tree/main/cure-improve
date: 2026-02-27 13:48:53
tags:
  - diffusion
  - unlearning
  - linear-algebra
description: >-
  An orthogonal subspace bank on top of training-free CURE erasure cuts
  collateral drift in Stable Diffusion v1.4 from 5 to 50 sequential erasures
  (LPIPS_u 0.7375 against 0.7735 at k=5), then reverses at 100.
---

CURE<sup>[[1]](#ref-1)</sup> erases a concept, such as a painter's style, from a text-to-image diffusion model with one closed-form edit to the cross-attention key and value weights. No training. The problem is erasing more than one. Erase concepts one after another and the edits don't compose cleanly: each pair leaves a cross term that damages images you never meant to touch. CURE-Improve is a small team research codebase built around one fix, CURE-Sequential. It keeps a bank of every direction already erased and forces each new concept's projector to be orthogonal to all of them, which makes the cross term zero by construction.

On Stable Diffusion v1.4<sup>[[2]](#ref-2)</sup>, with 3 seeds at 384 px, the bank helps up to a point. From 5 to 50 erased artists, CURE-Sequential drifts less on artists it was never asked to erase (LPIPS<sub>u</sub> 0.7375 against 0.7735 at k=5, CLIP<sub>u</sub> up 0.64 to 2.23 points). At 100 erasures it flips and is worse on both measures (LPIPS<sub>u</sub> 0.7939 against 0.7680, CLIP<sub>u</sub> 21.25 against 21.69). That lines up with the idea's one real cost: CLIP's text space has 768 dimensions, and every erased concept spends some of them for good.

## Why It Matters

Concept erasure is for the case where a model has to stop producing something. CURE erases a concept in about two seconds with no gradient steps, which makes it attractive for the "erase a hundred artists" setting. That is also where it degrades. The CURE paper's Figure 6 shows perceptual divergence starting around 50 erasures and stronger interference on untargeted prompts beyond about 100. The question for this project was whether that failure can be fixed without training anything.

The work was split. Arses Prasai built the implementation: a reimplementation of CURE for SD v1.4 (`cure/`, not the authors' code), the sequential variant (`cure_seq/`), a port of the same idea to SD3's MM-DiT architecture (`cure_dit/`), and a shared evaluation harness (`evaluation/`) that scores every method the same way. He also ran the quick-proof benchmark. Jeffrey Xie ran the Figure-6-style sweeps behind the main results. My part is the analysis in this post and two small weight-free scripts, written with AI help. One recomputes the method gaps from the committed results, and the other checks the linear algebra on synthetic data.

## Technical Details

### The CURE Edit

Take the text encoder's embeddings of prompts describing the concept, a matrix $E_f \in \mathbb{R}^{n \times 768}$. Its SVD gives right singular vectors $V_f$. CURE builds an energy-weighted projector onto that subspace and subtracts it from the key and value weights<sup>[[1]](#ref-1)</sup>:

$$
P = V_f^{\top} \operatorname{diag}\big(f(r_i;\alpha)\big) V_f, \qquad r_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}, \qquad f(r;\alpha) = \frac{\alpha r}{(\alpha - 1) r + 1}
$$

$$
W' = W - W P
$$

The function $f$ is a Tikhonov-style spectral expansion. It maps each direction's energy share $r_i$ to a weight in $[0, 1]$, so dominant directions are removed almost fully and weak ones barely at all. The repo also supports a Gavish-Donoho hard threshold<sup>[[3]](#ref-3)</sup> that keeps directions above a noise cutoff. In that mode the weight is a hard 0 or 1, so $\alpha$ plays no part. When retain prompts are given, the discriminative projector is $P_{dis} = P_f - P_f P_r$, which stops the edit from removing directions that a concept to keep also uses.

### Why Sequential Edits Interfere

Erase concept 1, then concept 2, with the same recipe.

$$
W_2 = W_1 (I - P_2) = W_0 (I - P_1)(I - P_2) = W_0 \big(I - P_1 - P_2 + P_1 P_2\big)
$$

The first three terms are what you wanted, one projector removed per concept. The last term, $W_0 P_1 P_2$, is an unintended edit. It is zero only when the two projectors annihilate each other, and two concepts that share directions (two painters with overlapping style words, say) won't. The cross term grows with the number of concepts.

### Orthogonalizing Against a Bank

If each new concept's basis is first projected off everything already erased, the subspaces are mutually orthogonal and the cross term is exactly zero for the forget projectors, since $V_i V_j^{\top} = 0$ implies $P_i P_j = 0$. The bank $B \in \mathbb{R}^{m \times 768}$ stores an orthonormal basis of everything erased so far. For a new $V_f$:

$$
V_f^{\perp} = V_f - (V_f B^{\top}) B
$$

then re-orthonormalize with QR and append.

The price is a finite budget. Every concept spends dimensions no later concept can use, and CLIP's embedding space has only 768 of them. The guarantee is also scoped to the forget subspaces. The retain projector $P_r$ is not orthogonalized against the bank, so with retain prompts the composed projectors are not exactly orthogonal.

<figure class="excal" data-diagram="cure-improve-architecture"><a href="/img/diagrams/cure-improve-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/cure-improve-architecture.webp" alt="Layout of the CURE-Improve repo: concept prompts become CLIP text embeddings of shape n by 768 and feed cure/ (the base CURE reimplementation editing attn2 to_k and to_v), cure_seq/ (a SubspaceBank with orthogonal projectors that imports from cure/) and cure_dit/ (targeting SD3 MM-DiT), all of which feed evaluation/ with one results.json schema computing LPIPS_e, LPIPS_u and CLIP_u, with labels crediting the packages and protocol to Arses Prasai and the Figure-6 sweeps to Jeffrey Xie." width="2400" height="1368" loading="lazy" decoding="async"></a></figure>

### How the Repo Fits Together

`cure/` holds the base eraser and the attention utilities that find and edit the `attn2` layers. `cure_seq/` imports those utilities and adds the `SubspaceBank`, a spectral module that computes orthogonalized projectors, and a `SequentialCURE` class. `cure_dit/` reuses the spectral idea for SD3. `evaluation/` runs any of them under one config and one `results.json` schema and computes the Figure-6 metrics. By line count that is about 2,100 lines for the base method, 1,300 for CURE-Sequential including experiments, 650 for CURE-DiT and 2,900 for evaluation.

<figure class="excal" data-diagram="cure-improve-subspace-bank"><a href="/img/diagrams/cure-improve-subspace-bank.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/cure-improve-subspace-bank.webp" alt="Flow of one sequential erasure in CURE-Sequential: forget prompts become CLIP embeddings and an SVD gives right singular vectors, an orthogonalize step against the SubspaceBank B and QR feed an adaptive alpha and the projector P, the weights are edited as W minus W P on to_k and to_v, and new directions with spectral weight above 0.01 are registered back into the bank, which lives in the 768-dimensional CLIP space so its capacity is bounded." width="2400" height="1428" loading="lazy" decoding="async"></a></figure>

### The DiT Port

SD3 has no cross-attention. Text and image tokens attend jointly, and the text stream has its own key and value projections. `cure_dit` applies the same edit to those, `add_k_proj` and `add_v_proj`, in each joint transformer block, in a 1152-dimensional context space, in float32 before casting back<sup>[[4]](#ref-4)</sup>. Every number in this post comes from SD v1.4.

### The Metrics

`paper_figure6_metrics.py` computes three numbers per checkpoint. LPIPS<sub>e</sub><sup>[[5]](#ref-5)</sup> is the perceptual distance from the unedited model on erased artists, and higher means stronger removal. LPIPS<sub>u</sub> is the same distance on unerased artists, and lower means less collateral damage. CLIP<sub>u</sub><sup>[[6]](#ref-6)</sup> is text-image alignment on unerased prompts, and higher is better. `run_shared_eval.py` writes one schema for all three methods, so every comparison below is like for like.

## Implementation

All code excerpts below are Arses's.

### The Weight Edit

The edit itself is two matrix products per attention layer, applied to the key and value projections.

```python
# cure/attention.py
Wk_new = Wk - Wk @ projector
Wv_new = Wv - Wv @ projector
```

The layers come from walking the SD v1.4 UNet's down, mid and up blocks for every transformer block's `attn2`, the cross-attention where image tokens attend to the 768-dimensional text embedding.

### The Subspace Bank

The core operation is a projection off the bank, a drop of rows the bank has already consumed, and a QR re-orthonormalization.

```python
# cure_seq/subspace_bank.py
overlap = Vhf @ B.T             # [k, m]
Vhf_orth = Vhf - overlap @ B    # remove what earlier concepts already own
norms = torch.norm(Vhf_orth, dim=1)
Vhf_orth = Vhf_orth[norms > self.orth_threshold]
Q, _ = torch.linalg.qr(Vhf_orth.T.cpu())   # QR is not implemented on MPS
Vhf_orth = Q.T.to(dev)
```

Registering a concept does not add all of its directions. Only directions whose spectral weight exceeds 0.01 go into the bank (problem 2 is why).

### Adaptive Alpha

Orthogonalizing removes some of a concept's energy, because part of it lived in directions earlier concepts already own. To compensate, the spectral expansion parameter is boosted in proportion to the energy lost.

```python
# cure_seq/spectral.py
boost = 1.0 / max(energy_retained, 0.1)
alpha_effective = min(alpha * boost, alpha_max)   # alpha_max = 10
```

## Problems

### 1. Cross Terms Accumulate With Every Erasure

The derivation above is the reason the project exists. **Plain sequential CURE leaves a $W_0 P_i P_j$ term for every pair of erased concepts, and that term shows up as damage on generations nobody asked to change.** The bank removes it for the forget subspaces. Problems 2 and 3 are what that fix costs.

### 2. One Concept Can Consume the Whole Space

A concept described by 20 prompts at 77 tokens each yields 1,540 embedding samples, which span all 768 dimensions. **Register every direction and the first concept exhausts the bank.** The fix is to register only directions with spectral weight above 0.01, which leaves roughly 20 to 40 directions per concept. The 5-concept demo used 133 of 768 dimensions (17.3%), about 27 per concept, at about 0.25 seconds per concept.

### 3. Orthogonality Has a Budget

At 20 to 40 directions per concept, **the bank holds roughly 20 to 40 concepts before the budget tightens.** That is well under the 100 erasures of the largest experiment. It is also why the k=100 result below is the interesting one. Past the budget, later concepts get less and less room of their own.

### 4. Orthogonalizing Weakens the Edit

Projecting a concept off the bank throws away whatever energy it shared with earlier concepts, so a late concept is erased more weakly than an early one. **Adaptive alpha compensates by boosting the spectral expansion by the inverse of the energy retained, capped at 10.** That keeps late erasures strong, but it also pushes a stronger edit through a smaller subspace, which is one of the two explanations for the k=100 reversal.

## Experiments

I ran no image generation. Everything that needs Stable Diffusion weights, a GPU and the LPIPS and CLIP models is Jeffrey's and Arses's output, and I recomputed only differences between numbers already in their results. Three sources feed this post.

1. **Figure-6-style sweeps (Jeffrey Xie).** SD v1.4, CURE against CURE-Sequential, seeds 11, 22 and 33, 20 sampling steps, guidance 7.5, 384 by 384 pixels, alpha 2.0, Tikhonov spectral mode, masked-mean embeddings. A 50-artist run on 2026-03-07 with checkpoints at k = 1, 5, 10, 25, 50, and a 100-artist extension on 2026-03-08 adding k = 100. The k = 1 to 50 checkpoints match between the two runs.
2. **Quick-proof benchmark (Arses Prasai).** Nine unique runs summarized on 2026-03-06, using a fast CLIP-based proxy with a "target" score (lower means better suppression) and a "drop" score (lower means better retention).
3. **A synthetic orthogonality check (mine, CPU, seconds).** Random low-rank concepts in 768 dimensions with no model in the loop. It tests only the linear algebra.

This is a fast setting, not a full benchmark. Three seeds, 20 steps and 384 pixels keep the sweeps cheap, and LPIPS at this scale is a proxy for what a user would call damage. Standard deviations are about 0.06 to 0.10 for LPIPS<sub>u</sub> (the largest, 0.098, is CURE-Sequential at k=100) and about 3 for CLIP<sub>u</sub>.

## Results

### Sequential Erasure on 50 and 100 Artists

<figure data-figure="chart:projects/cure-improve/cure-improve-lpips-u"></figure>

Differences, CURE-Sequential minus CURE, recomputed from Jeffrey's `results.json` by `results/figure6_deltas.py`.

| k | dLPIPS<sub>u</sub> (lower is better) | dCLIP<sub>u</sub> (higher is better) | dLPIPS<sub>e</sub> |
|---:|---:|---:|---:|
| 1 | +0.0000 | +0.000 | +0.0000 |
| 5 | -0.0360 | +1.392 | -0.0160 |
| 10 | -0.0304 | +2.232 | -0.0265 |
| 25 | -0.0244 | +1.036 | -0.0090 |
| 50 | -0.0142 | +0.641 | -0.0292 |
| 100 | +0.0260 | -0.439 | +0.0019 |

At k=1 the two methods are identical, as they must be, because the bank is empty. **From k=5 to k=50, CURE-Sequential has lower LPIPS<sub>u</sub> and higher CLIP<sub>u</sub> at every checkpoint, but the LPIPS<sub>u</sub> advantage shrinks steadily, from 0.036 to 0.014.** It is also less aggressive on the erased artists, with a lower LPIPS<sub>e</sub> at each of those checkpoints, which is the "more surgical" behavior the bank was designed for. **At k=100 the sign flips on both preservation metrics**, and CURE-Sequential's LPIPS<sub>u</sub> spread (0.098) is wider than CURE's (0.079).

<figure data-figure="chart:projects/cure-improve/cure-improve-clip-u"></figure>

LPIPS<sub>u</sub> is already 0.709 at k=1 for both methods, so the baseline drift from a single erasure is large in absolute terms. **The method gaps of 0.014 to 0.036 are smaller than or comparable to the LPIPS<sub>u</sub> standard deviations**, and the CLIP<sub>u</sub> gaps of 0.6 to 2.2 sit against a standard deviation of about 3. What holds up is the direction, which is the same at all four checkpoints from k=5 to k=50.

**The k=100 reversal fits the budget story**, since 100 concepts is far past the 20-to-40 estimate. The other candidate is adaptive alpha pushing a strong edit through a shrinking subspace (problem 4), and the sweeps don't separate the two.

### Quick-Proof Benchmark

**Across the nine summarized runs, the picture is mixed.** CURE-Sequential wins on both target and drop in the five Gavish-Donoho forward runs. Four of those are the same run in effect: the 20-concept forward runs at alpha 1.5, 2.0, 2.5 and 3.0 give identical numbers (target 26.7340 for CURE and 24.8841 for CURE-Sequential in all four), because the Gavish-Donoho threshold ignores alpha. In the reverse-order Gavish-Donoho run `gd_reverse_20c_v2`, CURE-Sequential has better retention (drop 1.4540 against 4.2923) and worse suppression (target 26.1652 against 24.3610). In the Tikhonov runs, `tikhonov_forward` favors CURE-Sequential on retention, and `tikhonov_reverse` gives it slightly better suppression and worse retention (drop 4.7717 against 3.5260). **Erasure order matters in the reverse runs.** Nine runs of 3 to 5 seeds each don't support more than that.

### The Linear Algebra Holds

The synthetic check builds random rank-8 concepts in 768 dimensions with 40 samples each and compares the projectors' pairwise products, with and without the bank, plus the cross term $\lVert W_0 P_0 P_1 \rVert_F$ for a random $W_0$. Its output matches the committed log exactly.

| Concepts | max off-diagonal $\lVert P_i P_j \rVert_F$, base | same, orthogonalized | bank dims used |
|---:|---:|---:|---:|
| 2 | 0.0163 | 2.3e-6 | 16 of 768 |
| 40 | 0.0239 | 7.5e-6 | 320 of 768 (41.7%) |

**The cross term $\lVert W_0 P_0 P_1 \rVert_F$ falls from 0.2982 to 4.2e-5, so the construction does what it says, to float32 rounding.** Random concepts in 768 dimensions are already nearly orthogonal, so the base projectors' cross terms are small to begin with (0.016 to 0.024). The check confirms the bank works as designed and shows the budget arithmetic at 8 dimensions per concept. It says nothing about real CLIP embeddings, where concepts overlap a lot.

## What I Would Change

### Log the Budget and Rerun k=100

Log the bank's dimension usage at every checkpoint and rerun the 100-artist sweep. Then run it again with adaptive alpha off. That would tell us whether the k=100 reversal is the budget, the adaptive alpha, or noise.

### Test the Retain Path

The zero-cross-term guarantee holds by construction for forget projectors only. A synthetic test with retain prompts, using the same harness as the orthogonality check, would show how large the residual cross term is when $P_r$ is not orthogonalized.

### Report Noise, Not Just Means

Report paired differences per seed and prompt with a confidence interval. A paired comparison would pin down how large CURE-Sequential's advantage at k=5 to 50 is.

### Measure Erasure Directly

LPIPS and CLIP score drift and alignment, not whether the artist is actually gone. A classifier or a round of human ratings on the erased artists would check the removal side, which the current metrics only approximate.

## References

1. <span id="ref-1"></span>Shristi Das Biswas, Arani Roy, Kaushik Roy. *CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models*. arXiv, 2025. [arXiv:2505.12677](https://arxiv.org/abs/2505.12677)
2. <span id="ref-2"></span>Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR, 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
3. <span id="ref-3"></span>Matan Gavish, David L. Donoho. *The Optimal Hard Threshold for Singular Values is 4/sqrt(3)*. IEEE Transactions on Information Theory, 2014. [arXiv:1305.5870](https://arxiv.org/abs/1305.5870)
4. <span id="ref-4"></span>Patrick Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML, 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
5. <span id="ref-5"></span>Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR, 2018. [arXiv:1801.03924](https://arxiv.org/abs/1801.03924)
6. <span id="ref-6"></span>Alec Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
