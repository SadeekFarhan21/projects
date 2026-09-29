# tiny-circuits

A reproducible mechanistic-interpretability study of **GPT-2 small** (124M),
built with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

**Research question:** *How does GPT-2 small implement in-context copying?*
We reverse-engineer the **induction circuit** end to end — discovery, causal
localization, weight-level mechanism, and ablation — then extend to the IOI
circuit as a second case study.

## Setup

```bash
uv sync
uv run python experiments/01_induction_scores.py
```

Runs locally on Apple Silicon (MPS), CUDA, or CPU. No API keys required.

## Experiments

Each script is self-contained and writes figures to `figures/` and tables to
`results/`, so the paper is fully reproducible from source.

| # | Script | Question | Status |
|---|--------|----------|--------|
| 01 | `experiments/01_induction_scores.py` | Which heads are induction heads? | ✅ |
| 02 | `experiments/02_attention_viz.py` | What do the top heads attend to? | ✅ |
| 03 | `experiments/03_activation_patching.py` | Which components are *causally* responsible? | ✅ |
| 04 | `experiments/04_prev_token_heads.py` | Which upstream heads feed the induction heads? | ✅ |
| 05 | `experiments/05_qk_ov_analysis.py` | Do the weights show attend-back + copy? | ✅ |
| 06 | `experiments/06_ablation.py` | Are these heads necessary? | ✅ |

## Findings so far

All numbers use repeated random sequences [BOS][A][A] with N = 50 and seed 1337.

- **01 Induction scores.** Top induction heads are **L5H5, L6H9, L5H1, L7H10, L7H2**
  (scores 0.81 to 0.91), then a gap to L10H1 (0.52). This matches the heads named in
  ARENA's TransformerLens tutorial.
- **02 Attention patterns.** On second-half queries the five heads put 0.82 to 0.93 of
  their attention on the induction target (mean 0.89) and almost none on the previous
  token. The contrast head L4H11 puts 0.99 on the previous token and 0.00 on the target.
- **03 Activation patching.** Corrupting the first half drops the second-half mean log
  prob from -0.22 to -12.36. Residual patching shows the information moving from
  first-half to second-half positions between resid_pre 5 and 8 (crossover at layer 7).
  Patching all five induction heads recovers 0.81 of performance (denoising) and
  removing them loses 0.77 (noising), while no single head does more than 0.13 and 0.16.
  L9H6 and L9H9 also recover about 0.10 each; the copy-suppression heads L10H7 (-0.09)
  and L11H10 (-0.06) push the other way. Path patching isolates L4H11 as the upstream
  head: its direct path into the induction heads' keys carries 0.24 of performance,
  against at most 0.013 for any other L0 to L4 head, and its query and value paths
  carry below 0.002. Its total effect is smaller (0.09) because MLP4 partly offsets the
  corruption (holding MLP4 clean restores the effect to 0.26), and zero-ablating L4H11
  costs only 0.02.
- **04 Previous-token heads.** L4H11 is by far the strongest previous-token head (0.99 on
  random tokens, 1.00 on English text). The next are L3H7 (0.54) and L2H2 (0.49). No
  layer 0 head plays this role, so the two-layer "L0 previous-token head" story does not
  carry over directly. L4H11 has the highest K-composition with the induction heads
  (mean 0.096 vs Q 0.052, V 0.038, random baseline 0.036), and across layers 0 to 4 the
  previous-token score and K-composition have Spearman correlation 0.78.
- **05 QK and OV circuits.** With the effective embedding (W_E plus MLP0), the OV
  circuits of L6H9, L7H10 and L7H2 map a token to itself for 87 to 92% of 2,000 random
  tokens (top-5 96 to 99%). L5H1 reaches 56% (top-5 79%) and L5H5 only 16% (top-5 35%).
  Positive-eigenvalue shares are 0.95 to 0.997, against -0.005 for a norm-matched random
  head. With the raw embedding the rates fall to 0.4 to 6%, so MLP0 acts as an extended
  embedding. The QK circuit, with L4H11 feeding the keys, picks the key whose previous
  token matches the query for 99% of tokens in every induction head. Direct logit
  attribution to the correct token on the second half is L7H2 +1.51, L7H10 +1.16,
  L6H9 +1.08, L5H1 +0.69 and L5H5 +0.30 (all 0.00 on the first half). L9H9, L9H6 and
  L10H0 add about as much, and L10H7 (-1.21) and L11H10 (-0.77) push against it.
- **06 Ablation.** The intact model has first-half loss 13.17, second-half loss 0.24 and
  ICL score -12.94 nats. Mean-ablating the five induction heads moves the ICL score to
  -8.69 (second-half loss 4.46), removing 33% of it; five random heads remove 1%
  (-12.86, sd 0.06 over 20 sets). No single induction head matters alone under mean
  ablation (each within 0.05 nats of intact); zero-ablating L5H1 costs 0.63 nats. Adding L4H11 takes the score to -6.44. Ablating the top k heads
  by induction score keeps eroding it (k = 10 gives -5.34, k = 20 gives -1.13) while
  random k heads stay near -12.7, so GPT-2 small has many backup induction heads. In
  the single-head knockout sweep L4H11 is the most important head (+0.29 nats).

## Layout

```
src/tiny_circuits/   shared utilities (model loading, metrics)
experiments/         numbered, reproducible experiment scripts
figures/  results/   generated outputs (committed for the writeup)
paper/               the writeup itself
notebooks/           scratch exploration
```
