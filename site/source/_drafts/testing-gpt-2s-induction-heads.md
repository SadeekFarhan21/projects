---
layout: post
title: "Testing GPT-2's Induction Heads and the Backups Behind Them"
code: https://github.com/SadeekFarhan21/projects/tree/main/tiny-circuits
tags:
  - interpretability
  - transformers
  - research
description: >-
  Activation patching, path patching, weight analysis and ablation on GPT-2
  small's five induction heads. They carry the copy together, one layer 4 head
  feeds their keys, and removing them costs only a third of in-context learning.
date: 2026-09-12 21:49:27
---

The [first post](/posts/reverse-engineering-gpt-2s-induction-circuit/) scored every attention head in GPT-2 small<sup>[[1]](#ref-1)</sup> and found five induction heads, **L5H5, L6H9, L7H10, L5H1 and L7H2**, with a gap of 0.290 to the sixth. It closed by saying an induction score is a correlation, not a cause, and that four experiments would decide whether the five heads do the copying or only look in the right place. This post reports those experiments, numbered 02 to 06 in the project.

The short version has four parts. The five heads carry the copy together but not alone. Patching all five recovers about 0.81 of clean performance, while no single head recovers more than 0.128. Their keys are fed by one upstream head, **L4H11** in layer 4, and not by a layer 0 head as the textbook two-layer story and our own outline assumed. Their weights implement the two halves of induction, attend back and copy, but the head with the highest induction score, L5H5, is the weakest direct copier of the five. And removing all five costs only **33%** of the model's in-context learning score, because GPT-2 small keeps many other induction heads in reserve.

That last result is the one the first post warned about. It does not undo the circuit. It changes what "the induction circuit" can mean in this model, from a pair of heads to a redundant population with a few members that matter more than the rest.

<figure class="excal" data-diagram="tiny-circuits-2-recovered-circuit"><a href="/img/diagrams/tiny-circuits-2-recovered-circuit.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/tiny-circuits-2-recovered-circuit.webp" alt="A left-to-right circuit diagram of GPT-2 small in which the effective embedding feeds a layer 4 previous-token head, L4H11, that supplies the keys of five induction heads in layers 5 to 7 while MLP4 partly compensates, copy suppression heads L10H7 and L11H10 oppose and backup heads run in parallel into the logits, with a callout that the textbook layer 0 previous-token head is absent and a badge that patching all five recovers 0.81." width="2400" height="1555" loading="lazy" decoding="async"></a></figure>

Everything runs on an M4 Pro laptop through MPS, with the same inputs as experiment 01. Every number below comes from the saved results of these runs unless the text says otherwise.

*Reading note.* The argument runs in the order of the experiments, from what the heads attend to, through what causes the output, to what the weights say and what removal costs. The surprises are in [The Upstream Head](#the-upstream-head), the L5H5 result in [The Weights](#the-weights), and [Necessity](#necessity). Skip to [Conclusions](#conclusions) for the claims and their limits.

## What the First Post Left Open

The claimed mechanism has two heads and one edge<sup>[[2]](#ref-2)</sup><sup>[[3]](#ref-3)</sup>. A previous-token head writes "the token before me was A" into each position. An induction head in a later layer uses that signature as its key, so a query for token A finds the position right after the last A, and its OV circuit copies the token there into the output. Experiment 01 tested only whether heads exist whose attention lands on the induction target. Experiment 03 tests causation by patching, experiments 03 and 04 test the upstream edge, experiment 05 reads the mechanism from the weights, and experiment 06 tests necessity by ablation. The first post's plan put path patching under experiment 04. It ended up in 03, next to the other patching runs.

## Inputs and Metrics

Every experiment uses the input from experiment 01, a BOS token followed by 50 random tokens and then the same 50 tokens again, with seed 1337. Experiments 02 to 05 use a batch of 8. Experiment 06 uses a batch of 32 whose first 8 rows are the same sequences, because loss differences between ablations are small and a larger batch steadies them.

Patching needs a corrupted input. We corrupt the *first* half with fresh random tokens, giving `[BOS][A'][A]`. The second half, and so every prediction target, is identical in both runs, and the only thing removed is the earlier occurrence induction needs to look back at. Corrupting the second half instead would change the labels at the scored positions, and a residual patch there would mostly re-insert the clean input token.

The patching metric is the mean log probability of the correct next token over second-half positions, normalized so the clean run scores 1 and the corrupted run scores 0. A patch that scores 0.4 recovered 40% of the gap between the two runs. In the clean run the second-half mean log probability is about -0.22 nats, and in the corrupted run it is about -12.36, so the gap is large and the normalization is well conditioned.

<figure class="excal" data-diagram="tiny-circuits-2-experiment-setup"><a href="/img/diagrams/tiny-circuits-2-experiment-setup.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/tiny-circuits-2-experiment-setup.webp" alt="Diagram of the patching setup: a clean run [BOS][A][A] and a corrupted run [BOS][A'][A] that differ only in the first half, so second-half targets and the normalized score (clean = 1, corrupted = 0) are identical, above three panels showing residual patching, head hook_z patching and path patching from a sender head into the induction heads' k/q/v inputs." width="2400" height="1772" loading="lazy" decoding="async"></a></figure>

The ablation metric in experiment 06 is an in-context learning score, the mean loss on the second half minus the mean loss on the first half, in the spirit of the ICL score in [Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)<sup>[[3]](#ref-3)</sup>. The first half is random, so its loss sits near chance. The second half is predictable only by copying. For the intact model the first-half loss is 13.17 nats, the second-half loss is 0.24, and the ICL score is -12.94. A score of 0 would mean the model gains nothing from having seen the sequence before.

## Attention Patterns

Experiment 02 plots the full attention pattern of each of the five heads on one sequence, next to L4H11, the head the literature names as GPT-2 small's main previous-token head. For every head we also split the attention of second-half queries into four buckets, the induction target, the previous token, BOS, and everything else, averaged over the batch.

| Head | Induction target | Previous token | BOS | Other |
| --- | --- | --- | --- | --- |
| L5H5 | 0.925 | 0.001 | 0.029 | 0.045 |
| L6H9 | 0.915 | 0.000 | 0.046 | 0.039 |
| L7H10 | 0.905 | 0.000 | 0.017 | 0.078 |
| L5H1 | 0.902 | 0.000 | 0.089 | 0.009 |
| L7H2 | 0.820 | 0.000 | 0.037 | 0.143 |
| L4H11 | 0.000 | 0.988 | 0.000 | 0.012 |

The induction column repeats the experiment 01 scores exactly, because it is the same quantity. What 02 adds is where the rest of the attention goes. None of the five heads puts more than 0.001 on the previous token, so they are not previous-token heads that happen to score well. Their leftover mass is split between BOS, where GPT-2 heads park attention when there is nothing to attend to, and a diffuse remainder that is largest for L7H2 at 0.143.

The plotted patterns show the geometry the score implies. In the first half, where no earlier copy exists, the five heads attend almost entirely to BOS. In the second half they draw one clean stripe offset 49 positions below the diagonal. L4H11 draws the off-by-one diagonal across the whole sequence and no stripe. This is still attention, and so still correlation. It rules out only that the scores come from some broader pattern overlapping the diagonal.

## Activation Patching

Experiment 03 intervenes on activations<sup>[[4]](#ref-4)</sup>. Each run takes one activation from the clean run and puts it into the corrupted run (denoising, which tests whether the component is sufficient to restore the answer), or takes one activation from the corrupted run and puts it into the clean run (noising, which tests whether the component is necessary)<sup>[[5]](#ref-5)</sup>.

### The Residual Stream Handoff

The first intervention patches the residual stream entering a layer at one half of the sequence at a time. If induction works as described, the information the second half needs starts out at first-half positions, since that is where the earlier copy lives, and is moved to second-half positions by the induction heads.

<figure data-figure="chart:tiny-circuits/residual-handoff"></figure>

That is what happens. Patching the clean first half into the corrupted run recovers 0.92 of performance at layers 4 and 5, then 0.35 at layer 6, 0.18 at layer 7 and 0.06 at layer 8. Patching the clean second half recovers nothing up to layer 5 (between -0.017 and 0.000), then 0.15 at layer 6, 0.46 at layer 7 and 0.87 at layer 8. The curves cross between layers 6 and 7. The handoff happens across exactly the layers where the five heads sit, 5 through 7, and it is complete by layer 8.

At layer 0 the two patches recover exactly 1.0 and 0.0, as they must, since the residual stream entering layer 0 is just the token embedding. After layer 8 the second-half curve keeps rising to 0.998 at layer 11, so later layers add to the answer without moving it again. Patching a single position is much weaker, at most about 0.022, because the task is spread over 50 positions.

### Heads One at a Time and Together

The second intervention patches one head's output, its attention-weighted values before the output projection, at all positions, in both directions.

<figure data-figure="chart:tiny-circuits/head-patching"></figure>

Single heads matter little. In the denoising direction the best head, L7H2, recovers 0.128 and L6H9 recovers 0.120. L5H5, the top head by induction score, recovers only 0.031. In the noising direction the picture is flatter still. Corrupting one induction head costs almost nothing, with one exception, L5H1, whose corruption loses 0.160. The other four each lose less than 0.003.

Patched together, the five heads are a different object. Denoising all five at once recovers about **0.81** of clean performance, and noising all five loses about **0.77**. The sum of the five single-head denoising effects is about 0.46, well short of 0.81, so the heads are more than additive when restored together. The reading we favor is redundancy. With four clean induction heads still running, corrupting a fifth changes little, because the others still deliver the copy. Only when all of them are corrupted at once does the redundancy run out.

The five are also not the only heads that recover performance. L9H6 recovers 0.104 and L9H9 recovers 0.103, level with the canonical heads. These rank seventh and eighth by induction score in experiment 01 (0.509 and 0.500). L10H0 recovers 0.082 and L10H1 recovers 0.067. The patching result is already hinting at the necessity result below. Induction in GPT-2 small is done by more heads than the five with the sharpest attention.

### Heads That Push Against the Answer

Two heads have clearly negative denoising effects. Restoring the clean output of **L10H7** lowers the metric by 0.092, and restoring **L11H10** lowers it by 0.055. Giving these heads their clean input makes the model worse at predicting the repeated token.

These are the heads [Wang et al. (2022)](https://arxiv.org/abs/2211.00593)<sup>[[6]](#ref-6)</sup> called negative name movers in the indirect-object circuit, and that [McDougall et al. (2023)](https://arxiv.org/abs/2310.04625)<sup>[[7]](#ref-7)</sup> reinterpreted as copy suppression heads. A copy suppression head attends to tokens that earlier heads are predicting from context and pushes their logits down, which calibrates overconfident copying. On a task where copying is always right, that calibration shows up as a cost. Experiment 05 confirms the mechanism from the weights.

## The Upstream Head

The outline for this project put the previous-token head in layer 0, following the two-layer attention-only model of [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html)<sup>[[2]](#ref-2)</sup>. In GPT-2 small it is not there.

### Path Patching into the Keys

Path patching isolates one edge<sup>[[6]](#ref-6)</sup><sup>[[8]](#ref-8)</sup>. For each of the 60 heads in layers 0 to 4, we take the difference between its corrupted and clean outputs and add it only to the key input of the five induction heads, leaving every other path, including the head's effects on MLPs and later heads, clean. We repeat this for the query and value inputs as a control. If a previous-token head feeds induction through K-composition, its effect should appear through keys and nowhere else.

<figure data-figure="chart:tiny-circuits/path-patching"></figure>

One sender dominates. Corrupting **L4H11**'s direct path into the induction keys loses **0.242** of clean performance. The next sender, L3H7, loses 0.013, and L2H2 loses 0.007. L4H11's query and value paths lose about 0.0001 each, and no sender's query or value path reaches 0.002. The edge is real, it runs through keys as K-composition predicts, and one head carries almost all of it.

### Why the Total Effect Is Smaller

Here is a surprise. L4H11's total effect is smaller than its direct path. Noising its output everywhere, rather than only on the path into the induction keys, loses 0.093 (from the head patching table), less than half of the 0.242 its key path alone loses.

The difference is MLP4, the MLP in the same layer. When L4H11's output is corrupted everywhere, MLP4 sees the corrupted output too and its response partly offsets the damage. Holding MLP4 at its clean output while noising L4H11 raises the loss to about 0.26, back to the size of the direct path. Zero-ablating L4H11 costs only about 0.02.

We do not know what MLP4 computes that lets it compensate. The simplest reading is that previous-token information is available to MLP4 from other sources, so it can partly rebuild the signal L4H11 would have written. Downstream components compensating for a damaged one is documented in larger models by [McGrath et al. (2023)](https://arxiv.org/abs/2307.15771)<sup>[[9]](#ref-9)</sup>, who call it the hydra effect, and backup heads in GPT-2 small by Wang et al.<sup>[[6]](#ref-6)</sup> We have not seen it reported for this MLP and this head. The practical point is about method. Had we measured L4H11 only by its total effect or by zero-ablation, we would have ranked it as a minor head. Path patching was needed to see that it is the main source of the keys.

### Previous-Token Scores

Experiment 04 scores every head on attention from each position to the one before it, on the repeated random sequences and, as a check that the score does not depend on random tokens, on a short passage of English.

| Head | On random tokens | On English |
| --- | --- | --- |
| L4H11 | 0.987 | 0.999 |
| L3H7 | 0.544 | 0.358 |
| L2H2 | 0.494 | 0.532 |
| L6H8 | 0.480 | 0.188 |
| L5H6 | 0.415 | 0.140 |
| L3H2 | 0.402 | 0.388 |

*Top six of 144 heads by the random-token score.*

L4H11 is a near-perfect previous-token head on both kinds of input. Everything else is partial. The strongest layer 0 head is L0H7, at 0.185 on random tokens and 0.264 on English, so no layer 0 head plays the role the two-layer story assigns. L2H2 is the head Wang et al.<sup>[[6]](#ref-6)</sup> name alongside L4H11 as a previous-token head, and it shows up here as the second strongest on English.

### Composition Scores

Composition scores ask the same question from the weights alone, with no input at all. Following Elhage et al.<sup>[[2]](#ref-2)</sup>, the K-composition score between an upstream head $A$ and an induction head $B$ is the normalized Frobenius norm of $W_{OV}^{A} (W_{QK}^{B})^{\top}$, which measures how much of what $A$ writes lands in the subspace $B$'s keys read. Q- and V-composition are the analogous products with $B$'s query side and $B$'s OV circuit. The baseline is the same score for random matrices of the same shapes, 0.036.

Averaged over the five induction heads, L4H11 has the highest K-composition of any head in layers 0 to 4, at 0.096, against 0.052 for its Q-composition and 0.038 for its V-composition. Its K-composition is between 0.087 and 0.104 with each individual induction head. Across the 60 heads in layers 0 to 4, previous-token score and mean K-composition have a Spearman correlation of 0.78.

The scores are noisy in the way [ARENA's tutorial](https://github.com/callummcdougall/ARENA_3.0)<sup>[[10]](#ref-10)</sup> and Elhage et al.<sup>[[2]](#ref-2)</sup> warn about. L4H7 has a K-composition of 0.091, almost level with L4H11, yet its previous-token score is 0.177 and its path-patching effect through the induction keys is -0.002. The weights say L4H7 could compose with the induction heads, and the activations say it does not in any way that matters here. That is why we treat path patching as the primary evidence for the edge and composition scores as support.

## The Weights

Experiment 05 asks whether the weights implement induction independent of any input, which is the kind of evidence the first post said a circuit claim needs alongside patching.

### OV Circuits Copy

A head's full OV circuit, embedding through value, output and unembedding, is a map from "the token this head attends to" to "which output logits go up." A copying head raises the logit of the token it attends to. We form this map on a fixed random subset of 2,000 vocabulary tokens and count how often a token's own logit is the largest in its row (top-1) or among the five largest (top-5). Chance is 1 in 2,000 for top-1. We also report the positive-eigenvalue share of Elhage et al.<sup>[[2]](#ref-2)</sup>, the sum of the eigenvalues divided by the sum of their magnitudes, $\sum_i \lambda_i / \sum_i |\lambda_i|$, which is 1 for a pure copying map and -1 for a pure anti-copying one.

The embedding matters. GPT-2 small is known to use its first MLP as an extension of the token embedding, so we compute the circuit twice, once with the raw embedding $W_E$ and once with the effective embedding $W_E + \operatorname{MLP}_0(W_E)$, as McDougall et al. do<sup>[[7]](#ref-7)</sup>.

| Head | Top-1, effective | Top-5, effective | Eigenvalue share | Top-1, raw |
| --- | --- | --- | --- | --- |
| L7H2 | 0.924 | 0.990 | 0.996 | 0.008 |
| L7H10 | 0.892 | 0.965 | 0.995 | 0.021 |
| L6H9 | 0.872 | 0.970 | 0.997 | 0.047 |
| L5H1 | 0.558 | 0.791 | 0.987 | 0.004 |
| L5H5 | 0.155 | 0.353 | 0.953 | 0.058 |

*Computed on the same random subset of 2,000 vocabulary tokens.*

With the effective embedding, L7H2, L7H10 and L6H9 map a token to itself for 87% to 92% of tokens, and to its top five for 96% to 99%. Their eigenvalue shares are 0.995 to 0.997. A Gaussian random head matched in norm scores a top-1 rate of 0.0006 and an eigenvalue share of -0.005. With the raw embedding, top-1 rates fall to between 0.004 and 0.058, which is the clearest single sign in this project that MLP0 acts as part of the embedding.

The eigenvalue test does not single out induction heads, and it should not be read as if it did. Among the other 139 heads the 90th percentile of the eigenvalue share is 0.988, and 33 of all 144 heads exceed 0.95. Copying OV circuits are common in the later layers of GPT-2 small. What distinguishes an induction head is copying combined with the right attention, not copying alone.

### QK Circuits Attend Back

The QK side is tested with L4H11 in the loop. For each query token $t$ we score every key position whose previous token is one of the 2,000 subset tokens, with the key built from L4H11's output on that previous token, and ask whether the highest-scoring key is the one whose previous token is $t$. For all five heads it is, for 98.9% to 99.8% of tokens (L5H5 0.994, L6H9 0.998, L5H1 0.998, L7H10 0.989, L7H2 0.989). As a control, the same computation with plain token keys, no previous-token head, picks the matching key for exactly 0.0% of tokens in every head. The attend-back rule is in the weights, and it only exists once the previous-token head writes into the keys.

### Direct Logit Attribution

Direct logit attribution asks how much each head's output, passed straight through the final LayerNorm scale and the unembedding, raises the logit of the correct next token.

<figure data-figure="chart:tiny-circuits/direct-logit"></figure>

On second-half positions the five heads add L7H2 +1.51, L7H10 +1.16, L6H9 +1.08, L5H1 +0.69 and L5H5 +0.30. On first-half positions, where there is nothing to copy, all five contribute between -0.001 and +0.003. Together the five write 4.74 of the 13.37 that all 144 heads write directly, against a correct-token logit of 21.49 and a mean logit of about 0. L9H9 (+1.42), L9H6 (+1.41) and L10H0 (+1.17) each write as much as a canonical head.

The copy suppression heads show up from the other side. L10H7 writes -1.21 and L11H10 writes -0.77, the two most negative heads in the model, and their OV eigenvalue shares are -0.999 and -0.996, which is almost perfect anti-copying. Their weights do exactly what the patching result implied.

### The L5H5 Surprise

The next surprise is L5H5. It has the highest induction score in the model (0.925) and attend-back weights as sharp as the others (0.994), and yet it is the weakest copier of the five (top-1 0.155), the smallest direct writer (+0.30), and the smallest single-head patching effect (0.031 recovered). Its eigenvalue share of 0.953 says its OV circuit is broadly positive, just not sharply diagonal.

One reading is that L5H5 matters mostly indirectly, by writing something later heads read, rather than by writing the answer itself. An early induction head whose output is consumed by later induction or copying heads would look exactly like this. We have not tested that reading. Path patching from L5H5 into later heads would, and it is the obvious next experiment. The broader lesson is that rank by induction score, which measures attention placement, does not predict rank by causal effect, copying or direct write.

## Necessity

Experiment 06 removes heads and measures what is lost. Zero ablation sets a head's output to 0. Mean ablation replaces it with its mean over a separate batch of 64 repeated random sequences, position by position, which removes the token-specific information while keeping the average activation the rest of the model expects. Mean ablation is the cleaner test, and we report it first.

### The Five Heads Together

Mean-ablating all five induction heads moves the ICL score from -12.94 to **-8.68**, and the second-half loss from 0.24 to 4.46 nats. That removes 33% of the ICL score. Zero ablation gives a similar -8.94. Adding L4H11 to the five takes the score to -6.44 under mean ablation, removing half of it.

Twenty random sets of five heads, drawn from the other 139, are the control. Under mean ablation they average -12.86 with a standard deviation of 0.055, and the worst set reaches -12.76. The five induction heads matter far more than five random heads. They are not, on their own, the in-context learning of this model.

### Single Heads

No induction head matters much alone under mean ablation. Each moves the ICL score by at most 0.047 nats (L6H9). L4H11 alone moves it by 0.294, the largest single-head effect in the model, which fits its role as the one head all five induction heads read their keys from.

Zero ablation tells a partly different story, and it is the one number here that disagrees with the patching picture in an informative way. Zero-ablating L5H1 costs 0.63 nats of ICL score, more than any other single head, while mean-ablating it costs 0.011. L5H1 also stood out in patching as the only induction head whose corruption alone loses real performance (0.160). One explanation is that L5H1's output carries a large constant component the rest of the model relies on, which the mean keeps and zero removes. We have not tested it.

### Backup Induction Heads

If the five are a subset of a larger population, ablating further heads in order of induction score should keep eroding the ICL score, and ablating random heads should not.

<figure data-figure="chart:tiny-circuits/backup-heads"></figure>

That is what the curve shows. With the top $k$ heads mean-ablated, the ICL score is -12.03 at $k = 4$, -8.68 at $k = 5$, -5.34 at $k = 10$ and -1.13 at $k = 20$. By $k = 20$ the second-half loss is 11.80 nats against a first-half loss of 12.93, so in-context copying is nearly gone. Random sets of $k$ heads, averaged over ten draws per $k$, stay near -12.7 all the way to $k = 20$.

The steps are informative. Most of the drop at $k = 5$ comes when L7H2 is added, after the first four heads together cost only 0.9 nats, so the four canonical heads above it are well covered by the rest. The curve jumps again at $k = 8$ when L9H9 joins L9H6, and at $k = 13$ when L8H1 is added. It bumps slightly the wrong way at $k = 9$ and $k = 11$, where the added heads are L10H7 and L11H10, the copy suppression heads. Removing a head that works against copying helps copying, which is the ablation result agreeing with patching and with the weights.

### The Knockout Sweep

The last run mean-ablates every head alone. The largest effect by far is L4H11, at +0.294 nats. The next largest are L6H10 and L8H6, at +0.077 each, and 16 heads in total exceed +0.05. The heads whose removal most improves the ICL score are L6H6 (-0.086), L8H10 (-0.070) and L10H7 (-0.063), and all three have strongly negative OV eigenvalue shares in experiment 05 (-0.969, -0.992 and -0.999). The anti-copying heads are the heads whose loss helps.

No induction head appears near the top of the single-head sweep. That is the backup picture in its plainest form. The model has one bottleneck upstream, L4H11, and a redundant population downstream.

## Against the Literature

The table sorts each result by how directly it can be compared with published work.

| Result | Reference | Comparison |
| --- | --- | --- |
| L5H5, L6H9, L5H1, L7H10 and L7H2 are the induction heads | ARENA tutorial names all five, Wang et al. 2022 name L5H5 and L6H9 | Match on the head identities |
| L4H11 is the main previous-token head, L2H2 a weaker one | Wang et al. 2022, ARENA | Match on the head identities |
| The previous-token edge runs through keys, not queries or values | Elhage et al. 2021 | Qualitative match, their model is a two-layer attention-only transformer |
| The previous-token head sits in layer 0 | Elhage et al. 2021, our outline | Does not carry over to GPT-2 small |
| L10H7 and L11H10 push against the copied token | Wang et al. 2022, McDougall et al. 2023 | Match on identities and sign |
| OV circuits of induction heads copy, with positive eigenvalues | Elhage et al. 2021, Olsson et al. 2022 | Qualitative match |
| MLP0 acts as part of the embedding | McDougall et al. 2023 | Match in direction |
| Ablating induction heads removes in-context learning, far more than other heads | Olsson et al. 2022 | Direction matches, magnitude only qualitative |
| Backup heads and self-repair | Wang et al. 2022 in GPT-2 small, McGrath et al. 2023 in larger models | Qualitative match |

What we can match is head identities and signs. Magnitudes mostly cannot be compared, because the published work used other models (Olsson et al. and Elhage et al. study their own small models), other tasks (Wang et al. study indirect-object identification), or other metrics. We know of no published patching map, K-path size or top-$k$ ablation curve for this setup. Every qualitative prediction we could check held, and the magnitudes here are new measurements rather than replications.

## What This Does Not Establish

Four limits apply to everything above.

First, the input is still repeated random tokens. That was the right choice for isolating induction, as the first post argued, and it means nothing here shows what these heads do on English. The one exception is the previous-token score of L4H11, which is 0.999 on a paragraph of English as well.

Second, the patching and ablation runs use one seed and small batches, 8 for patching and 32 for ablation. The random-head control gives a sense of the noise for ablation (a standard deviation of 0.055 nats under mean ablation), and there is no equivalent for patching. The ordering of the large effects is safe. Small differences, such as L9H6 against L9H9, are not.

Third, the mechanism is described for one edge and one layer of heads. We have not traced what reads from L5H5, what lets MLP4 compensate for L4H11, or why zero-ablating L5H1 costs so much more than mean-ablating it. Each is a named gap, not a detail.

Fourth, "the induction circuit" now needs a looser definition than the one the first post started with. A strict circuit, two heads and one edge that are together sufficient and necessary, does not describe GPT-2 small. A redundant population of induction and copying heads, fed by one previous-token head and opposed by a few copy suppression heads, does. That description fits every result here, and it is harder to test to the same standard.

## Instrument Notes

**Numbers Printed but Not Saved with the Results**

Most numbers above are read from the saved result tables. A few are only printed to the console during the runs, and so are recorded in the project's notes rather than in a results table. They are the clean and corrupted log probabilities (-0.22 and -12.36), the joint patching of all five induction heads (0.81 recovered, 0.77 lost), L4H11's effect with MLP4 held clean (0.26) and under zero ablation (0.02), and the random-$k$ ablation curve (-12.92 at $k = 1$, -12.67 at $k = 10$, -12.66 at $k = 20$). To check them we reran experiments 03, 04 and 06 in a separate copy of the project. The reruns reproduced every saved result byte for byte and printed these same values. Saving them alongside the rest is a small change to experiments 03 and 06.

**Why Mean Ablation, and Why It Differs from Zero Ablation**

Zero ablation removes a head's output entirely, including any constant component that downstream layers treat as a bias. That can push the residual stream off the distribution the model was trained on, and the resulting damage says as much about the distribution shift as about the head's function. Mean ablation keeps the average and removes only the input-dependent part, which is the part a copying head is supposed to contribute.

The sweep shows why this matters. Under zero ablation, several layer 0 heads change the ICL score by large amounts in the *helpful* direction, L0H8 by -2.87 nats. That is not a head that works against copying. The ICL score is a difference of two losses, and zeroing an early head raises the first-half loss (the random set that contains L0H6 reaches a first-half loss of 14.80 under zero ablation, against 13.17 intact, while its second-half loss stays at 0.23) more than the second-half loss. We read the zero-ablation sweep for layer 0 as an artifact of the metric and rely on mean ablation throughout.

**The MPS Warning**

TransformerLens<sup>[[11]](#ref-11)</sup> warns that the MPS backend may produce silently incorrect results with the installed PyTorch version. The consistency checks inside these runs argue against a problem here. Layer 0 residual patching returns exactly 1.0 and 0.0, the attention buckets in experiment 02 reproduce the experiment 01 scores to every printed digit, and the top-1 and top-5 mean-ablation rows reproduce the single-head and five-head rows exactly. The reruns above were also on MPS, so they show determinism, not correctness. A CPU rerun would settle it and has not been done.

## Conclusions

The first post found five heads that look in the right place. This one asked whether they do the work, and the answer has more parts than the two-head story allows.

- The five induction heads are causally responsible for in-context copying as a group. Patching all five recovers about 0.81 of clean performance and noising all five loses about 0.77, while the best single head recovers 0.128.
- The information moves from first-half to second-half positions between layers 5 and 8, across exactly the layers the five heads occupy.
- One upstream head, L4H11 in layer 4, feeds the induction keys. Its direct key path carries 0.242 of performance, the next sender carries 0.013, and its query and value paths carry nothing. No layer 0 head plays this role.
- MLP4 partly compensates when L4H11 is corrupted, so L4H11's total effect (0.093) understates its direct role. Path patching was needed to see it.
- The weights implement both halves of induction. The QK circuit with L4H11 keys picks the right key for about 99% of tokens and plain token keys for none. The OV circuits of three heads copy for 87% to 92% of tokens, once MLP0 is counted as part of the embedding.
- Rank by induction score does not predict rank by effect. L5H5 has the top score and is the weakest copier, direct writer and single-head patch of the five.
- Removing the five heads costs 33% of the ICL score, against about 1% for random heads. Removing the top 20 by induction score costs over 90%. GPT-2 small has many backup induction heads, and two copy suppression heads, L10H7 and L11H10, work against all of them.

Find the code at [github.com/SadeekFarhan21/projects/tiny-circuits](https://github.com/SadeekFarhan21/projects/tree/main/tiny-circuits).

## References

1. <span id="ref-1"></span>Alec Radford, Jeffrey Wu, Rewon Child et al. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report, 2019. [link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. <span id="ref-2"></span>Nelson Elhage, Neel Nanda, Catherine Olsson et al. *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread, 2021. [link](https://transformer-circuits.pub/2021/framework/index.html)
3. <span id="ref-3"></span>Catherine Olsson, Nelson Elhage, Neel Nanda et al. *In-context Learning and Induction Heads*. Transformer Circuits Thread, 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
4. <span id="ref-4"></span>Kevin Meng, David Bau, Alex Andonian et al. *Locating and Editing Factual Associations in GPT*. NeurIPS, 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
5. <span id="ref-5"></span>Stefan Heimersheim and Neel Nanda. *How to use and interpret activation patching*. arXiv preprint, 2024. [arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
6. <span id="ref-6"></span>Kevin Wang, Alexandre Variengien, Arthur Conmy et al. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*. ICLR, 2023. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
7. <span id="ref-7"></span>Callum McDougall, Arthur Conmy, Cody Rushing et al. *Copy Suppression: Comprehensively Understanding a Motif in Language Model Attention Heads*. BlackboxNLP Workshop, 2024. [doi:10.18653/v1/2024.blackboxnlp-1.22](https://doi.org/10.18653/v1/2024.blackboxnlp-1.22)
8. <span id="ref-8"></span>Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato et al. *Localizing Model Behavior with Path Patching*. arXiv preprint, 2023. [arXiv:2304.05969](https://arxiv.org/abs/2304.05969)
9. <span id="ref-9"></span>Thomas McGrath, Matthew Rahtz, Janos Kramar et al. *The Hydra Effect: Emergent Self-repair in Language Model Computations*. arXiv preprint, 2023. [arXiv:2307.15771](https://arxiv.org/abs/2307.15771)
10. <span id="ref-10"></span>ARENA. *ARENA course materials, Chapter 1: Transformer Interpretability*. GitHub, 2023. [link](https://github.com/ARENA-education/ARENA_materials)
11. <span id="ref-11"></span>Neel Nanda and Joseph Bloom. *TransformerLens*. GitHub, 2022. [link](https://github.com/TransformerLensOrg/TransformerLens)
