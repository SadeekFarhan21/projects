---
layout: post
title: "Every Line of a Small GPT, Written by Hand"
code: https://github.com/SadeekFarhan21/projects/tree/main/gpt-from-scratch
tags:
  - transformers
  - pytorch
description: "A 5.26M-parameter GPT built from scratch in PyTorch and trained on TinyStories to a cross-entropy of 2.14 on a laptop, where the loss is a receipt that the pipeline works rather than a result."
date: 2026-09-11 02:46:31
---

We trained a decoder-only GPT<sup>[[1]](#ref-1)</sup> written from scratch, with no `nn.Transformer`, no `F.scaled_dot_product_attention` and no HuggingFace model code, to a final cross-entropy of **2.14** on TinyStories<sup>[[2]](#ref-2)</sup>, in 20,000 steps on a single Apple Silicon laptop. The shipped checkpoint is **5,263,848 parameters**, configured as 6 pre-LN blocks<sup>[[3]](#ref-3)</sup>, 8 heads, a 256-dimensional residual stream, a 64-token context, and a 1,000-token byte-level BPE<sup>[[4]](#ref-4)</sup><sup>[[1]](#ref-1)</sup> vocabulary trained on the corpus itself.

The loss is not a result. It is a receipt, evidence that every stage of the pipeline, from tokenizer and cache through loader, model, loss and sampler, is wired correctly end to end, which was the entire goal. The run consumed $20{,}000 \times 8 \times 64 \approx 10.2\text{M}$ tokens against a cached corpus of roughly 632M tokens, so the model saw about 1.6% of the available data in a single pass. It is undertrained by construction, and no comparison to any published loss is meaningful. What the project did produce is a clear view of which lines are load-bearing. Two constants decide silently whether the model trains at all, and two bugs cost more time than the model code did.

*Reading note.* The argument is that the naive implementation makes the mechanism visible, and that the expensive failures were all silent.

## Why It Matters

Reading the transformer paper<sup>[[5]](#ref-5)</sup> and reading a reference implementation both leave the same gap. You can follow every line and still not know which lines are *load-bearing*. Typing it out closes that gap by force. Every constant omitted and every shape gotten wrong produces either a crash or, worse, a model that trains to nothing while looking fine.

## Technical Details

### Attention Is Three Linear Maps and a Mask

The single-head forward pass in full.

```python
keys, queries, values = self.key(x), self.query(x), self.value(x)
scores = queries @ keys.transpose(-2, -1)
scores = scores * (self.head_size ** -0.5)
scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
return F.softmax(scores, dim=-1) @ values
```

Four operations. A token's query is a question, every other token's key is an advertisement, the dot product scores the match, and the value is what actually gets moved. The causal mask is not a deep architectural property. It is a lower-triangular matrix of ones the size of the context, built with `torch.tril` and registered as a buffer, plus a `masked_fill`.

<figure class="excal" data-diagram="gpt-attention"><a href="/img/diagrams/gpt-attention.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/gpt-attention.webp" alt="Inside one SelfAttention head, x passes through self.query, self.key and self.value; queries @ keys.transpose(-2, -1) is scaled by head_size ** -0.5, positions where the lower-triangular self.mask is 0 are set to -inf with masked_fill, F.softmax turns the scores into weights, and weights @ values gives the head's output; MultiHeadAttention runs x through heads[0] through heads[7] (the default 8 heads), joins the results with torch.cat(dim=-1), and passes them through self.proj." width="2400" height="2060" loading="lazy" decoding="async"></a></figure>

### The Scale Factor Is Not a Detail

Multiplying the scores by one over the square root of the head size looks like a detail, and it decides whether the model trains.

Take a query $q$ and key $k$ in $\mathbb{R}^{d}$ with independent, zero-mean, unit-variance entries. Their dot product is a sum of $d$ independent terms, so

$$
\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d} \operatorname{Var}(q_i k_i) = d,
$$

and the logits entering the softmax have standard deviation $\sqrt{d}$. Dividing by $\sqrt{d}$ restores unit variance and keeps the softmax in a regime where it is not saturated<sup>[[5]](#ref-5)</sup>.

Without the scale, logits grow with head size, the softmax collapses toward one-hot, and the gradient through it vanishes, because $\partial \operatorname{softmax}$ is proportional to $p_i(\delta_{ij} - p_j)$, which goes to zero as any $p_i \to 1$. The model still trains, the loss barely moves, and nothing in the code looks wrong. That combination is what makes it dangerous. The failure has no error message and no obviously guilty line.

At $d = 32$ the factor is $1/\sqrt{32} \approx 0.177$, so the unscaled logits would be roughly $\sqrt{32} \approx 5.7$ times larger, enough to saturate and not enough to look absurd if printed.

### The Residual Stream Is the Actual Abstraction

```python
x = x + self.attention(self.layer_norm1(x))
x = x + self.feed_forward(self.layer_norm2(x))
```

Two `x + ...` lines, and they are the reason depth works at all. Each block *proposes an edit* to a running representation rather than replacing it. Normalization goes before the sublayer (pre-LN), so the residual path from input to output is unnormalized and gradients reach layer 0 intact<sup>[[3]](#ref-3)</sup>.

Once the residual stream reads as a shared bus that every layer writes to and reads from<sup>[[7]](#ref-7)</sup>, the later literature of logit lens<sup>[[8]](#ref-8)</sup>, activation patching<sup>[[9]](#ref-9)</sup> and circuits<sup>[[7]](#ref-7)</sup> stops being exotic and starts being the obvious next question. That framing is what sent this project toward induction heads<sup>[[10]](#ref-10)</sup> next.

### Architecture

<figure class="excal" data-diagram="gpt-architecture"><a href="/img/diagrams/gpt-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/gpt-architecture.webp" alt="GPT forward pass as one residual stream: token ids through a 1,000 × 256 token embedding plus a learned 64 × 256 position embedding give x of shape (8, 64, 256), which runs down a spine through six pre-LN blocks, where LayerNorm then 8-head causal self-attention and LayerNorm then a 256 to 1,024 to 256 ReLU MLP each branch off and add back; the spine then goes through a final LayerNorm and a 256 to 1,000 LM head, and the (8, 64, 1,000) logits meet the shifted targets in cross-entropy to give a scalar loss." width="2400" height="3212" loading="lazy" decoding="async"></a></figure>

<figure data-figure="gpt-parameters"></figure>

**Full Architecture, as Configured in the Shipped Checkpoint**

| Component | Value |
| --- | --- |
| Blocks | 6 |
| Attention heads | 8 |
| Residual width | 256 |
| Head size | 32 (residual width / heads) |
| Context | 64 |
| Vocabulary | 1,000 (byte-level BPE) |
| MLP | $4\times$ expansion, ReLU |
| Position encoding | Learned embeddings |
| Normalization | Pre-LN, two `LayerNorm`s<sup>[[6]](#ref-6)</sup> per block |
| Parameters | 5,263,848 |

The parameter count breaks down as 272,384 in the embeddings (256,000 token + 16,384 positional), 4,733,952 across the six blocks, 512 in the final `LayerNorm`, and 257,000 in the unembedding. The causal masks are registered buffers and are not counted, which is why the usual count over the model's parameters excludes the 196,608 mask entries in the state dict.

<figure class="excal" data-diagram="gpt-pipeline"><a href="/img/diagrams/gpt-pipeline.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/gpt-pipeline.webp" alt="The pipeline runs top to bottom: TinyStories text trains a 1,000-token byte-level BPE tokenizer.json, and tokenize_file() streams the text into a uint16 token cache that is rebuilt only when it is older than the text or the tokenizer. LanguageModelDataset memory-maps that cache and feeds (x, y) batches to the train.py loop (GPT forward, cross-entropy, backward, AdamW step), which writes gpt-tinystories.pt every 1,000 steps for generate.py and chat.py to load. Separately, launch_modal.py spawns modal_app.py, which runs the same train.py on an L4 GPU with a prebuilt token cache and saves its own checkpoint to the Modal volume every 500 steps." width="2400" height="4014" loading="lazy" decoding="async"></a></figure>

## Implementation

The project covers the whole path: a tokenizer, a streaming data pipeline, the model, a training loop, a sampler, a chat REPL, and a harness for the GPU runs on Modal.

### Multi-Head, Implemented the Slow Way on Purpose

```python
self.heads = nn.ModuleList([SelfAttention(...) for _ in range(num_heads)])
return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))
```

Production implementations fold all heads into one batched matmul. This one keeps them as independent modules and concatenates. That is measurably slower, and we kept it. Heads genuinely are independent subspaces, and writing them as separate objects makes that structural fact impossible to forget. Fusing is an optimization, not a concept, and the independence is exactly the property that later interpretability work depends on. The [induction-circuit project](/posts/reverse-engineering-gpt-2s-induction-circuit/) scores heads individually, which only means something because they *are* individual.

At 3.8M–5.3M parameters on a laptop the trade is free. At any serious scale it is not, and the right move is to fuse and keep a slow reference implementation for testing against.

### The Tokenizer and the Data Pipeline

The least interesting part took the most iterations.

Byte-level BPE<sup>[[4]](#ref-4)</sup><sup>[[1]](#ref-1)</sup> with a ByteLevel pre-tokenizer and decoder means **no unknown tokens are possible**, since every byte sequence encodes. That sounds like a footnote and is what makes the model robust to whatever the corpus contains. The vocabulary is trained on the corpus itself rather than borrowed, which at 1,000 merges over children's stories gives a tokenizer specialized to exactly this distribution.

**Streaming Tokenization and Cache Invalidation**

The corpus is 1.9 GB of text and stopped fitting comfortably in memory. Tokenization became a streaming pass that writes a `uint16` token cache to disk, invalidated by comparing modification times against both the source text and the tokenizer.

```python
cache_is_stale = not token_cache.exists() or token_cache.stat().st_mtime_ns < max(
    args.data.stat().st_mtime_ns,
    tokenizer_path.stat().st_mtime_ns,
)
```

`uint16` because a 1,000-token vocabulary fits in 16 bits, which halves the cache versus `int32`. The resulting cache is 1.26 GB, or roughly 632M tokens.

The staleness check exists because the tokenizer was retrained once, and an hour of training then ran against tokens from the *previous* vocabulary, a corruption that produces no error, just a model learning a scrambled language.

### Training, and How Much Data the Run Actually Saw

AdamW<sup>[[11]](#ref-11)</sup> at its default weight decay of 0.01, learning rate $3 \times 10^{-4}$, batch size 8, context 64, seed 1337, 20,000 steps, checkpointing every 1,000. Checkpoints store the model state, optimizer state, the full config, the training arguments, and the final loss, so a resume restores the run rather than approximating it.

No learning-rate schedule, no warmup, no gradient clipping. At this scale none were necessary, and leaving them out kept the loop readable. At larger scale each is the next thing to add.

The number worth stating plainly is the token budget.

$$
20{,}000 \text{ steps} \times 8 \text{ sequences} \times 64 \text{ tokens} = 10.24\text{M tokens},
$$

against a cache of roughly 632M. The run therefore saw about **1.6%** of the corpus, once. The final loss of 2.14 is a number from a model that is data-starved by design, and the correct reading of it is "the pipeline learns" rather than "the model is good." Anyone comparing it to a published TinyStories loss is comparing against a different experiment.

### Sampling

Greedy decoding on a model this size produces loops almost immediately<sup>[[12]](#ref-12)</sup>. Temperature and top-$k$<sup>[[13]](#ref-13)</sup> were about thirty lines and changed the output more than any architectural change we made.

```python
logits = logits[:, -1, :] / temperature
if top_k is not None:
    cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
    logits = logits.masked_fill(logits < cutoff, float("-inf"))
next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
```

The thing worth internalizing is that the model is a distribution, not a speaker. Every "the model said X" is really "this decoding strategy, at this temperature, said X." Two of the qualitative judgments we made early about model quality turned out to be judgments about decoding.

### Scaling to a GPU, and What It Bought

Once the laptop run worked, training moved to a Modal L4 with a larger configuration of 8 layers, 320-dimensional residual, 128-token context, mixed precision, and an auto batch-size search that doubles the batch until CUDA runs out of memory.

The observation worth recording is how little of that work was model code. The same training script runs unchanged on CPU, Apple's MPS backend and CUDA, picking the device automatically. What the GPU run required was infrastructure.

**The Supervising Loop**

The GPU run needed three things: a persistent volume so a preempted container does not lose the checkpoint, a resume path that points at the same checkpoint file the run writes to, and a supervising loop that commits the volume every 60 seconds.

```python
while True:
    try:
        return_code = process.wait(timeout=60)
        ...
    except subprocess.TimeoutExpired:
        storage.commit()
```

Six hours of GPU time is worth nothing if the container dies at hour five with the checkpoint in a tmpfs.

The model itself is 182 lines. The tokenizer, data pipeline, training loop, sampler, chat REPL, and Modal harness together are roughly four times that, and that is where all of the bugs were. The transformer was the easy half.

## Problems

| Problem | Symptom | Cause |
| --- | --- | --- |
| Stale token cache | Model trains, output is gibberish, loss looks plausible | Tokenizer retrained without invalidating the cached token file |
| Dataset target alignment | Off-by-one, loss plateaus higher than expected | Dataset pre-shifts targets; the loss shifts them again |

**Both are silent. Neither raises.** Both are the same category, **a pipeline that is internally consistent and describing the wrong thing.** That category is, we now believe, the dominant failure mode in small-scale ML work, and it is why the receipt matters more than the number on it.

## Results

Had we reported only the final loss, this post would have described a trained model. **The accurate version is that it describes a *validated pipeline* with a model attached that has seen 1.6% of its corpus once.**

- **The naive version is the point.** Every optimization skipped is a concept left visible. Unfused multi-head attention costs throughput and buys understanding, and at this scale that trade is free.
- **The scale factor and the mask are not details.** $1/\sqrt{d}$ is the difference between a model that trains and one that silently does not, and the failure has no error message.
- **Most of the code is not the model.** 182 lines of model against roughly four times that in everything else, and the bugs were all in the everything else.
- **Silent correctness bugs dominate.** Nothing in this project crashed in an informative way. Both expensive bugs produced a running system quietly learning the wrong function.
- **The loss number is not comparable to anything.** One pass over 1.6% of the data is not a training run anyone should benchmark against.

## References

1. <span id="ref-1"></span>Alec Radford, Jeffrey Wu, Rewon Child, et al. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report, 2019. [link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. <span id="ref-2"></span>Ronen Eldan, Yuanzhi Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv preprint, 2023. [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
3. <span id="ref-3"></span>Ruibin Xiong, Yunchang Yang, Di He, et al. *On Layer Normalization in the Transformer Architecture*. ICML, 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
4. <span id="ref-4"></span>Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units*. ACL, 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
5. <span id="ref-5"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. *Attention Is All You Need*. NeurIPS, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
6. <span id="ref-6"></span>Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. *Layer Normalization*. arXiv preprint, 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
7. <span id="ref-7"></span>Nelson Elhage, Neel Nanda, Catherine Olsson, et al. *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread, 2021. [link](https://transformer-circuits.pub/2021/framework/index.html)
8. <span id="ref-8"></span>nostalgebraist. *interpreting GPT: the logit lens*. LessWrong, 2020. [link](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
9. <span id="ref-9"></span>Stefan Heimersheim, Neel Nanda. *How to use and interpret activation patching*. arXiv preprint, 2024. [arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
10. <span id="ref-10"></span>Catherine Olsson, Nelson Elhage, Neel Nanda, et al. *In-context Learning and Induction Heads*. Transformer Circuits Thread, 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
11. <span id="ref-11"></span>Ilya Loshchilov, Frank Hutter. *Decoupled Weight Decay Regularization*. ICLR, 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
12. <span id="ref-12"></span>Ari Holtzman, Jan Buys, Li Du, et al. *The Curious Case of Neural Text Degeneration*. ICLR, 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
13. <span id="ref-13"></span>Angela Fan, Mike Lewis, Yann Dauphin. *Hierarchical Neural Story Generation*. ACL, 2018. [link](https://aclanthology.org/P18-1082/)
