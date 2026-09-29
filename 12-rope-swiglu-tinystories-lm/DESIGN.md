# Design

This document describes how v0 is put together: the components, the data that flows between them, the invariants the tests pin down, and the trade-offs I took and rejected.

## Components and data flow

```
                         OFFLINE (scripts/)                                   ONLINE (src/tfs/)
 ┌──────────────────────────────┐
 │ download_tinystories.py      │  HTTP Range: first 200 MB of train + full valid file
 │   data/train.txt, valid.txt  │
 └──────────────┬───────────────┘
                │ text, stories separated by <|endoftext|>
                v
 ┌──────────────────────────────┐   ┌───────────────────────────────────────────────┐
 │ prepare_data.py              │   │ bpe.py  BPETokenizer                          │
 │  1. train BPE on first 20 MB ├──>│  train(): chunk counts -> pair counts +       │
 │  2. encode both splits       │   │           inverted index -> lazy max-heap     │
 │     (process pool, 2000      │   │  encode(): split specials -> regex chunks ->  │
 │      stories per shard)      │   │           bytes -> merges by rank (cached)    │
 └──────────────┬───────────────┘   │  decode(): ids -> bytes -> utf-8              │
                │                   └───────────────────────────────────────────────┘
                │ data/{train,valid}.bin   flat uint16 ids, EOT after every story
                v
 ┌──────────────────────────────┐        (B, T) int64 windows, targets = inputs shifted by 1
 │ data.py  TokenDataset        ├─────────────────────────────┐
 │  np.memmap, random windows   │                             │
 └──────────────────────────────┘                             v
                                        ┌────────────────────────────────────────────────┐
                                        │ model.py  Transformer                          │
                                        │                                                │
                                        │  ids ──> Embedding (V x C, tied with lm_head)  │
                                        │            │ x: (B, T, C)                      │
                                        │            v                                   │
                                        │  ┌──── Block x N_layers ─────────────────────┐ │
                                        │  │ x = x + Attn(RMSNorm(x))                  │ │
                                        │  │        qkv Linear -> (B,H,T,D) x 3        │ │
                                        │  │        RoPE(q), RoPE(k) at pos start..    │ │
                                        │  │        KVCache.update (if decoding) ─────────┐
                                        │  │        softmax(q k^T / sqrt(D) + mask) v  │ │ │
                                        │  │        out Linear                         │ │ │
                                        │  │ x = x + SwiGLU(RMSNorm(x))                │ │ │
                                        │  │        W2(silu(W1 x) * W3 x)              │ │ │
                                        │  └───────────────────────────────────────────┘ │ │
                                        │            v                                   │ │
                                        │  RMSNorm -> lm_head -> logits (B, T, V)        │ │
                                        └───────┬───────────────────────┬────────────────┘ │
                                                │                       │                  │
                      ┌─────────────────────────v───────┐   ┌───────────v──────────────┐   │
                      │ train.py                        │   │ sampling.py              │   │
                      │  bf16 autocast forward          │   │  prefill prompt          │   │
                      │  CE loss in fp32                │   │  loop: filter_logits     │   │
                      │  backward, clip grad norm 1.0   │   │   (temp, top-k, top-p)   │   │
                      │  AdamW (decay on 2-D only)      │   │   sample -> feed 1 token │<──┘
                      │  warmup + cosine LR             │   │  KVCache k/v per layer   │
                      │  fixed-batch validation         │   └──────────────────────────┘
                      │  log.csv, ckpt.pt, summary.json │
                      └─────────────────────────────────┘
```

Default v0 model: vocab 4096, d_model 384, 6 layers, 6 heads (head dim 64), SwiGLU hidden 1024, context 256, weight tying. That is 12,194,688 parameters.

## Key data structures

### BPETokenizer (`src/tfs/bpe.py`)
- `merges: list[tuple[int, int]]`. Merge `i` creates token id `256 + i`. Its position in the list is its rank.
- `ranks: dict[pair, int]`, the inverse of `merges`, used by the encoder to pick the earliest-learned pair.
- `vocab: dict[int, bytes]`. Ids 0 to 255 are raw bytes; every merged token is the concatenation of its two parents' bytes.
- `special_tokens: dict[str, int]`, assigned after all merges. v0 has one: `<|endoftext|>` = 4095.
- `_cache: dict[str, list[int]]` memoizes chunk encodings. TinyStories has a small vocabulary of words, so almost every chunk hits the cache after the first few thousand stories.
- On disk it is JSON: the merge list, the special tokens, and the pre-tokenizer regex. Loading refuses a file trained with a different regex.

### Training-time structures inside `BPETokenizer.train`
- `words` (unique pre-token chunks as id lists) and `freqs` (their counts). BPE only ever needs unique chunks.
- `pair_counts[pair]`, the live frequency of every adjacent pair, weighted by chunk frequency.
- `where[pair]`, the set of word indices that contain the pair. A merge only rewrites those words.
- A heap of `(-count, pair)` with lazy invalidation. Popped entries whose count no longer matches `pair_counts` are re-pushed with the live count.

### Token files (`data/*.bin`)
Flat `uint16` arrays, read with `np.memmap`. Every story is followed by the EOT id, so a random window can span a story boundary and the model learns to end stories and start new ones.

### KVCache (`src/tfs/model.py`)
- `k[layer], v[layer]`: preallocated `(B, H, max_len, D)` tensors, one pair per layer.
- `pos`: the number of positions already written, shared by all layers.

## Invariants (and the tests that hold them)

1. `decode(encode(s)) == s` for every Python string, including empty strings, whitespace runs, underscores, combining marks, emoji and literal `<|endoftext|>`. Tested in `tests/test_bpe.py` with a list of tricky strings plus 300 random strings.
2. The pre-tokenizer regex covers every character: `"".join(pretokenize(s)) == s`. This is the invariant that broke first (see DEVLOG, Problems).
3. Training is deterministic: ties in pair count break on the smaller pair.
4. Causality: changing tokens at positions `>= t` never changes logits at positions `< t` (`test_causality_future_tokens_do_not_change_past_logits`).
5. KV cache slots `[0, pos)` hold rotated keys and values for the first `pos` tokens. `pos` advances once per forward call, after every layer has written, so every layer sees the same offset. A query at absolute position `p` may attend to keys `0..p`, which is exactly `_causal_mask(T, S, start)`.
6. Cached decoding is an optimization only. In float64 the cached and uncached per-step logits agree to 1e-10 and the tokens are identical, for both the hand-written and SDPA attention, and for uneven prefill chunks (`tests/test_kv_cache.py`). In float32 they agree to 1e-5 on CPU and 1e-4 on MPS.
7. Hand-written attention equals `F.scaled_dot_product_attention` to 1e-10 in float64.
8. RoPE preserves vector norms, is the identity at position 0, and makes `<R(m)q, R(n)k>` depend only on `m - n`. Point 5 relies on this: keys are rotated once, when written to the cache.
9. Reductions (RMSNorm, attention softmax, cross-entropy) run in at least fp32 under autocast, and never downcast fp64.

## Trade-offs

### Chosen
- **Byte-level BPE with a GPT-2 style regex, written in plain Python.** Every string is encodable and there is no unknown token. Training on 20 MB takes 6.5 s thanks to the inverted index and heap. Encoding is the slow part (about 4.5 MB/s through a process pool), which is fine for a one-off 200 MB preprocessing step.
- **Vocab 4096.** TinyStories has a tiny vocabulary. 4096 tokens gets 4.03 bytes per token (`results/tokenizer_stats.json`) and keeps the embedding at 1.6M of the 12.2M parameters. It also fits in uint16.
- **RoPE in the rotate-half layout with real cos/sin tables.** Complex-number RoPE is shorter but complex tensor support on MPS is patchy. Rotate-half is two slices and a concat.
- **Pre-norm with RMSNorm.** Pre-norm keeps an identity path through the residual stream, so training is stable without careful warmup tuning. RMSNorm drops the mean subtraction and bias of LayerNorm, which the Llama line showed costs nothing in quality.
- **SwiGLU with hidden size 8/3 of d_model rounded to 64.** Same parameter count as a 4x GELU MLP. W1 and W3 are fused into one matmul.
- **Hand-written attention by default, SDPA kept as a reference.** The brief asked for attention written by hand. Keeping `attn_impl="sdpa"` gives a correctness oracle for the tests and a performance baseline for the benchmark.
- **Preallocated KV cache.** Growing the cache with `torch.cat` each step reallocates every layer's cache on every token. Preallocation writes in place and lets the cache be sized to exactly prompt plus new tokens.
- **bf16 autocast on MPS for training.** Measured 20.2k vs 13.8k tokens/s against fp32 with hand-written attention (`results/train_throughput.csv`). fp16 was slightly faster (21.9k) but needs a GradScaler to avoid underflow in gradients; bf16 has fp32's exponent range and needs none.
- **AdamW with betas (0.9, 0.95), decay 0.1 on 2-D tensors only, warmup then cosine to 10% of peak, clip at 1.0.** Standard small-GPT recipe. The schedule is a pure function `lr_at(step)` so it is trivially testable.
- **Random contiguous windows instead of epoch-ordered batches.** Simpler and statistically fine at under one epoch.
- **Fixed validation batches.** The same 40 x 32 x 256 tokens every eval, so the curve is not noisy from resampling.

### Rejected
- **Learned absolute position embeddings** (what `../gpt-from-scratch` used). They tie the model to its training length and do not compose with a cache offset as cleanly as RoPE.
- **A per-head `nn.ModuleList`** (also from the earlier project). One fused `qkv` projection and batched heads is one matmul instead of 3 x H small ones.
- **The HF `tokenizers` library.** Not allowed by the brief, and writing BPE is the point.
- **`torch.compile`.** Its MPS backend is still maturing and it would hide the kernels I want to study later in project 13.
- **fp16 with loss scaling.** Marginal speedup over bf16 for more moving parts.
- **Sliding-window generation past the context length.** With RoPE and a cache it needs either cache eviction plus re-rotation or recomputation. v0 simply refuses to generate past `max_seq_len` and says so.
- **Dropout.** Under one epoch of data, the model is not in an overfitting regime.
