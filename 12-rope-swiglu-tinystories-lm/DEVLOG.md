# DEVLOG: a transformer from scratch, v0

### What I wanted to build

A language model stack where I wrote every part that matters: the tokenizer, the attention, the position encoding, the normalization, the training loop and the decoder with a KV cache. My earlier project, `../gpt-from-scratch`, got a GPT training on TinyStories, but it leaned on the HF `tokenizers` library, used 2019-era choices (learned position embeddings, LayerNorm, ReLU MLP, one module per head), had no validation loss or LR schedule, and regenerated the whole prefix for every new token.

The goal for v0 was a cleaner successor with the architecture of a modern small Llama-style model, trained on my laptop (M4 Pro, no NVIDIA GPU) to a reported validation loss in well under an hour, plus proof that the parts are correct: tokenizer round-trips, attention that matches a reference, and a KV cache that produces the same logits as recomputation.

### Theory

A decoder-only language model estimates p(x_t | x_<t) and is trained by minimizing the average cross-entropy of the next token. Validation loss in nats per token is the number I report; exp(loss) is the perplexity.

Byte-level BPE starts from the 256 byte values, so any string is representable, then repeatedly merges the most frequent adjacent pair into a new token. A regex first splits text into chunks (words with their leading space, runs of digits, punctuation, whitespace) and merges never cross chunk boundaries. Encoding replays the merges in the order they were learned: at each step, merge the adjacent pair with the lowest rank.

Attention for one head is softmax(Q K^T / sqrt(d) + M) V, where M is 0 where a query may look and minus infinity elsewhere. The causal rule is that the query at absolute position p may see keys 0..p. The 1/sqrt(d) keeps the dot products at unit variance so the softmax does not saturate at initialization.

RoPE encodes position by rotating each pair of query and key dimensions by an angle m * w_i, where m is the position and w_i = 10000^(-2i/d). A rotation by m*w on q and by n*w on k gives a dot product that depends only on m - n. So attention sees relative position, and a key rotated once when it is written to the cache never needs to be touched again.

RMSNorm rescales each vector to unit root-mean-square and multiplies by a learned gain. Pre-norm places it at the input of each sublayer, x + f(norm(x)), which leaves a clean identity path through the residual stream and makes deep stacks train without delicate warmup.

SwiGLU replaces the MLP's single nonlinearity with a gate: W2(silu(W1 x) * W3 x). With three matrices instead of two, the hidden size is cut to about 8/3 of d_model so the parameter count matches a 4x MLP.

The KV cache follows from causality. The keys and values of past tokens do not depend on future tokens, so once computed they can be stored. Without a cache, generating token n costs a forward pass over n tokens, so n tokens cost O(n^2) token-forwards in total. With a cache each step does one token's worth of projection and MLP work plus attention over n cached keys.

AdamW keeps per-parameter first and second moment estimates and applies weight decay directly to the weights instead of through the gradient. Linear warmup avoids large, badly-estimated Adam steps at the start; cosine decay anneals the step size so the model settles. Clipping the global gradient norm at 1.0 caps the damage from a rare bad batch. bf16 autocast runs matmuls in bfloat16, which has the exponent range of fp32, so unlike fp16 it needs no loss scaling.

### Architecture

The full diagram is in `DESIGN.md`. In short, there is an offline half and an online half.

Offline: `download_tinystories.py` fetches the first 200 MB of the TinyStories V2 training file with an HTTP Range request (trimmed back to the last complete story) plus the official 22 MB validation file. `prepare_data.py` trains the tokenizer on the first 20 MB of training text and encodes both splits into flat uint16 files, with an `<|endoftext|>` id after every story.

Online: `TokenDataset` memory-maps a token file and serves random windows of 257 tokens, split into 256 inputs and 256 shifted targets. The `Transformer` is an embedding, 6 pre-norm blocks, a final RMSNorm and an output head tied to the embedding. `train.py` runs the optimizer loop and logs to CSV. `sampling.py` prefills the prompt through the model, then feeds one token at a time, with a `KVCache` object threaded through every attention layer.

The v0 model is d_model 384, 6 layers, 6 heads of 64 dims, SwiGLU hidden 1024, context 256 and vocab 4096, for 12,194,688 parameters.

### Implementation

The tokenizer's training loop is the part where a naive version is too slow. Recomputing all pair counts after every merge is O(merges x corpus). Instead I count unique chunks once, keep a `pair -> count` table and a `pair -> set of word indices` index, and after each merge rewrite only the words that contain the merged pair, subtracting their old pairs and adding their new ones. The best pair comes from a heap of `(-count, pair)` entries with lazy invalidation: when a popped entry's count does not match the live table, it is pushed back with the live count. Ties break on the smaller pair, so training is deterministic. 3,839 merges over 20 MB took 6.5 s (`results/tokenizer_stats.json`). The first merges it learned were " t", "he", " a", " s", " w", "nd", " the"; the longest tokens are whole words like " compassionate" and " uncomfortable".

Encoding memoizes each chunk. TinyStories reuses a small vocabulary, so after a few thousand stories nearly every chunk is a dictionary hit. The corpus is split by story into shards and encoded in a process pool.

In the model, attention computes Q, K and V with one fused `(C, 3C)` projection and reshapes to `(B, H, T, D)`, so all heads run in one batched matmul. RoPE uses the rotate-half layout with cos and sin tables precomputed in float64 and stored as float32 buffers. The causal mask is built from absolute positions (`key_pos <= start + query_index`), which makes the same code path correct for training (start 0), prefill, and single-token decoding with a cache offset. The softmax, RMSNorm reduction and cross-entropy upcast bf16 to fp32 under autocast.

The KV cache preallocates `(B, H, max_len, D)` buffers per layer and writes new keys and values in place at `[pos, pos + T)`. The model advances `pos` once per forward call, after all layers have written. Generation sizes the cache to exactly prompt length plus new tokens.

`F.scaled_dot_product_attention` is available behind `attn_impl="sdpa"` purely as a reference. The tests assert that the hand-written path matches it to 1e-10 in float64.

The training loop is about 150 lines: warmup plus cosine LR computed by a pure function, AdamW with weight decay applied only to 2-D tensors, bf16 autocast, clip at 1.0, evaluation on 40 fixed validation batches every 250 steps, and a wall-clock budget flag. Everything a run produces (config, CSV log, summary, checkpoint) goes into its out-dir.

The test suite has 47 tests across the tokenizer, model, KV cache, sampling and training loop, including an end-to-end CLI run on a tiny synthetic corpus.

### Problems

#### The stdlib regex silently dropped underscores

GPT-2's pre-tokenizer pattern uses `\p{L}` and `\p{N}`, which need the third-party `regex` module. I translated it to the stdlib `re` by spelling letters as `[^\W\d_]` and kept ` ?[^\s\w]+` for punctuation. The first test run failed three tests: `"".join(pretokenize("snake_case"))` came back as `"snakecase"`. Underscore is in `\w`, so the punctuation class excludes it, and my letter class excludes it too, so no alternative matched it and `findall` skipped it. A tokenizer that loses characters still "works" in training and you would only notice in samples. The fix was ` ?(?:[^\s\w]|_)+`, plus a test that the chunks concatenate back to the input for a list of tricky strings, and a 300-string random round-trip test.

#### My fp32 upcast was an fp64 downcast

To keep the softmax and RMSNorm accurate under bf16 autocast I wrote `scores.float()` and `x.float()`. The KV cache and SDPA equivalence tests run in float64 so that any mismatch above 1e-10 means a real bug, and two of them failed with differences around 3e-8. That size is too large for float64 rounding and exactly float32 rounding: `.float()` was casting float64 activations down to float32 in the middle of the network, and the rounding differed between the cached (one query row) and uncached (full matrix) paths. I replaced it with a helper that upcasts only bf16 and fp16, and the equivalence then held at 1e-10. Without the float64 tests I would have accepted a 1e-5 tolerance in float32 and never seen it.

#### A tiny corpus runs out of merges

A tokenizer test that asked for 120 merges on a small repeated corpus got a vocab of 363 instead of 377. There were only about 106 distinct pairs left to merge. Training now stops cleanly when the heap is empty and a test covers that case; the returned vocab can be smaller than requested, and the special token id is always the last id.

#### The validation file starts mid-story

The official `TinyStoriesV2-GPT4-valid.txt` begins with the tail of a story ("u don't have to be scared..."). It is one story out of 27,630 so I left it, but it is why I trim my Range-request slice of the training file back to the last `<|endoftext|>`: a byte range can also cut a UTF-8 character in half, so the slice is decoded with `errors="ignore"` before trimming.

#### Buffered stdout hid training progress

I launched the long run with stdout redirected to a file and saw nothing for minutes, because Python block-buffers stdout when it is not a terminal. The CSV log was flushed on every eval so no data was lost, and the README command now uses `python -u`.

#### A heavily shared machine

Other jobs were running on the same laptop during this session; `uptime` showed 1-minute load averages between 130 and 356 on 14 cores. Every throughput number here is therefore a lower bound and noisier than it would be on an idle machine. The pooled corpus encoder also came out slower than a single process. `results/encode_profile.txt` shows one process with a warm cache encoding 3,000 validation stories in 0.58 s (about 1.0M tokens/s), which extrapolates to roughly 5 s for all 27,630, yet the 14-worker pool took 25.5 s (`results/tokenizer_stats.json`) at about 124% total CPU. Each spawned worker re-imports torch, starts with a cold chunk cache, and competes with everything else running. I did not fix this in v0 because preprocessing is a one-off 80 s step, but the pool is clearly not earning its keep at this data size.

### Experiments

EXPERIMENTS_PLACEHOLDER

### Results

RESULTS_PLACEHOLDER

### What I would change

CHANGE_PLACEHOLDER
