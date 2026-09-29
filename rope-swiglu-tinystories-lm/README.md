# Transformer From Scratch

A small, complete language-model stack written from PyTorch tensor ops: a byte-level BPE tokenizer trained from scratch, a decoder-only transformer (hand-written causal multi-head attention, RoPE, pre-norm RMSNorm, SwiGLU), a training loop (AdamW, warmup plus cosine, gradient clipping, bf16 autocast on Apple MPS), and a sampler with temperature, top-k, top-p and a KV cache. It is trained on a 200 MB slice of TinyStories on an M4 Pro laptop.

This is the successor to `../gpt-from-scratch`, my earlier educational GPT. That project used the HF `tokenizers` library, learned position embeddings, LayerNorm, a ReLU MLP, one module per attention head, no validation loss, no LR schedule and no KV cache. This one replaces all of those and adds tests that pin down the invariants (see `DESIGN.md`).

No `nn.Transformer`, no HF model code, no tokenizer library. PyTorch supplies tensors, autograd, `nn.Linear`, `nn.Embedding` and AdamW.

## Status

| Milestone | Status |
|---|---|
| v0: byte-level BPE tokenizer with round-trip tests | done |
| v0: decoder-only transformer (hand-written attention, RoPE, RMSNorm, SwiGLU) | done |
| v0: training loop (AdamW, warmup + cosine, clipping, bf16 autocast on MPS) | done |
| v0: TinyStories run under an hour with reported validation loss | done |
| v0: sampling (temperature, top-k, top-p) with KV cache, cache equivalence test, tokens/sec benchmark | done |
| Port the model onto the project 11 autograd engine | planned |
| Grouped-query attention (GQA) | planned |
| Flash-attention style tiled attention (online softmax) | planned |
| Scaling-law mini study (loss vs params and tokens) | planned |
| Feed custom kernels from project 13 into the model | planned |

## Headline numbers

All numbers come from runs in this repo on an M4 Pro (48 GB) while many other jobs were running on the same machine, so throughput is a lower bound. Details and caveats are in `DEVLOG.md`.

HEADLINE_PLACEHOLDER

## Layout

```
src/tfs/
  bpe.py        byte-level BPE: train, encode, decode, save, load
  data.py       parallel corpus encoding to uint16, memmapped random-window batches
  model.py      ModelConfig, RMSNorm, RoPE, CausalSelfAttention, SwiGLU, Block, Transformer, KVCache
  sampling.py   temperature / top-k / top-p filtering, generate() with or without the cache
  train.py      training CLI (python -m tfs.train)
  generate.py   sampling CLI (python -m tfs.generate)
scripts/
  download_tinystories.py   fetch a bounded slice of TinyStories V2 into data/
  prepare_data.py           train the tokenizer, encode train and valid splits
  bench_train.py            training throughput: fp32 / bf16 / fp16 x manual / SDPA attention
  bench_generate.py         decode tokens/sec with and without the KV cache
  plot_training.py          loss curve PNG from a log.csv
tests/          pytest suite (tokenizer, model, KV cache, sampling, training loop)
results/        raw outputs of every run reported in DEVLOG.md
```

## Build

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.12.

```bash
uv sync --group dev
```

## Test

```bash
uv run pytest -q
```

The MPS cache test is skipped automatically on machines without an Apple GPU.

## Get data and train

```bash
uv run python scripts/download_tinystories.py --train-mb 200   # about 230 MB into data/ (gitignored)
uv run python scripts/prepare_data.py --vocab-size 4096         # tokenizer.json, train.bin, valid.bin
uv run python -u -m tfs.train --out-dir runs/v0 --steps 4000 --warmup 200 --eval-every 250 --max-minutes 45
cp runs/v0/log.csv results/train_v0_log.csv && uv run python scripts/plot_training.py --log results/train_v0_log.csv
```

`--max-minutes` is a hard wall-clock stop; the run writes `log.csv`, `summary.json` and `ckpt.pt` into its out-dir.

## Benchmark

```bash
uv run python scripts/bench_train.py                           # results/train_throughput.csv
uv run python scripts/bench_generate.py --ckpt runs/v0/ckpt.pt # results/generation_throughput.{csv,png}
```

## Sample

```bash
uv run python -m tfs.generate --ckpt runs/v0/ckpt.pt --prompt "Once upon a time" \
    --temperature 0.8 --top-k 50 --top-p 0.95 --num-samples 3
```

Add `--no-cache` to decode without the KV cache.

## What was cut from v0

CUT_PLACEHOLDER

## Known issues

KNOWN_PLACEHOLDER
