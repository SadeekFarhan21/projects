"""Train a tiny character-level language model on Tiny Shakespeare with smolgrad.

Model (Bengio et al. 2003 style, plus LayerNorm):
  16 previous chars -> Embedding(V, 32) -> flatten (512) -> Linear(512, 512)
  -> LayerNorm -> tanh -> Linear(512, V) -> cross-entropy on the next char.

Usage: uv run python scripts/train_char.py [--steps 6000]
Writes results/char_train.csv, results/char_summary.json, results/char_samples.txt,
results/char_curves.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

import smolgrad as sg
import smolgrad.functional as F
import smolgrad.nn as nn
from datasets import RESULTS, load_text


class CharMLP(nn.Module):
    def __init__(self, vocab: int, context: int, emb: int, hidden: int):
        self.context = context
        self.emb = nn.Embedding(vocab, emb)
        self.fc1 = nn.Linear(context * emb, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, idx: np.ndarray) -> sg.Tensor:
        x = self.emb(idx).flatten(1)            # (B, context*emb)
        h = self.ln(self.fc1(x)).tanh()         # (B, hidden)
        return self.fc2(h)                      # (B, vocab) logits


def batches(data: np.ndarray, context: int, bs: int, rng: np.random.Generator):
    # Sliding windows: x = data[i : i+context], y = data[i+context].
    ix = rng.integers(0, len(data) - context - 1, size=bs)
    x = np.stack([data[i:i + context] for i in ix])
    return x, data[ix + context]


def eval_loss(model, data, context, n_batches=40, bs=512, seed=123):
    rng = np.random.default_rng(seed)  # fixed windows so evals are comparable
    tot = 0.0
    with sg.no_grad():
        for _ in range(n_batches):
            x, y = batches(data, context, bs, rng)
            tot += F.cross_entropy(model(x), y).item()
    return tot / n_batches


def bigram_val_loss(train: np.ndarray, val: np.ndarray, V: int) -> float:
    """Count-based bigram model with add-one smoothing, a reference point."""
    counts = np.ones((V, V))
    np.add.at(counts, (train[:-1], train[1:]), 1)
    logp = np.log(counts / counts.sum(1, keepdims=True))
    return float(-logp[val[:-1], val[1:]].mean())


def sample(model, stoi, itos, prompt: str, n: int, context: int, rng, temperature=0.8) -> str:
    ids = [stoi[c] for c in prompt]
    ids = [stoi["\n"]] * max(0, context - len(ids)) + ids
    with sg.no_grad():
        for _ in range(n):
            logits = model(np.array([ids[-context:]])).data[0] / temperature
            p = np.exp(logits - logits.max())
            p /= p.sum()
            ids.append(int(rng.choice(len(p), p=p)))
    return "".join(itos[i] for i in ids).lstrip("\n")  # drop the newline padding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = load_text()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    unigram = np.bincount(train, minlength=V) + 1.0
    unigram_loss = float(-np.log(unigram / unigram.sum())[val].mean())
    bigram_loss = bigram_val_loss(train, val, V)
    print(f"vocab={V} train_chars={len(train):,} val_chars={len(val):,}")
    print(f"baselines on val: uniform={np.log(V):.4f} unigram={unigram_loss:.4f} bigram={bigram_loss:.4f}")

    sg.manual_seed(args.seed)
    model = CharMLP(V, args.context, args.emb, args.hidden)
    opt = sg.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"model: {model.num_parameters():,} parameters")

    rng = np.random.default_rng(args.seed)
    rows, t0, step_time, run = [], time.perf_counter(), 0.0, []
    for step in range(1, args.steps + 1):
        # Linear warmdown over the last 30% of training.
        frac = step / args.steps
        opt.lr = args.lr * (1.0 if frac < 0.7 else (1 - frac) / 0.3 + 0.02)
        x, y = batches(train, args.context, args.batch_size, rng)
        ts = time.perf_counter()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        step_time += time.perf_counter() - ts
        run.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            vl = eval_loss(model, val, args.context)
            row = dict(step=step, train_loss=float(np.mean(run)), val_loss=vl, lr=opt.lr,
                       mean_step_ms=1000 * step_time / step)
            rows.append(row)
            run = []
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)

    samples = [sample(model, stoi, itos, p, 300, args.context, np.random.default_rng(i))
               for i, p in enumerate(["ROMEO:\n", "First Citizen:\n", "KING"])]

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "char_train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "char_samples.txt").write_text(
        "Samples at temperature 0.8 from the final model (scripts/train_char.py)\n\n"
        + "\n\n----------\n\n".join(samples) + "\n")
    summary = dict(config=vars(args), vocab=V, params=model.num_parameters(),
                   final_val_loss=rows[-1]["val_loss"], best_val_loss=min(r["val_loss"] for r in rows),
                   final_train_loss=rows[-1]["train_loss"],
                   baseline_uniform=float(np.log(V)), baseline_unigram=unigram_loss,
                   baseline_bigram=bigram_loss, mean_step_ms=rows[-1]["mean_step_ms"],
                   total_wall_seconds=time.perf_counter() - t0)
    (RESULTS / "char_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(samples[0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    st = [r["step"] for r in rows]
    ax.plot(st, [r["train_loss"] for r in rows], "o-", label="train (running mean)")
    ax.plot(st, [r["val_loss"] for r in rows], "o-", label="val")
    ax.axhline(bigram_loss, color="gray", ls="--", lw=1, label=f"bigram baseline {bigram_loss:.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.legend(frameon=False)
    ax.set_title("Char MLP on Tiny Shakespeare (source: results/char_train.csv)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "char_curves.png", dpi=150)


if __name__ == "__main__":
    main()
