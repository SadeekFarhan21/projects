import csv
import json
import math

import numpy as np
import pytest
import torch

from tfs.bpe import EOT, BPETokenizer
from tfs.data import TokenDataset
from tfs.model import ModelConfig, Transformer
from tfs.train import lr_at, main, make_optimizer


def test_lr_schedule_shape():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup=10, total=110)
    assert lr_at(0, **kw) == pytest.approx(1e-4)  # (0 + 1) / 10 of peak
    assert lr_at(9, **kw) == pytest.approx(1e-3)  # end of warmup hits peak
    assert lr_at(10, **kw) == pytest.approx(1e-3)  # cosine starts at peak
    assert lr_at(60, **kw) == pytest.approx(5.5e-4)  # halfway: mean of peak and floor
    assert lr_at(110, **kw) == pytest.approx(1e-4)
    assert lr_at(500, **kw) == pytest.approx(1e-4)
    lrs = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # monotone decay


def test_weight_decay_only_on_matrices():
    m = Transformer(ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=8))
    opt = make_optimizer(m, 1e-3, 0.1)
    decay, no_decay = opt.param_groups
    assert all(p.dim() == 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() == 1 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == m.num_params()  # tied weight appears once


def test_batches_are_shifted_windows(tmp_path):
    arr = np.arange(1000, dtype=np.uint16)
    arr.tofile(tmp_path / "t.bin")
    ds = TokenDataset(tmp_path / "t.bin")
    x, y = ds.get_batch(8, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_training_cli_end_to_end_on_tiny_corpus(tmp_path):
    # Build a tiny tokenizer + token files and train for a few steps on CPU.
    text = ("Once upon a time there was a cat. The cat liked milk." + EOT) * 200
    tok = BPETokenizer.train(text.split(EOT), 300, [EOT])
    tok.save(tmp_path / "tokenizer.json")
    ids = np.array(tok.encode(text), dtype=np.uint16)
    ids.tofile(tmp_path / "train.bin")
    ids.tofile(tmp_path / "valid.bin")
    out = tmp_path / "run"
    main([
        "--data-dir", str(tmp_path), "--out-dir", str(out), "--device", "cpu", "--amp", "none",
        "--d-model", "32", "--n-layers", "2", "--n-heads", "2", "--seq-len", "32",
        "--batch-size", "8", "--steps", "60", "--warmup", "5", "--eval-every", "30",
        "--eval-batches", "2", "--lr", "3e-3",
    ])
    rows = list(csv.DictReader((out / "log.csv").open()))
    assert [int(r["step"]) for r in rows] == [0, 30, 60]
    vals = [float(r["val_loss"]) for r in rows]
    assert vals[0] == pytest.approx(math.log(tok.vocab_size), abs=0.3)
    assert vals[-1] < vals[0] - 2.0  # a repetitive corpus is easy to learn
    summary = json.loads((out / "summary.json").read_text())
    assert summary["steps"] == 60 and (out / "ckpt.pt").exists()


def test_checkpoint_roundtrip(tmp_path):
    from tfs.generate import load_model

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    m = Transformer(cfg).eval()
    torch.save({"model": m.state_dict(), "config": cfg.to_dict(), "step": 0}, tmp_path / "c.pt")
    m2 = load_model(tmp_path / "c.pt", torch.device("cpu"))
    idx = torch.randint(0, 50, (2, 8))
    assert torch.equal(m(idx), m2(idx))
    assert m2.lm_head.weight.data_ptr() == m2.embed.weight.data_ptr()  # tying survives load
