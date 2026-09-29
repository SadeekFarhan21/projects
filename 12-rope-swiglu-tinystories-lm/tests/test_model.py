import math

import pytest
import torch
import torch.nn.functional as F

from tfs.model import KVCache, ModelConfig, RMSNorm, Transformer, apply_rope, rope_tables

SMALL = dict(vocab_size=97, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)


def make_model(dtype=torch.float32, **kw) -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig(**{**SMALL, **kw})).to(dtype).eval()


def test_output_shape_and_initial_loss():
    m = make_model()
    idx = torch.randint(0, 97, (3, 20))
    assert m(idx).shape == (3, 20, 97)
    # an untrained model should be close to uniform: loss ~= ln(vocab)
    loss = m.loss(idx, torch.randint(0, 97, (3, 20)))
    assert abs(loss.item() - math.log(97)) < 0.3


def test_causality_future_tokens_do_not_change_past_logits():
    m = make_model()
    a = torch.randint(0, 97, (1, 16))
    b = a.clone()
    b[0, 10:] = torch.randint(0, 97, (6,))
    la, lb = m(a), m(b)
    torch.testing.assert_close(la[:, :10], lb[:, :10])
    assert not torch.allclose(la[:, 10:], lb[:, 10:])


def test_manual_attention_matches_sdpa():
    m1 = make_model(torch.float64)
    m2 = make_model(torch.float64, attn_impl="sdpa")
    m2.load_state_dict(m1.state_dict())
    idx = torch.randint(0, 97, (2, 33))
    torch.testing.assert_close(m1(idx), m2(idx), atol=1e-10, rtol=1e-10)


def test_rmsnorm_matches_formula():
    torch.manual_seed(0)
    n = RMSNorm(8, 1e-6)
    n.weight.data = torch.randn(8)
    x = torch.randn(4, 8)
    ref = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-6) * n.weight
    torch.testing.assert_close(n(x), ref)
    if hasattr(F, "rms_norm"):
        torch.testing.assert_close(n(x), F.rms_norm(x, (8,), n.weight, 1e-6))


def test_rope_preserves_norm_and_is_relative():
    D = 16
    cos, sin = rope_tables(D, 64, 10000.0)
    q = torch.randn(1, 1, 1, D, dtype=torch.float64)
    k = torch.randn(1, 1, 1, D, dtype=torch.float64)
    cos, sin = cos.double(), sin.double()

    def rot(x, p):
        return apply_rope(x, cos[p : p + 1], sin[p : p + 1])

    # rotation keeps vector length
    torch.testing.assert_close(rot(q, 7).norm(), q.norm())
    # <R(m) q, R(n) k> depends only on m - n
    d1 = (rot(q, 10) * rot(k, 3)).sum()
    d2 = (rot(q, 27) * rot(k, 20)).sum()
    torch.testing.assert_close(d1, d2)
    # position 0 is the identity
    torch.testing.assert_close(rot(q, 0), q)


def test_weight_tying_and_param_count():
    m = make_model()
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr()
    c = m.cfg
    per_layer = 4 * c.d_model**2 + 3 * c.d_model * c.ffn_hidden + 2 * c.d_model
    expected = c.vocab_size * c.d_model + c.n_layers * per_layer + c.d_model
    assert m.num_params() == expected


def test_too_long_sequence_raises():
    m = make_model()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 65, dtype=torch.long))


def test_kv_cache_overflow_raises():
    m = make_model()
    cache = KVCache(m.cfg, 1, "cpu", torch.float32, max_len=8)
    m(torch.zeros(1, 8, dtype=torch.long), cache)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 1, dtype=torch.long), cache)


def test_gradients_flow_to_every_parameter():
    m = make_model().train()
    idx = torch.randint(0, 97, (2, 12))
    m.loss(idx, torch.randint(0, 97, (2, 12))).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_can_overfit_a_tiny_batch():
    # Sanity check for the whole stack: a small model must memorize one batch.
    m = make_model().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(0, 97, (4, 16))
    tgt = torch.roll(idx, -1, dims=1)
    for _ in range(150):
        opt.zero_grad()
        loss = m.loss(idx, tgt)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
