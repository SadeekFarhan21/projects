"""The KV cache must be an optimization only: same logits, same tokens."""

import pytest
import torch

from tfs.model import KVCache, ModelConfig, Transformer
from tfs.sampling import SamplingConfig, generate

CFG = dict(vocab_size=97, d_model=32, n_layers=3, n_heads=4, max_seq_len=64)


def make_model(dtype, attn_impl="manual", device="cpu"):
    torch.manual_seed(0)
    return Transformer(ModelConfig(**CFG, attn_impl=attn_impl)).to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_cached_and_uncached_logits_identical_fp64(attn_impl):
    # In float64 the only differences are reduction-order rounding, far below 1e-10.
    m = make_model(torch.float64, attn_impl)
    prompt = torch.randint(0, 97, (2, 7))
    seq_a, logits_a = generate(m, prompt, 40, use_cache=True, return_logits=True)
    seq_b, logits_b = generate(m, prompt, 40, use_cache=False, return_logits=True)
    assert torch.equal(seq_a, seq_b)
    torch.testing.assert_close(logits_a, logits_b, atol=1e-10, rtol=0)


def test_cached_and_uncached_logits_close_fp32():
    m = make_model(torch.float32)
    prompt = torch.randint(0, 97, (1, 5))
    seq_a, logits_a = generate(m, prompt, 50, use_cache=True, return_logits=True)
    seq_b, logits_b = generate(m, prompt, 50, use_cache=False, return_logits=True)
    assert torch.equal(seq_a, seq_b)
    torch.testing.assert_close(logits_a, logits_b, atol=1e-5, rtol=1e-5)


def test_full_forward_equals_token_by_token_and_chunked_prefill():
    m = make_model(torch.float64)
    idx = torch.randint(0, 97, (2, 30))
    full = m(idx)
    # one token at a time
    cache = KVCache(m.cfg, 2, "cpu", torch.float64)
    steps = torch.cat([m(idx[:, t : t + 1], cache) for t in range(30)], dim=1)
    torch.testing.assert_close(steps, full, atol=1e-10, rtol=0)
    # uneven chunks: exercises the causal mask with a nonzero cache offset
    cache = KVCache(m.cfg, 2, "cpu", torch.float64)
    chunks = [m(idx[:, a:b], cache) for a, b in [(0, 11), (11, 12), (12, 25), (25, 30)]]
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-10, rtol=0)
    assert cache.pos == 30


def test_sampled_generation_same_with_and_without_cache():
    m = make_model(torch.float64)
    prompt = torch.randint(0, 97, (1, 4))
    cfg = SamplingConfig(temperature=0.9, top_k=20, top_p=0.9)
    a = generate(m, prompt, 30, cfg, use_cache=True, generator=torch.Generator().manual_seed(1))
    b = generate(m, prompt, 30, cfg, use_cache=False, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_cache_matches_on_mps_fp32():
    m = make_model(torch.float32, device="mps")
    prompt = torch.randint(0, 97, (1, 5), device="mps")
    seq_a, la = generate(m, prompt, 40, use_cache=True, return_logits=True)
    seq_b, lb = generate(m, prompt, 40, use_cache=False, return_logits=True)
    assert torch.equal(seq_a, seq_b)
    torch.testing.assert_close(la, lb, atol=1e-4, rtol=1e-4)
