import math

import torch

from tfs.sampling import SamplingConfig, filter_logits, sample_next


def probs(p):
    return torch.log(torch.tensor([p], dtype=torch.float32))


def kept(logits):
    return set(torch.isfinite(logits[0]).nonzero().flatten().tolist())


def test_top_k_keeps_k_largest():
    logits = probs([0.1, 0.4, 0.2, 0.3])
    assert kept(filter_logits(logits, SamplingConfig(top_k=2))) == {1, 3}


def test_top_p_keeps_smallest_prefix_reaching_p():
    logits = probs([0.1, 0.4, 0.2, 0.3])
    # sorted: 0.4, 0.3, 0.2, 0.1 -> cumulative 0.4, 0.7, 0.9, 1.0
    assert kept(filter_logits(logits, SamplingConfig(top_p=0.5))) == {1, 3}
    assert kept(filter_logits(logits, SamplingConfig(top_p=0.7))) == {1, 3}
    assert kept(filter_logits(logits, SamplingConfig(top_p=0.71))) == {1, 3, 2}
    # tiny p still keeps the argmax
    assert kept(filter_logits(logits, SamplingConfig(top_p=1e-6))) == {1}


def test_temperature_scales_logits():
    logits = torch.tensor([[1.0, 2.0, 4.0]])
    out = filter_logits(logits, SamplingConfig(temperature=2.0))
    torch.testing.assert_close(out, logits / 2)


def test_greedy_and_top_k_1_are_argmax():
    logits = torch.randn(5, 50)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample_next(logits, SamplingConfig(temperature=0)), logits.argmax(-1, keepdim=True))
    assert torch.equal(sample_next(logits, SamplingConfig(top_k=1), g), logits.argmax(-1, keepdim=True))


def test_sampling_frequencies_match_distribution():
    p = [0.5, 0.3, 0.2]
    logits = probs(p).repeat(20000, 1)
    g = torch.Generator().manual_seed(0)
    draws = sample_next(logits, SamplingConfig(temperature=1.0), g).flatten()
    freq = torch.bincount(draws, minlength=3).float() / draws.numel()
    for f, q in zip(freq.tolist(), p):
        assert math.isclose(f, q, abs_tol=0.015)
