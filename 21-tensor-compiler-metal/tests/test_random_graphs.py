"""Property tests: random graphs on the GPU match the numpy oracle, and the
fused schedule matches the unfused one in f32."""

import numpy as np
from hypothesis import HealthCheck, given, settings

import tcm
from conftest import assert_close, normalized_error
from graphgen import random_graph, trace_random

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def feeds(g, seed):
    rng = np.random.default_rng(seed)
    out = []
    for nid in g.inputs:
        n = g[nid]
        out.append(rng.uniform(-1, 1, n.shape).astype(np.float32 if n.dtype == "f32" else np.float16))
    return out


@SETTINGS
@given(random_graph("f32"))
def test_random_f32_matches_oracle(case):
    specs, program, seed = case
    g = trace_random(specs, program)
    xs = feeds(g, seed)
    got = tcm.compile(g)(*xs)
    for a, b in zip(got, tcm.interpret(g, xs)):
        assert_close(a, b, "f32")


@SETTINGS
@given(random_graph("f16"))
def test_random_f16_matches_oracle(case):
    specs, program, seed = case
    g = trace_random(specs, program)
    xs = feeds(g, seed)
    got = tcm.compile(g)(*xs)
    for a, b in zip(got, tcm.interpret(g, xs)):
        assert_close(a, b, "f16")


@SETTINGS
@given(random_graph("f32"))
def test_fused_equals_unfused_f32(case):
    specs, program, seed = case
    g = trace_random(specs, program)
    xs = feeds(g, seed)
    fused = tcm.compile(g, fuse=True)
    unfused = tcm.compile(g, fuse=False)
    assert fused.num_kernels <= unfused.num_kernels
    for a, b in zip(fused(*xs), unfused(*xs)):
        assert normalized_error(a, b) <= 1e-6


@SETTINGS
@given(random_graph("f32"))
def test_no_buffer_reuse_same_result(case):
    specs, program, seed = case
    g = trace_random(specs, program)
    xs = feeds(g, seed)
    a = tcm.compile(g, fuse=False, reuse_buffers=True)(*xs)
    b = tcm.compile(g, fuse=False, reuse_buffers=False)(*xs)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


from hypothesis import strategies as st


@settings(max_examples=25, deadline=None)
@given(
    st.sampled_from([1, 2, 4]),
    st.sampled_from([3, 16, 37, 64]),
    st.sampled_from([8, 20, 32]),
    st.sampled_from(["f32", "f16"]),
    st.integers(0, 2**31 - 1),
)
def test_random_transformer_blocks(b, t, d, dtype, seed):
    """Attention and MLP blocks with random shapes: matmul + fused row kernels together."""

    def block(x, wq, wk, wv, w1, w2, gam, bet):
        h = tcm.layernorm(x, gam, bet)
        q, k, v = h @ wq, h @ wk, h @ wv
        att = tcm.softmax((q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(d)))
        x = x + att @ v
        return x + tcm.gelu(tcm.layernorm(x) @ w1) @ w2

    specs = [((b, t, d), dtype)] + [((d, d), dtype)] * 3 + [((d, 2 * d), dtype), ((2 * d, d), dtype), ((d,), dtype), ((d,), dtype)]
    g = tcm.trace(block, specs)
    rng = np.random.default_rng(seed)
    npdt = np.float32 if dtype == "f32" else np.float16
    xs = [(rng.standard_normal(s) * (1.0 if i == 0 else 0.2)).astype(npdt) for i, (s, _) in enumerate(specs)]
    fused = tcm.compile(g)
    got = fused(*xs)
    assert_close(got[0], tcm.interpret(g, xs)[0], dtype)
    if dtype == "f32":
        unfused = tcm.compile(g, fuse=False)
        assert normalized_error(got[0], unfused(*xs)[0]) <= 1e-6
        assert fused.num_kernels < unfused.num_kernels
