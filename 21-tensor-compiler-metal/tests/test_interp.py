import numpy as np
import pytest

import tcm
from conftest import rand

UNARY = {
    "neg": np.negative, "exp": np.exp, "tanh": np.tanh, "abs": np.abs,
}


@pytest.mark.parametrize("op", sorted(UNARY))
def test_unary_matches_numpy(op, rng):
    x = rand(rng, (3, 5))
    g = tcm.trace(lambda a: getattr(a, op)() if op != "neg" else -a, [((3, 5), "f32")])
    (y,) = tcm.interpret(g, [x])
    np.testing.assert_allclose(y, UNARY[op](x.astype(np.float64)).astype(np.float32), rtol=1e-6)


def test_composites_match_closed_form(rng):
    x = rand(rng, (4, 16), lo=-3, hi=3)
    (s,) = tcm.interpret(tcm.trace(tcm.softmax, [((4, 16), "f32")]), [x])
    e = np.exp(x - x.max(-1, keepdims=True))
    np.testing.assert_allclose(s, e / e.sum(-1, keepdims=True), rtol=1e-5)
    gm, bt = rand(rng, (16,)), rand(rng, (16,))
    (ln,) = tcm.interpret(tcm.trace(tcm.layernorm, [((4, 16), "f32"), ((16,), "f32"), ((16,), "f32")]), [x, gm, bt])
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    np.testing.assert_allclose(ln, (x - mu) / np.sqrt(var + 1e-5) * gm + bt, rtol=1e-4, atol=1e-5)


def test_matmul_views_reductions(rng):
    a, b = rand(rng, (2, 3, 4)), rand(rng, (2, 5, 4))
    g = tcm.trace(lambda a, b: (a @ b.transpose(0, 2, 1)).max(-1), [((2, 3, 4), "f32"), ((2, 5, 4), "f32")])
    (y,) = tcm.interpret(g, [a, b])
    np.testing.assert_allclose(y, (a @ b.transpose(0, 2, 1)).max(-1, keepdims=True), rtol=1e-5)


def test_f16_rounds_each_node(rng):
    x = rand(rng, (8,), "f16")
    g = tcm.trace(lambda a: a * 3.0 + 1.0, [((8,), "f16")])
    (y,) = tcm.interpret(g, [x])
    assert y.dtype == np.float16
    ref = ((x.astype(np.float32) * np.float16(3.0)).astype(np.float16).astype(np.float32) + 1).astype(np.float16)
    np.testing.assert_array_equal(y, ref)
