"""Finite-difference gradient checks for every differentiable op.

Each case is (name, function, list of input arrays). Inputs are float64 so the
central-difference error (about eps^2 plus rounding / eps) stays near 1e-9.
"""

import numpy as np
import pytest

import smolgrad as sg
import smolgrad.functional as F
from smolgrad.gradcheck import gradcheck

rng = np.random.default_rng(1234)


def r(*shape):
    return rng.standard_normal(shape)


def pos(*shape):
    return rng.uniform(0.5, 2.0, size=shape)


def away_from_zero(*shape):
    # relu and max are not differentiable at kinks; keep inputs clear of them.
    x = rng.uniform(0.2, 1.5, size=shape)
    return x * rng.choice([-1.0, 1.0], size=shape)


IDX = np.array([[0, 3, 3], [1, 0, 4]])  # repeated rows on purpose
TARGET = np.array([2, 0, 4, 4])

CASES = [
    # binary elementwise, with and without broadcasting
    ("add", lambda a, b: a + b, [r(3, 4), r(3, 4)]),
    ("add_bcast_row", lambda a, b: a + b, [r(3, 4), r(4)]),
    ("add_bcast_col", lambda a, b: a + b, [r(3, 1), r(1, 4)]),
    ("add_bcast_3d", lambda a, b: a + b, [r(2, 3, 4), r(3, 1)]),
    ("add_scalar", lambda a: a + 2.5, [r(3, 4)]),
    ("sub", lambda a, b: a - b, [r(3, 4), r(1, 4)]),
    ("rsub", lambda a: 1.0 - a, [r(3, 4)]),
    ("neg", lambda a: -a, [r(5)]),
    ("mul", lambda a, b: a * b, [r(3, 4), r(3, 4)]),
    ("mul_bcast", lambda a, b: a * b, [r(2, 3, 4), r(4)]),
    ("mul_self", lambda a: a * a, [r(3, 4)]),
    ("div", lambda a, b: a / b, [r(3, 4), pos(3, 4)]),
    ("div_bcast", lambda a, b: a / b, [r(3, 4), pos(3, 1)]),
    ("rdiv", lambda a: 2.0 / a, [pos(3, 4)]),
    ("pow", lambda a: a ** 3, [r(3, 4)]),
    ("pow_neg_half", lambda a: a ** -0.5, [pos(3, 4)]),
    # matmul family
    ("matmul", lambda a, b: a @ b, [r(3, 4), r(4, 5)]),
    ("matmul_batched", lambda a, b: a @ b, [r(2, 3, 4), r(2, 4, 5)]),
    ("matmul_bcast_batch", lambda a, b: a @ b, [r(2, 3, 4), r(4, 5)]),
    ("matmul_vec_right", lambda a, b: a @ b, [r(3, 4), r(4)]),
    ("matmul_vec_left", lambda a, b: a @ b, [r(4), r(4, 5)]),
    ("matmul_dot", lambda a, b: a @ b, [r(4), r(4)]),
    # reductions
    ("sum_all", lambda a: a.sum(), [r(3, 4)]),
    ("sum_axis0", lambda a: a.sum(axis=0), [r(3, 4)]),
    ("sum_axis_neg_keep", lambda a: a.sum(axis=-1, keepdims=True), [r(2, 3, 4)]),
    ("sum_multi_axis", lambda a: a.sum(axis=(0, 2)), [r(2, 3, 4)]),
    ("mean_all", lambda a: a.mean(), [r(3, 4)]),
    ("mean_axis1", lambda a: a.mean(axis=1), [r(2, 3, 4)]),
    ("max_axis", lambda a: a.max(axis=1), [away_from_zero(3, 5)]),
    ("max_all", lambda a: a.max(), [away_from_zero(3, 5)]),
    # shape ops
    ("reshape", lambda a: a.reshape(4, 3) * np.arange(12.0).reshape(4, 3), [r(3, 4)]),
    ("flatten", lambda a: a.flatten(), [r(2, 3, 4)]),
    ("transpose_2d", lambda a: a.T, [r(3, 4)]),
    ("transpose_swap", lambda a: a.transpose(0, 2), [r(2, 3, 4)]),
    ("permute", lambda a: a.permute((2, 0, 1)), [r(2, 3, 4)]),
    ("getitem_slice", lambda a: a[1:, ::2], [r(3, 4)]),
    ("getitem_int", lambda a: a[1], [r(3, 4)]),
    ("getitem_fancy_repeat", lambda a: a[np.array([0, 2, 2, 1])], [r(3, 4)]),
    ("getitem_pairs", lambda a: a[np.arange(4), TARGET], [r(4, 5)]),
    # unary
    ("exp", lambda a: a.exp(), [r(3, 4)]),
    ("log", lambda a: a.log(), [pos(3, 4)]),
    ("sqrt", lambda a: a.sqrt(), [pos(3, 4)]),
    ("relu", lambda a: a.relu(), [away_from_zero(3, 4)]),
    ("tanh", lambda a: a.tanh(), [r(3, 4)]),
    ("sigmoid", lambda a: a.sigmoid(), [r(3, 4)]),
    # softmax family
    ("softmax_last", lambda a: a.softmax(-1), [r(3, 5)]),
    ("softmax_axis0", lambda a: a.softmax(0), [r(3, 5)]),
    ("log_softmax_last", lambda a: a.log_softmax(-1), [r(3, 5)]),
    ("log_softmax_3d", lambda a: a.log_softmax(1), [r(2, 3, 4)]),
    ("softmax_large_logits", lambda a: (a * 50.0).softmax(-1), [r(2, 4)]),
    # fused / functional
    ("embedding", lambda w: F.embedding(w, IDX), [r(5, 3)]),
    ("cross_entropy_mean", lambda x: F.cross_entropy(x, TARGET), [r(4, 5)]),
    ("cross_entropy_sum", lambda x: F.cross_entropy(x, TARGET, reduction="sum"), [r(4, 5)]),
    ("cross_entropy_3d", lambda x: F.cross_entropy(x, TARGET.reshape(2, 2)), [r(2, 2, 5)]),
    ("linear", lambda x, w, b: F.linear(x, w, b), [r(4, 3), r(5, 3), r(5)]),
    ("layer_norm", lambda x, w, b: F.layer_norm(x, w, b), [r(4, 6), r(6), r(6)]),
    # composites that exercise graph reuse (diamonds) and chains
    ("diamond", lambda a: (a * a.exp()).sum(axis=0) + a.tanh().mean(axis=0), [r(3, 4)]),
    ("mlp_2layer", lambda x, w1, w2: ((x @ w1).tanh() @ w2).log_softmax(-1), [r(4, 3), r(3, 6), r(6, 5)]),
]


@pytest.mark.parametrize("name,fn,inputs", CASES, ids=[c[0] for c in CASES])
def test_gradcheck(name, fn, inputs):
    err = gradcheck(fn, inputs, atol=1e-6, rtol=1e-5)
    assert err < 1e-6


def test_gradcheck_catches_a_wrong_gradient():
    """The checker itself must fail on a deliberately broken op."""
    def bad_square(a):
        return sg.Tensor._make(a.data ** 2, (a,), lambda g: (g * a.data,), "bad_square")  # missing 2x

    with pytest.raises(AssertionError):
        gradcheck(bad_square, [r(3)])
