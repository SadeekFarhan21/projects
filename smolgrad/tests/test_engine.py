"""Semantics of the autodiff engine itself: accumulation, graph reuse, no_grad,
dtype handling and error cases."""

import numpy as np
import pytest

import smolgrad as sg
from smolgrad import Tensor


def test_scalar_chain_rule():
    x = Tensor(3.0, requires_grad=True, dtype=np.float64)
    y = x * x * x + 2.0 * x  # dy/dx = 3x^2 + 2 = 29
    y.backward()
    assert x.grad == pytest.approx(29.0)


def test_node_used_twice_accumulates():
    # y = a*b + a  -> dy/da = b + 1 ; a feeds two consumers (a diamond)
    a = Tensor(np.array([2.0, -1.0]), requires_grad=True)
    b = Tensor(np.array([5.0, 7.0]), requires_grad=True)
    ((a * b) + a).sum().backward()
    np.testing.assert_allclose(a.grad, [6.0, 8.0])
    np.testing.assert_allclose(b.grad, [2.0, -1.0])


def test_grad_accumulates_across_backward_calls_until_zeroed():
    x = Tensor(np.ones(3), requires_grad=True)
    (x * 2.0).sum().backward()
    (x * 3.0).sum().backward()
    np.testing.assert_allclose(x.grad, [5.0, 5.0, 5.0])
    x.zero_grad()
    assert x.grad is None


def test_intermediate_nodes_do_not_keep_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    h = x * 2.0
    h.sum().backward()
    assert h.grad is None and x.grad is not None


def test_same_graph_backward_twice_is_consistent():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = (x.exp() * x).sum()
    y.backward()
    g1 = x.grad.copy()
    x.zero_grad()
    y.backward()
    np.testing.assert_allclose(x.grad, g1)


def test_leaf_grads_do_not_alias():
    # add passes the same gradient array to both parents; in-place edits to one
    # leaf's .grad must not leak into the other.
    a = Tensor(np.zeros(3), requires_grad=True)
    b = Tensor(np.zeros(3), requires_grad=True)
    (a + b).sum().backward()
    a.grad += 100.0
    np.testing.assert_allclose(b.grad, [1.0, 1.0, 1.0])


def test_no_grad_builds_no_graph():
    x = Tensor(np.ones(3), requires_grad=True)
    with sg.no_grad():
        y = x * 2.0
    assert not y.requires_grad and y._parents == ()
    assert sg.is_grad_enabled()


def test_constants_do_not_require_grad():
    x = Tensor(np.ones(3))
    y = x * 2.0 + 1.0
    assert not y.requires_grad
    with pytest.raises(RuntimeError):
        y.sum().backward()


def test_non_scalar_backward_needs_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()
    (x * 2.0).backward(np.array([1.0, 0.0, 2.0]))
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 4.0])


def test_float32_stays_float32_with_python_scalars():
    x = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    y = (x * 0.1 + 3.0) / 7.0
    assert y.dtype == np.float32
    y.sum().backward()
    assert x.grad.dtype == np.float32


def test_default_dtype_is_float32():
    assert Tensor([1, 2, 3]).dtype == np.float32
    assert Tensor(np.zeros(2, dtype=np.float64)).dtype == np.float64


def test_ndarray_on_left_dispatches_to_tensor():
    x = Tensor(np.ones(3), requires_grad=True)
    y = np.array([1.0, 2.0, 3.0]) * x
    assert isinstance(y, Tensor)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [1.0, 2.0, 3.0])


def test_deep_graph_does_not_recurse():
    # 20k sequential ops would blow Python's default recursion limit (1000)
    # with a recursive topological sort.
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(20_000):
        y = y * 1.0 + 0.0
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_subgraph_not_requiring_grad_is_skipped():
    w = Tensor(np.ones(2), requires_grad=True)
    c = Tensor(np.array([3.0, 4.0]))  # constant branch
    (w * c.exp()).sum().backward()
    assert c.grad is None
    np.testing.assert_allclose(w.grad, np.exp([3.0, 4.0]))


def test_cross_entropy_matches_composed_version():
    rng = np.random.default_rng(0)
    x1 = Tensor(rng.standard_normal((6, 4)), requires_grad=True)
    x2 = Tensor(x1.data.copy(), requires_grad=True)
    t = np.array([0, 3, 1, 1, 2, 0])
    sg.functional.cross_entropy(x1, t).backward()
    (-(x2.log_softmax(-1)[np.arange(6), t]).mean()).backward()
    np.testing.assert_allclose(x1.grad, x2.grad, atol=1e-12)


def test_cross_entropy_is_stable_for_huge_logits():
    x = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    loss = sg.functional.cross_entropy(x, [0])
    loss.backward()
    assert np.isfinite(loss.item()) and loss.item() == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.isfinite(x.grad))


@pytest.mark.parametrize("V,N", [(7, 30), (5000, 3000)])  # one-hot path and sort/reduceat path
def test_scatter_add_rows_matches_add_at(V, N):
    from smolgrad.functional import _scatter_add_rows
    rng = np.random.default_rng(V)
    idx = rng.integers(0, V, size=N)
    g = rng.standard_normal((N, 4))
    ref = np.zeros((V, 4))
    np.add.at(ref, idx, g)
    np.testing.assert_allclose(_scatter_add_rows(idx, g, V), ref, atol=1e-12)


def test_seed_gradient_is_not_aliased():
    a = Tensor(np.zeros(3), requires_grad=True)
    seed = np.ones(3)
    (a + 0.0).backward(seed)
    a.grad[0] = 42.0
    assert seed[0] == 1.0
