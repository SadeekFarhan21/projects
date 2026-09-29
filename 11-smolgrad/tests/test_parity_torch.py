"""Parity with PyTorch (used only as a reference oracle): forward values and
gradients of ops, modules, and multi-step optimizer trajectories."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import smolgrad as sg  # noqa: E402
import smolgrad.functional as F  # noqa: E402
import smolgrad.nn as nn  # noqa: E402
from smolgrad import Tensor  # noqa: E402

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(7)


def pair(arr):
    """The same float64 data as a smolgrad leaf and a torch leaf."""
    return Tensor(arr.copy(), requires_grad=True), torch.tensor(arr.copy(), requires_grad=True)


OPS = [
    ("add_bcast", lambda a, b: a + b, lambda a, b: a + b, [(4, 1, 3), (5, 3)]),
    ("mul_bcast", lambda a, b: a * b, lambda a, b: a * b, [(4, 5, 3), (3,)]),
    ("div", lambda a, b: a / (b * b + 1.0), lambda a, b: a / (b * b + 1.0), [(4, 3), (4, 3)]),
    ("matmul_batched_bcast", lambda a, b: a @ b, lambda a, b: a @ b, [(2, 1, 4, 3), (5, 3, 6)]),
    ("sum_axes", lambda a: a.sum(axis=(1, 2)), lambda a: a.sum(dim=(1, 2)), [(2, 3, 4)]),
    ("mean_keep", lambda a: a.mean(axis=1, keepdims=True), lambda a: a.mean(dim=1, keepdim=True), [(2, 3, 4)]),
    ("exp_log", lambda a: (a.exp() + 1.0).log(), lambda a: (a.exp() + 1.0).log(), [(3, 4)]),
    ("relu", lambda a: a.relu(), lambda a: a.relu(), [(3, 4)]),
    ("tanh", lambda a: a.tanh(), lambda a: a.tanh(), [(3, 4)]),
    ("softmax", lambda a: a.softmax(1), lambda a: a.softmax(1), [(2, 5, 3)]),
    ("log_softmax", lambda a: a.log_softmax(-1), lambda a: a.log_softmax(-1), [(4, 7)]),
    ("transpose", lambda a: a.transpose(0, 2), lambda a: a.transpose(0, 2), [(2, 3, 4)]),
    ("reshape", lambda a: a.reshape(6, 4), lambda a: a.reshape(6, 4), [(2, 3, 4)]),
    ("getitem", lambda a: a[1:, [0, 0, 2]], lambda a: a[1:, [0, 0, 2]], [(3, 4)]),
]


@pytest.mark.parametrize("name,sg_fn,t_fn,shapes", OPS, ids=[o[0] for o in OPS])
def test_op_parity(name, sg_fn, t_fn, shapes):
    arrs = [rng.standard_normal(s) for s in shapes]
    pairs = [pair(a) for a in arrs]
    out_s = sg_fn(*[p[0] for p in pairs])
    out_t = t_fn(*[p[1] for p in pairs])
    np.testing.assert_allclose(out_s.data, out_t.detach().numpy(), rtol=1e-10, atol=1e-12)
    w = rng.standard_normal(out_s.shape)
    (out_s * Tensor(w)).sum().backward()
    (out_t * torch.tensor(w)).sum().backward()
    for s, t in pairs:
        np.testing.assert_allclose(s.grad, t.grad.numpy(), rtol=1e-10, atol=1e-12)


def test_cross_entropy_parity():
    x_s, x_t = pair(rng.standard_normal((16, 10)))
    y = rng.integers(0, 10, size=16)
    l_s = F.cross_entropy(x_s, y)
    l_t = torch.nn.functional.cross_entropy(x_t, torch.tensor(y))
    assert l_s.item() == pytest.approx(l_t.item(), rel=1e-12)
    l_s.backward()
    l_t.backward()
    np.testing.assert_allclose(x_s.grad, x_t.grad.numpy(), rtol=1e-10, atol=1e-14)


def test_embedding_parity_with_repeats():
    w_s, w_t = pair(rng.standard_normal((7, 4)))
    idx = rng.integers(0, 7, size=(5, 6))  # 30 lookups into 7 rows, so many repeats
    o_s = F.embedding(w_s, idx)
    o_t = torch.nn.functional.embedding(torch.tensor(idx), w_t)
    np.testing.assert_allclose(o_s.data, o_t.detach().numpy())
    g = rng.standard_normal(o_s.shape)
    o_s.backward(g)
    o_t.backward(torch.tensor(g))
    np.testing.assert_allclose(w_s.grad, w_t.grad.numpy(), rtol=1e-12, atol=1e-12)


def test_layer_norm_parity():
    x_s, x_t = pair(rng.standard_normal((3, 5, 8)))
    ln = nn.LayerNorm(8, dtype=np.float64)
    ln.weight.data[:] = rng.standard_normal(8)
    ln.bias.data[:] = rng.standard_normal(8)
    tln = torch.nn.LayerNorm(8)
    with torch.no_grad():
        tln.weight.copy_(torch.tensor(ln.weight.data))
        tln.bias.copy_(torch.tensor(ln.bias.data))
    o_s, o_t = ln(x_s), tln(x_t)
    np.testing.assert_allclose(o_s.data, o_t.detach().numpy(), rtol=1e-9, atol=1e-10)
    g = rng.standard_normal(o_s.shape)
    o_s.backward(g)
    o_t.backward(torch.tensor(g))
    np.testing.assert_allclose(x_s.grad, x_t.grad.numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ln.weight.grad, tln.weight.grad.numpy(), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(ln.bias.grad, tln.bias.grad.numpy(), rtol=1e-9, atol=1e-10)


def _mlp_pair(dtype=np.float64):
    sg.manual_seed(3)
    m = nn.Sequential(nn.Linear(6, 16, dtype=dtype), nn.Tanh(), nn.Linear(16, 16, dtype=dtype),
                      nn.ReLU(), nn.Linear(16, 4, dtype=dtype))
    tm = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 16),
                             torch.nn.ReLU(), torch.nn.Linear(16, 4))
    copy_weights(m, tm)
    return m, tm


def copy_weights(m, tm):
    """Copy smolgrad parameters into a torch module with the same layout."""
    tparams = list(tm.parameters())
    sparams = m.parameters()
    assert len(tparams) == len(sparams)
    with torch.no_grad():
        for s, t in zip(sparams, tparams):
            assert tuple(t.shape) == s.shape
            t.copy_(torch.tensor(s.data, dtype=t.dtype))


def test_module_parameter_discovery_order():
    m, _ = _mlp_pair()
    names = [n for n, _ in m.named_parameters()]
    assert names == ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                     "layers.4.weight", "layers.4.bias"]
    assert m.num_parameters() == 6 * 16 + 16 + 16 * 16 + 16 + 16 * 4 + 4


def test_state_dict_roundtrip():
    m, _ = _mlp_pair()
    sd = m.state_dict()
    for p in m.parameters():
        p.data[...] = 0.0
    m.load_state_dict(sd)
    assert all(np.array_equal(sd[n], p.data) for n, p in m.named_parameters())


@pytest.mark.parametrize("opt_name", ["sgd", "sgd_momentum_wd", "adamw"])
def test_optimizer_trajectory_parity(opt_name):
    """20 steps of training on the same data must track PyTorch to ~1e-10."""
    m, tm = _mlp_pair()
    if opt_name == "sgd":
        opt, topt = sg.optim.SGD(m.parameters(), lr=0.1), torch.optim.SGD(tm.parameters(), lr=0.1)
    elif opt_name == "sgd_momentum_wd":
        opt = sg.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
        topt = torch.optim.SGD(tm.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    else:
        opt = sg.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
        topt = torch.optim.AdamW(tm.parameters(), lr=1e-2, weight_decay=0.1)
    data_rng = np.random.default_rng(11)
    for _ in range(20):
        x = data_rng.standard_normal((8, 6))
        y = data_rng.integers(0, 4, size=8)
        opt.zero_grad()
        loss = F.cross_entropy(m(Tensor(x)), y)
        loss.backward()
        opt.step()
        topt.zero_grad()
        tloss = torch.nn.functional.cross_entropy(tm(torch.tensor(x)), torch.tensor(y))
        tloss.backward()
        topt.step()
        assert loss.item() == pytest.approx(tloss.item(), rel=1e-9)
    for s, t in zip(m.parameters(), tm.parameters()):
        np.testing.assert_allclose(s.data, t.detach().numpy(), rtol=1e-8, atol=1e-10)


def test_float32_mlp_parity():
    """Same check in float32 (the training dtype) with float32-appropriate tolerances."""
    m, tm = _mlp_pair(np.float32)
    tm = tm.float()
    copy_weights(m, tm)
    x = rng.standard_normal((32, 6)).astype(np.float32)
    y = rng.integers(0, 4, size=32)
    loss = F.cross_entropy(m(Tensor(x)), y)
    loss.backward()
    tloss = torch.nn.functional.cross_entropy(tm(torch.tensor(x)), torch.tensor(y))
    tloss.backward()
    assert loss.dtype == np.float32
    assert loss.item() == pytest.approx(tloss.item(), rel=1e-5)
    for s, t in zip(m.parameters(), tm.parameters()):
        assert s.grad.dtype == np.float32
        np.testing.assert_allclose(s.grad, t.grad.numpy(), rtol=1e-4, atol=1e-6)
