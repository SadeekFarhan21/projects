import numpy as np
import pytest

import tcm
from tcm.passes import constant_fold, cse, dce, infer_shapes, optimize, simplify
from tcm.schedule import fuse_groups, legalize, plan_buffers, resolve_view, restride
from conftest import rand


def ops(g):
    return [n.op for n in g]


def test_constant_folding_and_dce():
    def f(x):
        c = tcm.constant(np.arange(4, dtype=np.float32))
        k = (c * 2.0 + 1.0).exp()  # all constant
        unused = x.tanh()  # dead
        return x + k

    g = optimize(tcm.trace(f, [((4,), "f32")]))
    assert ops(g) == ["input", "const", "add"]
    np.testing.assert_allclose(g[1].attrs["value"], np.exp(np.arange(4) * 2 + 1).astype(np.float32), rtol=1e-6)


def test_cse_merges_duplicates():
    g = tcm.trace(lambda x: x.exp() + x.exp(), [((4,), "f32")])
    g2 = dce(cse(g))
    assert g2.count_ops().get("exp") == 1


def test_algebraic_identities():
    g = tcm.trace(lambda x: -(-((x * 1.0 + 0.0) / 1.0)), [((4, 4), "f32")])
    g = optimize(g)
    assert ops(g) == ["input", "copy"] or ops(g) == ["input"]


def test_div_by_const_becomes_mul_and_rsqrt():
    g = optimize(tcm.trace(lambda x: x / 4.0 + 1.0 / x.sqrt(), [((4,), "f32")]))
    c = g.count_ops()
    assert "div" not in c and c.get("rsqrt") == 1 and c.get("mul") == 1


def test_transpose_reshape_chains_collapse():
    g = optimize(tcm.trace(lambda x: x.T.T.reshape(2, 8).reshape(4, 4) + 1.0, [((4, 4), "f32")]))
    assert "transpose" not in g.count_ops() and "reshape" not in g.count_ops()


def test_reduce_over_unit_axis_removed():
    g = optimize(tcm.trace(lambda x: x.sum(-1).sum(-1), [((3, 5), "f32")]))
    assert g.count_ops()["reduce_sum"] == 1


def test_shape_inference_catches_corruption():
    g = tcm.trace(lambda x: x.exp(), [((4,), "f32")])
    n = g[1]
    g.nodes[1] = type(n)(n.id, n.op, n.inputs, (5,), n.dtype, n.attrs)
    with pytest.raises(AssertionError):
        infer_shapes(g)


def test_optimize_preserves_semantics(rng):
    def f(x, y):
        return ((x * 1.0) / 2.0 + y.T.T).exp().sum(-1) + (x - 0.0).max(-1)

    g = tcm.trace(f, [((3, 6), "f32"), ((3, 6), "f32")])
    xs = [rand(rng, (3, 6)), rand(rng, (3, 6))]
    np.testing.assert_allclose(tcm.interpret(optimize(g), xs)[0], tcm.interpret(g, xs)[0], rtol=1e-6)


def test_restride():
    assert restride((4, 6), (6, 1), (24,)) == (1,)
    assert restride((4, 6), (6, 1), (2, 2, 6)) == (12, 6, 1)
    assert restride((6, 4), (1, 6), (24,)) is None  # transposed: needs a copy
    assert restride((6, 4), (1, 6), (6, 2, 2)) == (1, 12, 6)


def test_legalize_inserts_copy_for_non_view_reshape():
    g = tcm.trace(lambda x: x.T.reshape(24) + 1.0, [((4, 6), "f32")])
    g = legalize(optimize(g))
    assert g.count_ops().get("copy") == 1
    for n in g:
        if n.op == "reshape":
            resolve_view(g, n.id)  # must not raise


def _kernels(fn, specs, fuse=True):
    g = legalize(optimize(tcm.trace(fn, specs)))
    return g, fuse_groups(g, fuse)


def test_softmax_layernorm_gelu_fuse_to_one_kernel():
    _, gs = _kernels(tcm.softmax, [((8, 128), "f32")])
    assert len(gs) == 1 and gs[0].kind == "row"
    _, gs = _kernels(tcm.layernorm, [((8, 128), "f32"), ((128,), "f32"), ((128,), "f32")])
    assert len(gs) == 1 and gs[0].kind == "row"
    _, gs = _kernels(tcm.gelu, [((8, 128), "f32")])
    assert len(gs) == 1 and gs[0].kind == "ew"


def test_unfused_is_one_kernel_per_op():
    g, gs = _kernels(tcm.softmax, [((8, 128), "f32")], fuse=False)
    assert len(gs) == 5


def test_matmul_breaks_fusion():
    g, gs = _kernels(lambda x, w: tcm.gelu(x @ w).sum(-1), [((8, 16), "f32"), ((16, 32), "f32")])
    assert [k.kind for k in gs] == ["matmul", "row"]


def test_multi_consumer_value_is_materialized():
    g, gs = _kernels(lambda x: (x.exp() + 1.0, x.exp() * 2.0), [((8, 16), "f32")])
    assert len(gs) == 3  # shared exp materialized once, two consumers


def test_buffer_reuse_reduces_memory():
    def chain(x):
        for _ in range(6):
            x = (x * 1.5).tanh() + 0.1
        return x

    g, gs = _kernels(chain, [((256, 256), "f32")], fuse=False)
    plan = plan_buffers(g, gs)
    assert plan.naive_temp_bytes > plan.pooled_temp_bytes
    assert plan.pooled_temp_bytes <= 2 * 256 * 256 * 4
    # a kernel never writes a buffer it reads
    for grp in gs:
        from tcm.schedule import group_leaves

        out_slot = plan.slot_of[grp.root]
        for leaf in group_leaves(g, grp):
            if g[leaf].op == "const":
                continue
            base, _ = resolve_view(g, leaf)
            assert plan.slot_of[base] != out_slot
