import numpy as np
import pytest

import tcm
from tcm.ir import Graph, broadcast_shapes


def test_trace_records_ssa_graph():
    g = tcm.trace(lambda x, y: (x + y).exp().sum(-1), [((4, 8), "f32"), ((8,), "f32")])
    g.verify()
    ops = [n.op for n in g]
    assert ops == ["input", "input", "add", "exp", "reduce_sum"]
    assert g[g.outputs[0]].shape == (4, 1)
    assert g.input_names() == ["in0", "in1"]


def test_broadcast_rules():
    assert broadcast_shapes((4, 1, 3), (5, 1)) == (4, 5, 3)
    with pytest.raises(ValueError):
        broadcast_shapes((4, 3), (2, 3))


def test_dtype_mismatch_rejected():
    with pytest.raises(TypeError):
        tcm.trace(lambda a, b: a + b, [((4,), "f32"), ((4,), "f16")])


def test_cast_and_scalar_lifting():
    g = tcm.trace(lambda a: (a * 2.0).astype("f16") + 1.0, [((4,), "f32")])
    out = g[g.outputs[0]]
    assert out.dtype == "f16"
    consts = [n for n in g if n.op == "const"]
    assert {c.dtype for c in consts} == {"f32", "f16"}


def test_views_shapes():
    g = tcm.trace(lambda a: a.transpose(2, 0, 1).reshape(6, -1).T, [((2, 3, 4), "f32")])
    assert g[g.outputs[0]].shape == (4, 6)


def test_matmul_shapes():
    g = tcm.trace(lambda a, b: a @ b, [((5, 3, 4), "f32"), ((4, 7), "f32")])
    assert g[g.outputs[0]].shape == (5, 3, 7)
    with pytest.raises(ValueError):
        tcm.trace(lambda a, b: a @ b, [((3, 4), "f32"), ((5, 7), "f32")])


def test_only_last_axis_reductions():
    with pytest.raises(NotImplementedError):
        tcm.trace(lambda a: a.sum(0), [((3, 4), "f32")])


def test_ssa_violation_detected():
    g = Graph()
    with pytest.raises(ValueError):
        g.add("neg", [42], (3,), "f32")


def test_composites_are_primitives_only():
    g = tcm.trace(tcm.layernorm, [((2, 8), "f32"), ((8,), "f32"), ((8,), "f32")])
    assert set(g.count_ops()) <= {"input", "const", "reduce_sum", "mul", "sub", "add", "rsqrt"}
    g = tcm.trace(tcm.softmax, [((2, 8), "f32")])
    assert set(g.count_ops()) == {"input", "reduce_max", "sub", "exp", "reduce_sum", "div"}
