"""Every primitive on the GPU against the numpy oracle, f32 and f16."""

import numpy as np
import pytest

import tcm
from conftest import assert_close, rand

DTYPES = ["f32", "f16"]


def run_both(fn, specs, arrays, dtype, fuse=True):
    g = tcm.trace(fn, specs)
    got = tcm.compile(g, fuse=fuse)(*arrays)
    ref = tcm.interpret(g, arrays)
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype
        assert_close(a, b, dtype)
    return got


UNARY = {
    "neg": (lambda x: -x, -1, 1),
    "exp": (lambda x: x.exp(), -2, 2),
    "log": (lambda x: x.log(), 0.1, 3),
    "sqrt": (lambda x: x.sqrt(), 0.1, 3),
    "rsqrt": (lambda x: x.rsqrt(), 0.1, 3),
    "tanh": (lambda x: x.tanh(), -3, 3),
    "abs": (lambda x: x.abs(), -1, 1),
    "recip": (lambda x: x.recip(), 0.5, 3),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", sorted(UNARY))
@pytest.mark.parametrize("shape", [(7,), (4, 64), (3, 5, 6)])
def test_unary(op, dtype, shape, rng):
    fn, lo, hi = UNARY[op]
    x = rand(rng, shape, dtype, lo, hi)
    run_both(fn, [(shape, dtype)], [x], dtype)


BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (b.abs() + 0.5),
    "maximum": lambda a, b: a.maximum(b),
    "minimum": lambda a, b: a.minimum(b),
}
BCAST = [((4, 8), (4, 8)), ((4, 8), (8,)), ((4, 1), (1, 8)), ((2, 3, 8), (3, 1)), ((5,), (1,)), ((6, 7), (6, 1))]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", sorted(BINARY))
@pytest.mark.parametrize("shapes", BCAST)
def test_binary_broadcast(op, dtype, shapes, rng):
    sa, sb = shapes
    a, b = rand(rng, sa, dtype), rand(rng, sb, dtype)
    run_both(BINARY[op], [(sa, dtype), (sb, dtype)], [a, b], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("red", ["sum", "max"])
@pytest.mark.parametrize("shape", [(5,), (3, 7), (8, 128), (2, 3, 1000), (4, 4099), (300, 64)])
def test_reductions(red, dtype, shape, rng):
    x = rand(rng, shape, dtype)
    run_both(lambda t: getattr(t, red)(-1), [(shape, dtype)], [x], dtype)


def test_cast_roundtrip(rng):
    x = rand(rng, (4, 16), "f32", -100, 100)
    got = run_both(lambda t: t.astype("f16").astype("f32") * 2.0, [((4, 16), "f32")], [x], "f32")
    np.testing.assert_array_equal(got[0], x.astype(np.float16).astype(np.float32) * 2)


MM_SHAPES = [(1, 1, 1), (17, 33, 9), (64, 64, 64), (100, 70, 130), (128, 256, 96), (5, 300, 7)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mkn", MM_SHAPES)
def test_matmul_2d(mkn, dtype, rng):
    m, k, n = mkn
    a, b = rand(rng, (m, k), dtype), rand(rng, (k, n), dtype)
    run_both(lambda x, y: x @ y, [((m, k), dtype), ((k, n), dtype)], [a, b], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_batched_and_views(dtype, rng):
    a, b = rand(rng, (3, 20, 12), dtype), rand(rng, (3, 12, 9), dtype)
    run_both(lambda x, y: x @ y, [((3, 20, 12), dtype), ((3, 12, 9), dtype)], [a, b], dtype)
    w = rand(rng, (12, 9), dtype)
    run_both(lambda x, y: x @ y, [((3, 20, 12), dtype), ((12, 9), dtype)], [a, w], dtype)
    bt = rand(rng, (3, 9, 12), dtype)
    run_both(lambda x, y: x @ y.transpose(0, 2, 1), [((3, 20, 12), dtype), ((3, 9, 12), dtype)], [a, bt], dtype)
    at = rand(rng, (12, 20), dtype)
    run_both(lambda x, y: x.T @ y, [((12, 20), dtype), ((12, 9), dtype)], [at, w], dtype)


def test_matmul_same_buffer_both_operands(rng):
    x = rand(rng, (40, 24))
    run_both(lambda t: t @ t.T, [((40, 24), "f32")], [x], "f32")


@pytest.mark.parametrize("dtype", DTYPES)
def test_views_reshape_transpose(dtype, rng):
    x = rand(rng, (2, 3, 8), dtype)
    run_both(lambda t: t.transpose(2, 0, 1) * 2.0, [((2, 3, 8), dtype)], [x], dtype)
    run_both(lambda t: t.reshape(6, 8).T.exp(), [((2, 3, 8), dtype)], [x], dtype)
    run_both(lambda t: t.T.reshape(48), [((2, 3, 8), dtype)], [x], dtype)  # needs a copy
    run_both(lambda t: t.transpose(1, 0, 2).sum(-1), [((2, 3, 8), dtype)], [x], dtype)
    run_both(lambda t: t.reshape(2, 24).max(-1), [((2, 3, 8), dtype)], [x], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 10), (32, 768), (7, 3, 1000), (2, 4100)])
def test_composites(dtype, shape, rng):
    n = shape[-1]
    x = rand(rng, shape, dtype, -3, 3)
    g, b = rand(rng, (n,), dtype), rand(rng, (n,), dtype)
    run_both(tcm.softmax, [(shape, dtype)], [x], dtype)
    run_both(tcm.layernorm, [(shape, dtype), ((n,), dtype), ((n,), dtype)], [x, g, b], dtype)
    run_both(tcm.gelu, [(shape, dtype)], [x], dtype)


def test_constants_as_buffers(rng):
    w = rand(rng, (16, 8))
    x = rand(rng, (4, 16))
    run_both(lambda t: tcm.relu(t @ tcm.constant(w)) + tcm.constant(np.arange(8, dtype=np.float32)), [((4, 16), "f32")], [x], "f32")


def test_output_that_is_input_or_view(rng):
    x = rand(rng, (4, 6))
    got = run_both(lambda t: (t, t.T, t.exp()), [((4, 6), "f32")], [x], "f32")
    np.testing.assert_array_equal(got[0], x)
    np.testing.assert_array_equal(got[1], x.T)


def test_every_matmul_tile_config_is_correct(rng):
    """The autotuner may pick any candidate, so all of them must be correct (ragged shape)."""
    from tcm.autotune import matmul_candidates
    from tcm.codegen import gen_matmul
    from tcm.compiler import Program

    m, k, n = 70, 50, 90
    g = tcm.trace(lambda x, y: x @ y.T, [((m, k), "f32"), ((n, k), "f32")])
    a, b = rand(rng, (m, k)), rand(rng, (n, k))
    ref = tcm.interpret(g, [a, b])[0]
    p = Program(g)
    for cfg in matmul_candidates():
        p.replace_kernel(0, gen_matmul(p.graph, p.groups[0], "mm", cfg))
        assert_close(p(a, b)[0], ref, "f32")


def test_every_row_config_is_correct(rng):
    from tcm.autotune import row_candidates
    from tcm.codegen import gen_row
    from tcm.compiler import Program

    for shape in [(37, 1000), (5, 30)]:
        g = tcm.trace(tcm.softmax, [(shape, "f32")])
        x = rand(rng, shape, lo=-3, hi=3)
        ref = tcm.interpret(g, [x])[0]
        p = Program(g)
        vec = p.kernels[0].config["vec"]
        for cfg in row_candidates(shape[-1], vec):
            p.replace_kernel(0, gen_row(p.graph, p.groups[0], "r", cfg))
            assert_close(p(x)[0], ref, "f32")
