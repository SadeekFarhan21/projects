import json

import numpy as np
import pytest

import tcm
from tcm import _metal
from tcm.autotune import Autotuner
from conftest import assert_close, rand

SRC = """#include <metal_stdlib>
using namespace metal;
kernel void scale2(device const float* a [[buffer(0)]], device float* b [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) { if (i < 1000u) b[i] = a[i] * 2.0f; }
"""


def test_device_info():
    d = tcm.device()
    assert "Apple" in d.name
    assert d.max_threadgroup_memory >= 32768


def test_pipeline_cache_hits():
    d = tcm.device()
    before = d.stats()
    p1 = d.compile(SRC, "scale2")
    p2 = d.compile(SRC, "scale2")
    after = d.stats()
    assert p1 is p2 or p1.name == p2.name
    assert after["cache_hits"] >= before["cache_hits"] + 1


def test_compile_error_is_reported():
    with pytest.raises(RuntimeError, match="MSL compile failed"):
        tcm.device().compile("kernel void broken( {", "broken")


def test_batched_vs_per_dispatch_and_profile(rng):
    d = tcm.device()
    p = d.compile(SRC, "scale2")
    x = rand(rng, (1000,))
    a, b = d.alloc(4000), d.alloc(4000)
    d.upload(a, x)
    disp = _metal.Dispatch(p, [a, b], (4, 1, 1), (256, 1, 1))
    for mode in ("batched", "per_dispatch"):
        r = d.run([disp, disp], repeat=3, mode=mode)
        assert r["gpu_s"] > 0
        out = np.empty(1000, np.float32)
        d.download(b, out)
        np.testing.assert_array_equal(out, x * 2)
    r = d.run([disp, disp], repeat=2, profile=True)
    assert len(r["per_kernel_s"]) == 2 and all(t > 0 for t in r["per_kernel_s"])


def test_program_timing_and_profile(rng):
    g = tcm.trace(lambda x: tcm.softmax(x) + tcm.gelu(x).sum(-1), [((64, 256), "f32")])
    p = tcm.compile(g)
    p(rand(rng, (64, 256)))
    t = p.time(repeat=5)
    assert t["gpu_s"] > 0
    per = p.profile(repeat=3)
    assert len(per) == p.num_kernels


def test_autotune_cache_roundtrip(tmp_path, rng):
    cache = tmp_path / "tune.json"
    specs = [((64, 96), "f32"), ((96, 48), "f32")]
    g = tcm.trace(lambda a, b: tcm.softmax(a @ b), specs)
    xs = [rand(rng, s) for s, _ in specs]
    p = tcm.compile(g, autotune=True, tune_cache=str(cache))
    assert_close(p(*xs)[0], tcm.interpret(g, xs)[0], "f32")
    data = json.loads(cache.read_text())
    assert len(data) == p.num_kernels
    assert len(p.tuner.log) > 0
    # second compile hits the cache: no candidates are timed
    p2 = tcm.compile(g, autotune=True, tune_cache=str(cache))
    assert len(p2.tuner.log) == 0
    assert [k.config for k in p2.kernels] == [k.config for k in p.kernels]
    assert_close(p2(*xs)[0], tcm.interpret(g, xs)[0], "f32")
