import numpy as np
import pytest

RTOL = {"f32": 1e-5, "f16": 1e-2}


def normalized_error(got, ref) -> float:
    """max |got - ref| / max(|ref|): the scale-normalized relative error used by every GPU test."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1e-30)
    return float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0


def assert_close(got, ref, dtype="f32", rtol=None):
    rtol = RTOL[dtype] if rtol is None else rtol
    # per-element relative test with an absolute floor tied to the tensor scale
    got64 = np.asarray(got, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    assert got64.shape == ref64.shape, (got64.shape, ref64.shape)
    assert np.all(np.isfinite(got64) == np.isfinite(ref64)), "finiteness differs"
    scale = max(float(np.max(np.abs(ref64[np.isfinite(ref64)]))) if np.isfinite(ref64).any() else 1.0, 1e-30)
    ok = np.abs(got64 - ref64) <= rtol * (np.abs(ref64) + scale)
    ok |= ~np.isfinite(ref64) & (got64 == ref64)
    if not ok.all():
        i = np.unravel_index(np.argmin(ok), ok.shape)
        raise AssertionError(
            f"mismatch at {i}: got {got64[i]} ref {ref64[i]} (normalized err {normalized_error(got64, ref64):.3g}, rtol {rtol})"
        )


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def rand(rng, shape, dtype="f32", lo=-1.0, hi=1.0):
    npdt = np.float32 if dtype == "f32" else np.float16
    return rng.uniform(lo, hi, size=shape).astype(npdt)
