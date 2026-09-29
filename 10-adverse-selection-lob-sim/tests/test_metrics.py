import numpy as np

from marketsim.metrics import acf, excess_kurtosis, hill_tail_index, log_returns


def test_kurtosis_normal_vs_student_t():
    rng = np.random.default_rng(0)
    assert abs(excess_kurtosis(rng.standard_normal(200_000))) < 0.1
    assert excess_kurtosis(rng.standard_t(5, 200_000)) > 3  # theory: 6


def test_acf_white_noise_and_ar1():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100_000)
    assert np.all(np.abs(acf(x, 10)) < 0.02)
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = 0.7 * y[i - 1] + x[i]
    ac = acf(y, 3)
    np.testing.assert_allclose(ac, [0.7, 0.49, 0.343], atol=0.02)


def test_hill_recovers_pareto_index():
    rng = np.random.default_rng(2)
    x = rng.pareto(3.0, 500_000) + 1.0  # tail index 3
    assert abs(hill_tail_index(x, 0.01) - 3.0) < 0.2


def test_log_returns_subsampling():
    p = np.exp(np.arange(11, dtype=float))
    np.testing.assert_allclose(log_returns(p, 5), [5.0, 5.0])
