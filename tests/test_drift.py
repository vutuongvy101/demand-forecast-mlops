"""PSI drift detection on synthetic shifted distributions."""

import numpy as np

from forecast.drift import population_stability_index


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    psi = population_stability_index(data, data)
    assert psi < 0.01


def test_psi_shifted_distribution():
    rng = np.random.default_rng(42)
    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(2, 1, 1000)  # shifted mean
    psi = population_stability_index(expected, actual)
    assert psi > 0.1


def test_psi_empty_arrays():
    assert population_stability_index(np.array([]), np.array([1.0])) == 0.0
