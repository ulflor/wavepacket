import numpy as np
from numpy.testing import assert_allclose

import wavepacket as wp
from wavepacket.testing import assert_close


def test_random_wave_function(grid_1d):
    generator = np.random.default_rng(42)

    result = wp.builder.random_wave_function(grid_1d, generator)
    assert result.is_wave_function()

    expected_abs = wp.builder.product_wave_function(grid_1d, lambda x: np.ones(x.shape), normalize=False)
    assert_allclose(np.abs(result.data), expected_abs.data, rtol=0, atol=1e-12)

    # proxy check that the arguments to the complex numbers differ
    distinct_real_vals = set(np.real(result.data).flat)
    assert len(distinct_real_vals) == result.grid.size


def test_random_reproducibility(grid_2d):
    generator = np.random.default_rng(1)
    same_generator = np.random.default_rng(1)
    other_generator = np.random.default_rng(2)

    psi = wp.builder.random_wave_function(grid_2d, generator)
    next_psi = wp.builder.random_wave_function(grid_2d, generator)
    same_psi = wp.builder.random_wave_function(grid_2d, same_generator)
    other_psi = wp.builder.random_wave_function(grid_2d, other_generator)

    assert np.all(psi.data == same_psi.data)
    assert np.all(psi.data != next_psi.data)
    assert np.all(psi.data != other_psi.data)


def test_unit_wave_function(grid_2d):
    psi = wp.builder.unit_wave_function(grid_2d)
    zero = wp.builder.zero_wave_function(grid_2d)

    assert_close(zero + 1, psi)


def test_zero_wave_function(grid_2d):
    psi = wp.builder.zero_wave_function(grid_2d)
    assert psi.is_wave_function()
    assert np.all(psi.data == 0.0)
