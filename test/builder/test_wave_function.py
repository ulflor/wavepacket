import math
import numpy as np

import pytest

import wavepacket as wp


def test_reject_invalid_arguments(grid_2d):
    generator = np.random.default_rng(42)

    with pytest.raises(wp.InvalidValueError):
        wp.builder.random_wave_function(grid_2d, generator, -1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.builder.random_wave_function(grid_2d, generator, 0.0)


def test_random_wave_function():
    # The grid has small spacing, hence small weight values.
    # This is intentional to highlight differences between DVR and weighted DVR,
    # to check that the wave function is returned in the correct representation!
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(0, 0.1, 10),
                         wp.grid.PlaneWaveDof(1, 1.1, 4)])
    generator = np.random.default_rng(42)
    max_value = 0.5

    result = wp.builder.random_wave_function(grid, generator, max_value)

    assert result.grid == grid
    assert result.is_wave_function()
    assert np.all(np.imag(result.data).all() == 0)

    # The actual comparisons are statistical, hence a bit weak
    max_expected = max_value * math.sqrt(0.1 / 10) * math.sqrt(0.1 / 4)
    data = np.real(result.data)

    np.testing.assert_array_less(np.abs(data), max_expected)
    assert np.any(data > max_value / 100)
    assert np.any(data < -max_value / 100)


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


def test_zero_wave_function(grid_2d):
    psi = wp.builder.zero_wave_function(grid_2d)
    assert psi.is_wave_function()
    assert np.all(psi.data == 0.0)
