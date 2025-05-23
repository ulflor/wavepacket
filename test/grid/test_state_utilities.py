import numpy as np
import pytest
from numpy.testing import assert_allclose

import wavepacket as wp


def test_reject_invalid_states(grid_2d):
    invalid_state = wp.grid.State(grid_2d, np.ones(5))

    with pytest.raises(wp.BadStateError):
        wp.grid.dvr_density(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.grid.trace(invalid_state)


def test_wave_function_density():
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(5, 6, 3),
                         wp.grid.PlaneWaveDof(10, 12, 5)])
    psi = wp.testing.random_state(grid, 42)

    result = wp.grid.dvr_density(psi)

    dx = 1 / 3.0 * 2 / 5.0
    expected = np.abs(psi.data ** 2) / dx
    assert_allclose(result, expected, atol=1e-14, rtol=0)


def test_density_operator_density(grid_2d):
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)

    density_from_psi = wp.grid.dvr_density(psi)
    density_from_rho = wp.grid.dvr_density(rho)

    assert_allclose(density_from_rho, density_from_psi, atol=1e-14, rtol=0)


def test_trace():
    grid = wp.grid.Grid(wp.grid.PlaneWaveDof(1, 3, 4))
    psi = wp.grid.State(grid, np.array([0.5, 0.5j, 1, 2]))

    result = wp.grid.trace(psi)
    assert_allclose(result, 5.5, atol=1e-12, rtol=0)

    rho = wp.builder.pure_density(psi)
    result_rho = wp.grid.trace(rho)
    assert_allclose(result_rho, result, atol=1e-12, rtol=0)
