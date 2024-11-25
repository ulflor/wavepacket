import math
import numpy as np
from numpy.testing import assert_allclose
import pytest
import wavepacket as wp


def test_invalid_parameters():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.plane_wave_dof(10, 5, 10)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.plane_wave_dof(10, 10, 10)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.plane_wave_dof(5, 10, 0)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.plane_wave_dof(5, 10, -1)


def test_dvr_grid():
    dvr, _, _ = wp.grid.plane_wave_dof(1, 11, 5)

    expected = np.arange(1, 11.0, 2)
    assert_allclose(dvr, expected, atol=1e-10)


def test_fbr_grid():
    interval = 10.0
    _, fbr, _ = wp.grid.plane_wave_dof(-5, interval - 5, 16)

    # k points should be equidistant, and have the same period as the grid
    dk = 2 * math.pi / interval
    delta_fbr = fbr[1:] - fbr[:-1]
    assert_allclose(delta_fbr, dk * np.ones(15), atol=1e-10)

    args = 1j * fbr * interval
    assert_allclose(np.exp(args), np.ones(16))

    # We could place the grid anywhere in Fourier space, but we want
    # the zero in the center
    assert_allclose(fbr[8], 0.0, atol=1e-10)


def test_fbr_grid_for_uneven_points():
    # The same rules hold as for even grids, but the construction is different
    interval = 10.0
    n = 15
    _, fbr, _ = wp.grid.plane_wave_dof(-5, interval - 5, n)

    # k points should be equidistant, and have the same period as the grid
    dk = 2 * math.pi / interval
    delta_fbr = fbr[1:] - fbr[:-1]
    assert_allclose(delta_fbr, dk * np.ones(n-1), atol=1e-10)

    args = 1j * fbr * interval
    assert_allclose(np.exp(args), np.ones(n))

    # We could place the grid anywhere in Fourier space, but we want
    # the zero in the center
    assert_allclose(fbr[7], 0.0, atol=1e-10)


def test_weights():
    _, _, weights = wp.grid.plane_wave_dof(5, 10, 10)

    expected = 0.5 * np.ones(10)
    assert_allclose(weights, expected)