import math
import numpy as np
import pytest
import wavepacket as wp

from numpy.testing import assert_allclose


def test_reject_invalid_constructor_values():
    with pytest.raises(wp.InvalidValueError):
        wp.PlaneWaveDof(10, 5, 10)

    with pytest.raises(wp.InvalidValueError):
        wp.PlaneWaveDof(5, 10, 0)

    with pytest.raises(wp.InvalidValueError):
        wp.PlaneWaveDof(5, 10, -1)


def test_dvr_grid():
    dof = wp.PlaneWaveDof(1, 11, 5)

    expected = np.arange(1, 11.0, 2)
    assert_allclose(dof.dvr_array, expected, atol=1e-10)


def test_fbr_grid():
    interval = 10.0
    dof = wp.PlaneWaveDof(-5, interval - 5, 16)

    # k points should be equidistant, and have the same period as the grid
    dk = 2 * math.pi / interval
    delta_fbr = dof.fbr_array[1:] - dof.fbr_array[:-1]
    assert_allclose(delta_fbr, dk * np.ones(15), atol=1e-10)

    args = 1j * dof.fbr_array * interval
    assert_allclose(np.exp(args), np.ones(16))

    # We could place the grid anywhere in Fourier space, but we want
    # the zero in the center
    assert_allclose(dof.fbr_array[8], 0.0, atol=1e-10)


def test_fbr_grid_for_uneven_points():
    # The same rules hold as for even grids, but the construction is different
    interval = 10.0
    n = 15
    dof = wp.PlaneWaveDof(-5, interval - 5, n)

    # k points should be equidistant, and have the same period as the grid
    dk = 2 * math.pi / interval
    delta_fbr = dof.fbr_array[1:] - dof.fbr_array[:-1]
    assert_allclose(delta_fbr, dk * np.ones(n - 1), atol=1e-10)

    args = 1j * dof.fbr_array * interval
    assert_allclose(np.exp(args), np.ones(n))

    # We could place the grid anywhere in Fourier space, but we want
    # the zero in the center
    assert_allclose(dof.fbr_array[7], 0.0, atol=1e-10)