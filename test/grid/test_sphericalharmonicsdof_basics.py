import pytest
from numpy.testing import assert_equal, assert_allclose

import wavepacket as wp


def test_too_small_angular_momentum():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.SphericalHarmonicsDof(lmax=-1, m=0)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.SphericalHarmonicsDof(lmax=5, m=6)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.SphericalHarmonicsDof(lmax=5, m=-6)


def test_size_of_grid():
    dof = wp.grid.SphericalHarmonicsDof(lmax=5, m=2)
    dof2 = wp.grid.SphericalHarmonicsDof(lmax=5, m=-2)

    assert dof.fbr_points.shape == (5 - 2 + 1,)
    assert dof.fbr_points.shape == dof2.fbr_points.shape


def test_fbr_grid():
    dof = wp.grid.SphericalHarmonicsDof(lmax=6, m=2)
    assert_equal(dof.fbr_points, [2, 3, 4, 5, 6], 0)

    dof2 = wp.grid.SphericalHarmonicsDof(lmax=6, m=-2)
    assert_equal(dof.fbr_points, dof2.fbr_points)

    dof3 = wp.grid.SphericalHarmonicsDof(lmax=5, m=0)
    assert_equal(dof3.fbr_points, [0, 1, 2, 3, 4, 5])


def test_dvr_grid():
    dof = wp.grid.SphericalHarmonicsDof(lmax=6, m=3)

    # values taken from the C++ reference implementation
    expected = [0.87546157, 1.34273687, 1.79885579, 2.26613109]
    assert_allclose(dof.dvr_points, expected, atol=1e-7, rtol=0)


def test_grids_independent_of_sign_of_m():
    dof1 = wp.grid.SphericalHarmonicsDof(7, 2)
    dof2 = wp.grid.SphericalHarmonicsDof(7, -2)

    assert_allclose(dof1.dvr_points, dof2.dvr_points, atol=1e-12, rtol=0)
    assert_allclose(dof1.fbr_points, dof2.fbr_points, atol=1e-12, rtol=0)
