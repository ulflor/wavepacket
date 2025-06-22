import pytest

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
