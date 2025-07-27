import numpy as np
import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.typing as wpt


def spherical_harmonics(dof: wp.grid.SphericalHarmonicsDof) -> wpt.RealData:
    # harmonics[k,l] = l-th harmonic, point x_k
    return np.stack([wp.SphericalHarmonic(L + abs(dof.m), dof.m)(dof.dvr_points) for L in range(dof.size)], 1)


def test_correct_weights():
    dof = wp.grid.SphericalHarmonicsDof(5, 0)
    harmonics = spherical_harmonics(dof)
    weighted = dof.from_dvr(harmonics, 0)

    scalar_product = np.matmul(np.conjugate(weighted.T), weighted)
    assert_allclose(scalar_product, np.eye(6), rtol=0, atol=1e-12)

    # same for m > 0
    dof_m = wp.grid.SphericalHarmonicsDof(5, 2)
    harmonics_m = spherical_harmonics(dof_m)
    weighted_m = dof_m.from_dvr(harmonics_m, 0)

    scalar_product = np.matmul(np.conjugate(weighted_m.T), weighted_m)
    assert_allclose(scalar_product, np.eye(4), rtol=0, atol=1e-12)

    # and m < 0
    dof_mm = wp.grid.SphericalHarmonicsDof(5, -1)
    harmonics_mm = spherical_harmonics(dof_mm)
    weighted_mm = dof_mm.from_dvr(harmonics_mm, 0)

    scalar_product = np.matmul(np.conjugate(weighted_mm.T), weighted_mm)
    assert_allclose(scalar_product, np.eye(5), rtol=0, atol=1e-12)


def test_from_dvr_along_specified_index():
    dof = wp.grid.SphericalHarmonicsDof(6, 1)
    harmonics = spherical_harmonics(dof)
    harmonics_t = np.transpose(harmonics)

    result = dof.from_dvr(harmonics, 0)
    result_t = dof.from_dvr(harmonics_t, 1)

    assert_allclose(np.transpose(result), result_t, rtol=0, atol=1e-12)


def test_reject_invalid_index():
    dof = wp.grid.SphericalHarmonicsDof(6, 3)
    data = np.ones([3, 4, 5])

    with pytest.raises(IndexError):
        dof.from_dvr(data, 3)

    with pytest.raises(IndexError):
        dof.to_dvr(data, 3)

    with pytest.raises(IndexError):
        dof.from_fbr(data, 3)

    with pytest.raises(IndexError):
        dof.to_fbr(data, 3)


def test_to_dvr_is_inverse_of_from_dvr():
    dof = wp.grid.SphericalHarmonicsDof(7, 2)
    data = np.random.default_rng(42).random((dof.size, dof.size, dof.size))

    tmp = dof.to_dvr(data, 0)
    result = dof.from_dvr(tmp, 0)
    assert_allclose(result, data, rtol=0, atol=1e-12)

    tmp = dof.to_dvr(data, 1)
    result = dof.from_dvr(tmp, 1)
    assert_allclose(result, data, rtol=0, atol=1e-12)


def test_transformation_from_fbr():
    dof = wp.grid.SphericalHarmonicsDof(7, 2)
    harmonics = spherical_harmonics(dof)
    fbr_harmonics = np.eye(harmonics.shape[0])

    expected = dof.from_dvr(harmonics, 0)
    got = dof.from_fbr(fbr_harmonics, 0)

    assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_from_fbr_along_specified_index():
    dof = wp.grid.SphericalHarmonicsDof(7, 3)
    data = np.random.default_rng(42).random((dof.size, dof.size, dof.size))

    result = dof.from_fbr(data, 0)
    transposed_result = dof.from_fbr(np.swapaxes(data, 0, 2), 2)

    assert_allclose(result, np.swapaxes(transposed_result, 0, 2), rtol=0, atol=1e-12)


def test_to_fbr():
    dof = wp.grid.SphericalHarmonicsDof(7, -4)
    harmonics = spherical_harmonics(dof)
    transposed_harmonics = np.transpose(harmonics)

    result = dof.to_fbr(dof.from_dvr(harmonics, 0), 0)
    transposed_result = dof.to_fbr(dof.from_dvr(transposed_harmonics, -1), -1)

    assert_allclose(np.eye(4), result, rtol=0, atol=1e-12)
    assert_allclose(np.eye(4), transposed_result, rtol=0, atol=1e-12)
