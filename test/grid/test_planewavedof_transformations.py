import numpy as np
import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.typing as wpt


@pytest.fixture
def dof() -> wp.grid.PlaneWaveDof:
    return wp.grid.PlaneWaveDof(10, 20, 8)


@pytest.fixture
def dx(dof) -> float:
    return 10.0 / dof.size


def plane_wave(dof: wp.grid.PlaneWaveDof, k_index: int) -> wpt.ComplexData:
    weight = 1.0 / len(dof.dvr_points)
    k = dof.fbr_points[k_index]
    psi = np.exp(1j * k * dof.dvr_points)

    full_size = np.ones([2, len(dof.dvr_points), 4])
    return np.sqrt(weight) * np.einsum("ijk, j -> ijk", full_size, psi)


def plane_wave_fbr(dof: wp.grid.PlaneWaveDof, k_index: int) -> wpt.RealData:
    psi = np.zeros(len(dof.fbr_points))
    psi[k_index] = 1.0

    full_size = np.ones([2, len(dof.fbr_points), 4])
    return np.einsum("ijk, j -> ijk", full_size, psi)


def test_reject_invalid_index(dof):
    psi = plane_wave(dof, 3)

    with pytest.raises(IndexError):
        dof.to_fbr(psi, 3)

    with pytest.raises(IndexError):
        dof.from_fbr(psi, 3)

    with pytest.raises(IndexError):
        dof.to_dvr(psi, 3)

    with pytest.raises(IndexError):
        dof.from_dvr(psi, 3)

    with pytest.raises(IndexError):
        dof.to_fbr(psi, -4)


def test_ket_to_fbr(dof):
    psi = plane_wave(dof, 5)

    result = dof.to_fbr(psi, 1)

    expected = plane_wave_fbr(dof, 5)
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_bra_to_fbr(dof):
    psi = np.conj(plane_wave(dof, 6))

    result = dof.to_fbr(psi, 1, is_ket=False)

    expected = plane_wave_fbr(dof, 6)
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_ket_from_fbr(dof):
    psi = plane_wave_fbr(dof, 2)

    result = dof.from_fbr(psi, 1)

    expected = plane_wave(dof, 2)
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_bra_from_fbr(dof):
    psi = plane_wave_fbr(dof, 1)

    result = dof.from_fbr(psi, 1, is_ket=False)

    expected = np.conj(plane_wave(dof, 1))
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_to_dvr(dof, dx):
    psi = plane_wave(dof, 6)

    result = dof.to_dvr(psi, 1)

    expected = psi / np.sqrt(dx)
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_from_dvr(dof, dx):
    psi = plane_wave(dof, 1)

    result = dof.from_dvr(psi / np.sqrt(dx), 1)

    assert_allclose(result, psi, atol=1e-12, rtol=0)


def test_negative_indices(dof):
    psi = plane_wave(dof, 2)

    positive = dof.to_fbr(psi, 1)
    negative = dof.to_fbr(psi, -2)
    assert_allclose(positive, negative, atol=1e-12, rtol=0)

    positive = dof.from_fbr(psi, 1)
    negative = dof.from_fbr(psi, -2)
    assert_allclose(positive, negative, atol=1e-12, rtol=0)

    positive = dof.to_dvr(psi, 1)
    negative = dof.to_dvr(psi, -2)
    assert_allclose(positive, negative, atol=1e-12, rtol=0)

    positive = dof.from_dvr(psi, 1)
    negative = dof.from_dvr(psi, -2)
    assert_allclose(positive, negative, atol=1e-12, rtol=0)
