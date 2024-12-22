import math
import numpy as np
import pytest
import wavepacket as wp
import wavepacket.typing as wpt
from numpy.testing import assert_allclose


@pytest.fixture
def dof() -> wp.PlaneWaveDof:
    return wp.PlaneWaveDof(10, 20, 8)


@pytest.fixture
def dx() -> float:
    return 10.0 / 8


def plane_wave(dof: wp.PlaneWaveDof, k_index: int) -> wpt.ComplexData:
    weight = 1.0 / len(dof.dvr_array)
    k = dof.fbr_array[k_index]
    psi = np.exp(1j * k * dof.dvr_array)

    full_size = np.ones([2, len(dof.dvr_array), 4])
    return math.sqrt(weight) * np.einsum("ijk, j -> ijk", full_size, psi)


def plane_wave_fbr(dof: wp.PlaneWaveDof, k_index: int) -> wpt.RealData:
    psi = np.zeros(len(dof.fbr_array))
    psi[k_index] = 1.0

    full_size = np.ones([2, len(dof.fbr_array), 4])
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


def test_ket_to_fbr(dof):
    psi = plane_wave(dof, 5)

    result = dof.to_fbr(psi, 1)

    expected = plane_wave_fbr(dof, 5)
    assert_allclose(result, expected, atol=1e-12)


def test_bra_to_fbr(dof):
    psi = np.conj(plane_wave(dof, 6))

    result = dof.to_fbr(psi, 1, is_ket=False)

    expected = plane_wave_fbr(dof, 6)
    assert_allclose(result, expected, atol=1e-12)


def test_ket_from_fbr(dof):
    psi = plane_wave_fbr(dof, 2)

    result = dof.from_fbr(psi, 1)

    expected = plane_wave(dof, 2)
    assert_allclose(result, expected, atol=1e-12)


def test_bra_from_fbr(dof):
    psi = plane_wave_fbr(dof, 1)

    result = dof.from_fbr(psi, 1, is_ket=False)

    expected = np.conj(plane_wave(dof, 1))
    assert_allclose(result, expected, atol=1e-12)


def test_to_dvr(dof, dx):
    psi = plane_wave(dof, 6)

    result = dof.to_dvr(psi, 1)

    expected = psi / math.sqrt(dx)
    assert_allclose(result, expected, atol=1e-12)


def test_from_dvr(dof, dx):
    psi = plane_wave(dof, 1)

    result = dof.from_dvr(psi / math.sqrt(dx), 1)

    assert_allclose(result, psi, atol=1e-12)