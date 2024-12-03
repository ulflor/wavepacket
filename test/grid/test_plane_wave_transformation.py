import math
import numpy as np
import pytest
import wavepacket as wp
from numpy.testing import assert_allclose


@pytest.fixture
def grid() -> wp.grid.Grid:
    return wp.grid.Grid([wp.grid.plane_wave_dof(0, 10, 5),
                         wp.grid.plane_wave_dof(10, 10 + 7, 6)])


def plane_wave_dvr(grid: wp.grid.Grid, k_index: int) -> wp.State:
    dof = grid.dof[1]
    k = dof.fbr[k_index]
    exp = np.exp(1j * k * dof.dvr)

    psi = math.sqrt(1 / dof.size) * np.ones(grid.shape)
    return wp.State(grid, psi * exp)


def plane_wave_fbr(grid: wp.grid.Grid, k_index: int) -> wp.State:
    psi = np.zeros(grid.shape)
    psi[:, k_index] = 1.0

    return wp.State(grid, psi)


def test_reject_invalid_index(grid):
    with pytest.raises(IndexError):
        wp.grid.PlaneWaveTransformation(grid, 2)

    with pytest.raises(IndexError):
        wp.grid.PlaneWaveTransformation(grid, -3)

    wp.grid.PlaneWaveTransformation(grid, -2)
    wp.grid.PlaneWaveTransformation(grid, 1)


def test_require_plane_wave_dof():
    dof = wp.grid.DegreeOfFreedom(wp.grid.DofType.OTHER,
                                  np.ones(2), np.ones(2), np.ones(2))
    grid = wp.grid.Grid(dof)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.PlaneWaveTransformation(grid, 0)


def test_require_same_grid(grid):
    other_grid = wp.grid.Grid(wp.grid.plane_wave_dof(10, 20, 5))
    transformation = wp.grid.PlaneWaveTransformation(other_grid, 0)

    with pytest.raises(wp.BadGridError):
        state = plane_wave_dvr(grid, 0)
        transformation.ket_to_fbr(state)


def test_transform_wave_function_to_fbr(grid):
    transformation = wp.grid.PlaneWaveTransformation(grid, 1)
    psi_dvr = plane_wave_dvr(grid, 1)

    psi_fbr = transformation.ket_to_fbr(psi_dvr)

    assert_allclose(psi_fbr.data, plane_wave_fbr(grid, 1).data, atol=1e-10)


def test_transform_density_ket_to_fbr(grid):
    transformation = wp.grid.PlaneWaveTransformation(grid, 1)
    psi_dvr = plane_wave_dvr(grid, 3)
    rho_dvr = wp.State(grid, np.tensordot(psi_dvr.data, np.conj(psi_dvr.data), axes=((), ())))

    fbr_dvr = transformation.ket_to_fbr(rho_dvr)

    psi_fbr = plane_wave_fbr(grid, 3)
    expected_fbr_dvr = wp.State(grid, np.tensordot(psi_fbr.data, np.conj(psi_dvr.data), axes=((), ())))
    assert_allclose(fbr_dvr.data, expected_fbr_dvr.data)


def test_throw_on_invalid_ket(grid):
    transformation = wp.grid.PlaneWaveTransformation(grid, 1)
    state = wp.State(grid, np.ones(5))

    with pytest.raises(wp.BadStateError):
        transformation.ket_to_fbr(state)