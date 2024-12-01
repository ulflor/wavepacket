import math
import numpy as np
import pytest
import wavepacket as wp


L = [10, 5]

@pytest.fixture
def grid() -> wp.grid.Grid:
    return wp.grid.Grid([wp.grid.plane_wave_dof(0, L[0], 5),
                         wp.grid.plane_wave_dof(10, 10+L[1], 4)])


def build_plane_wave(grid: wp.grid.Grid, k_index: int) -> wp.State:
    k = grid.dof[1].fbr[k_index]
    exp = np.exp(1j * k * grid.dof[1].dvr)

    psi = math.sqrt(1 / L[0] / L[1]) * np.ones(grid.shape)
    return wp.State(grid, psi * exp)


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


def test_transform_wave_function(grid):
    plane_wave = build_plane_wave(grid, 0)
