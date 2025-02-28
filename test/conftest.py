import pytest

import wavepacket as wp


@pytest.fixture(scope="module")
def grid_1d() -> wp.grid.Grid:
    return wp.grid.Grid(wp.grid.PlaneWaveDof(1, 10, 6))


@pytest.fixture(scope="module")
def grid_2d() -> wp.grid.Grid:
    return wp.grid.Grid([wp.grid.PlaneWaveDof(1, 10, 6),
                         wp.grid.PlaneWaveDof(10, 10 + 7, 3)])
