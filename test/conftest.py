import pytest

import wavepacket as wp


@pytest.fixture(scope="module")
def grid_1d() -> wp.Grid:
    return wp.Grid(wp.PlaneWaveDof(1, 10, 6))


@pytest.fixture(scope="module")
def grid_2d() -> wp.Grid:
    return wp.Grid([wp.PlaneWaveDof(1, 10, 6),
                    wp.PlaneWaveDof(10, 10 + 7, 3)])
