import numpy as np
import pytest
import wavepacket as wp

from numpy.testing import assert_allclose
from wavepacket.testing import random_state


@pytest.fixture
def grid() -> wp.Grid:
    return wp.Grid([wp.PlaneWaveDof(0, 10, 5),
                    wp.PlaneWaveDof(10, 10 + 7, 3)])


@pytest.fixture
def psi(grid) -> wp.State:
    return random_state(grid, 42)


def test_throw_on_bad_input_state(grid, psi):
    bad_input = wp.State(grid, np.ones(grid.operator_shape))

    with pytest.raises(wp.BadStateError):
        wp.pure_density(bad_input)

    with pytest.raises(wp.BadStateError):
        wp.direct_product(bad_input, psi)
    with pytest.raises(wp.BadStateError):
        wp.direct_product(psi, bad_input)


def test_require_all_grids_to_be_same(grid):
    grid2 = wp.grid.Grid(wp.PlaneWaveDof(0, 10, 5))

    wf1 = wp.State(grid, np.ones(grid.shape))
    wf2 = wp.State(grid2, np.ones(grid2.shape))

    with pytest.raises(wp.BadGridError):
        wp.direct_product(wf1, wf2)


def test_construct_pure_density(psi):
    size = psi.data.size

    rho = wp.pure_density(psi)

    assert rho.grid == psi.grid

    psi_data = np.reshape(psi.data, size)
    rho_data = np.reshape(rho.data, (size, size))
    for i in range(size):
        for j in range(size):
            assert_allclose(rho_data[i, j], psi_data[i] * psi_data[j].conjugate())


def test_direct_product(psi):
    bra = wp.State(psi.grid, psi.data + 1j)

    rho = wp.direct_product(psi, bra)

    assert rho.grid == psi.grid
    assert rho.is_density_operator()

    size = psi.data.size
    ket_data = np.reshape(psi.data, size)
    bra_data = np.reshape(bra.data, size)
    rho_data = np.reshape(rho.data, [size, size])
    for i in range(size):
        for j in range(size):
            assert_allclose(rho_data[i, j], ket_data[i] * bra_data[j].conjugate())
