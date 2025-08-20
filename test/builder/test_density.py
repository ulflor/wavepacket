import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import wavepacket as wp


@pytest.fixture
def psi(grid_2d) -> wp.grid.State:
    return wp.testing.random_state(grid_2d, 42)


def test_throw_on_bad_input_state(grid_2d, psi):
    bad_input = wp.grid.State(grid_2d, np.ones(grid_2d.operator_shape))

    with pytest.raises(wp.BadStateError):
        wp.builder.pure_density(bad_input)

    with pytest.raises(wp.BadStateError):
        wp.builder.direct_product(bad_input, psi)
    with pytest.raises(wp.BadStateError):
        wp.builder.direct_product(psi, bad_input)


def test_require_all_grids_to_be_same(grid_2d):
    grid2 = wp.grid.Grid(wp.grid.PlaneWaveDof(0, 10, 5))

    wf1 = wp.grid.State(grid_2d, np.ones(grid_2d.shape))
    wf2 = wp.grid.State(grid2, np.ones(grid2.shape))

    with pytest.raises(wp.BadGridError):
        wp.builder.direct_product(wf1, wf2)


def test_construct_pure_density(psi):
    size = psi.data.size

    rho = wp.builder.pure_density(psi)

    assert rho.grid == psi.grid

    psi_data = np.reshape(psi.data, size)
    rho_data = np.reshape(rho.data, (size, size))
    for i in range(size):
        for j in range(size):
            assert_allclose(rho_data[i, j], psi_data[i] * psi_data[j].conjugate(), atol=1e-12, rtol=0)


def test_direct_product(psi):
    bra = wp.grid.State(psi.grid, psi.data + 1j)

    rho = wp.builder.direct_product(psi, bra)

    assert rho.grid == psi.grid
    assert rho.is_density_operator()

    size = psi.data.size
    ket_data = np.reshape(psi.data, size)
    bra_data = np.reshape(bra.data, size)
    rho_data = np.reshape(rho.data, [size, size])
    for i in range(size):
        for j in range(size):
            assert_allclose(rho_data[i, j], ket_data[i] * bra_data[j].conjugate(), atol=1e-12, rtol=0)


def test_unit_density(grid_2d):
    rho = wp.builder.unit_density(grid_2d)
    assert rho.is_density_operator()

    matrix = np.reshape(rho.data, [grid_2d.size, grid_2d.size])
    assert_array_equal(np.eye(grid_2d.size), matrix)


def test_zero_density(grid_2d):
    rho = wp.builder.zero_density(grid_2d)

    assert rho.is_density_operator()
    assert np.all(rho.data == 0)
