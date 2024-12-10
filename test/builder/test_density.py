import numpy as np
import pytest
import wavepacket as wp
from numpy.testing import assert_allclose


@pytest.fixture
def grid() -> wp.grid.Grid:
    return wp.grid.Grid([wp.grid.plane_wave_dof(0, 10, 5),
                         wp.grid.plane_wave_dof(10, 10 + 7, 3)])


@pytest.fixture
def psi(grid) -> wp.grid.State:
    rng = np.random.default_rng()
    data = rng.random(grid.shape) + 1j * rng.random(grid.shape)

    return wp.grid.State(grid, data)


def test_throw_on_bad_input_state(grid, psi):
    bad_input = wp.grid.State(grid, np.ones(grid.operator_shape))

    with pytest.raises(wp.BadStateError):
        wp.builder.pure_density(bad_input)

    with pytest.raises(wp.BadStateError):
        wp.builder.direct_product(bad_input, psi)
    with pytest.raises(wp.BadStateError):
        wp.builder.direct_product(psi, bad_input)


def test_require_all_grids_to_be_same(grid):
    grid2 = wp.grid.Grid(wp.grid.plane_wave_dof(0, 10, 5))

    wf1 = wp.grid.State(grid, np.ones(grid.shape))
    wf2 = wp.grid.State(grid2, np.ones(grid2.shape))

    with pytest.raises(wp.BadGridError):
        wp.builder.direct_product(wf1, wf2)


def test_construct_pure_density(psi: wp.grid.State):
    size = psi.data.size

    rho = wp.builder.pure_density(psi)

    assert rho.grid == psi.grid

    psi_data = np.reshape(psi.data, size)
    rho_data = np.reshape(rho.data, (size, size))
    for i in range(size):
        for j in range(size):
            assert_allclose(rho_data[i, j], psi_data[i] * psi_data[j].conjugate())


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
            assert_allclose(rho_data[i, j], ket_data[i] * bra_data[j].conjugate())
