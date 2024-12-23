import numpy as np
import wavepacket as wp
import pytest

from numpy.testing import assert_allclose
from wavepacket.testing import random_state


@pytest.fixture
def grid() -> wp.Grid:
    dofs = [wp.PlaneWaveDof(1, 2, 3), wp.PlaneWaveDof(1, 3, 5)]
    return wp.Grid(dofs)


def test_reject_invalid_states(grid):
    invalid_state = wp.State(grid, np.ones(5))

    with pytest.raises(wp.BadStateError):
        wp.dvr_density(invalid_state)


def test_wave_function_density(grid):
    psi = random_state(grid, 42)

    result = wp.dvr_density(psi)

    dx = 1 / 3.0 * 2 / 5.0
    expected = np.abs(psi.data ** 2) / dx
    assert_allclose(result, expected, atol=1e-14)


def test_density_operator_density(grid):
    psi = random_state(grid, 42)
    rho = wp.pure_density(psi)

    density_from_psi = wp.dvr_density(psi)
    density_from_rho = wp.dvr_density(rho)

    assert_allclose(density_from_rho, density_from_psi, atol=1e-14)
