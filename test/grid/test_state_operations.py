import pytest
import numpy as np
import wavepacket as wp
from wavepacket.testing import assert_close

@pytest.fixture
def grid() -> wp.grid.Grid:
    return wp.grid.Grid([wp.grid.plane_wave_dof(0, 10, 5),
                         wp.grid.plane_wave_dof(10, 10 + 7, 6)])


def random_state(grid) -> wp.grid.State:
    rng = np.random.default_rng()
    data = rng.random(grid.shape) + 1j * rng.random(grid.shape)

    return wp.grid.State(grid, data)


def test_throw_on_different_grids(grid):
    other_grid = wp.grid.Grid(wp.grid.plane_wave_dof(0, 10, 30))

    psi1 = random_state(grid)
    psi2 = random_state(other_grid)

    with pytest.raises(wp.BadGridError):
        psi1 + psi2

    with pytest.raises(wp.BadGridError):
        psi1 - psi2


def test_throw_on_different_states(grid):
    psi = random_state(grid)
    rho = wp.builder.pure_density(random_state(grid))

    with pytest.raises(wp.BadStateError):
        psi + rho

    with pytest.raises(wp.BadStateError):
        psi - rho


def test_state_addition(grid):
    psi = random_state(grid)
    two_j = wp.grid.State(grid, 2j * np.ones(grid.shape))

    result1 = psi + two_j
    result2 = psi + 2j
    result3 = 2j + psi

    assert result1.grid == grid
    assert_close(result2, result1, 1e-12)
    assert_close(result3, result1, 1e-12)


def test_state_subtraction(grid):
    psi = random_state(grid)
    two_j = wp.grid.State(grid, 2j * np.ones(grid.shape))

    result1 = psi - two_j
    result2 = psi - 2j
    result3 = 2j - psi

    assert result1.grid == grid
    assert_close(result2, result1, 1e-12)
    assert_close(result3, (-1) * result1, 1e-12)


def test_state_multiplication(grid):
    psi = random_state(grid)

    result1 = psi * 2j
    result2 = 2j * psi

    assert result1.grid == grid
    assert_close(result2, result1, 1e-12)


def test_state_division(grid):
    psi = random_state(grid)

    result = psi / 2j
    expected = -0.5j * psi

    assert expected.grid == grid
    assert_close(result, expected, 1e-12)
