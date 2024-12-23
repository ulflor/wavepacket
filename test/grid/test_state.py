import numpy as np
import pytest
import wavepacket as wp

from wavepacket.testing import assert_close, random_state


@pytest.fixture
def grid() -> wp.Grid:
    dofs = [wp.PlaneWaveDof(10, 20, 7), wp.PlaneWaveDof(1, 2, 3)]
    return wp.Grid(dofs)


def test_state_is_something(grid):
    psi = wp.State(grid, np.ones(grid.shape))
    rho = wp.State(grid, np.ones(grid.operator_shape))
    invalid = wp.State(grid, np.ones(17))

    assert psi.is_wave_function()
    assert not psi.is_density_operator()

    assert not rho.is_wave_function()
    assert rho.is_density_operator()

    assert not invalid.is_wave_function()
    assert not invalid.is_density_operator()


def test_operations_fail_on_different_grids(grid):
    psi = random_state(grid, 1)

    other_grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
    other_psi = random_state(other_grid, 2)

    with pytest.raises(wp.BadGridError):
        psi + other_psi

    with pytest.raises(wp.BadGridError):
        psi - other_psi


def test_operations_fail_on_different_types(grid):
    psi = random_state(grid, 1)
    rho = wp.State(grid, np.ones(grid.operator_shape))

    with pytest.raises(wp.BadStateError):
        psi + rho

    with pytest.raises(wp.BadStateError):
        psi - rho


def test_state_addition(grid):
    psi = random_state(grid, 1)
    two_j = wp.grid.State(grid, 2j * np.ones(grid.shape))

    result1 = psi + two_j
    result2 = psi + 2j
    result3 = 2j + psi

    assert result1.grid == grid
    assert abs(result1.data[0, 0] - psi.data[0, 0] - 2j) < 1e-12
    assert_close(result2, result1, 1e-12)
    assert_close(result3, result1, 1e-12)


def test_state_subtraction(grid):
    psi = random_state(grid, 2)
    minus_psi = wp.State(grid, -psi.data)
    two_j = wp.grid.State(grid, 2j * np.ones(grid.shape))

    result1 = psi - two_j
    result2 = psi - 2j
    result3 = 2j - minus_psi

    assert result1.grid == grid
    assert abs(result1.data[0, 0] - psi.data[0, 0] + 2j) < 1e-12
    assert_close(result2, result1, 1e-12)
    assert_close(result3, psi + 2j, 1e-12)


def test_state_multiplication(grid):
    psi = random_state(grid, 3)

    result1 = psi * 2.0
    result2 = 2.0 * psi

    assert_close(result1, psi + psi, 1e-12)
    assert_close(result2, psi + psi, 1e-12)


def test_state_division(grid):
    psi = random_state(grid, 4)

    result = psi / 4.0
    assert_close(result, 0.25 * psi, 1e-12)

    with pytest.raises(ZeroDivisionError):
        psi / 0.0


def test_state_unary_minus(grid):
    psi = random_state(grid, 5)
    zero = wp.State(grid, np.zeros(grid.shape))

    result = -psi

    expected = zero - psi
    assert_close(result, expected, 1e-12)
