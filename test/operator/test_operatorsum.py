import wavepacket as wp
import pytest
from numpy.testing import assert_allclose
from wavepacket.testing import random_state


def test_reject_invalid_arguments():
    grid1 = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
    op1 = wp.CartesianKineticEnergy(grid1, 0, 1.0)

    grid2 = wp.Grid(wp.PlaneWaveDof(2, 3, 4))
    op2 = wp.CartesianKineticEnergy(grid2, 0, 1.0)

    with pytest.raises(wp.BadGridError):
        op1 + op2


def test_apply():
    grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
    op1 = wp.CartesianKineticEnergy(grid, 0, 2.0)
    op2 = wp.CartesianKineticEnergy(grid, 0, 5.0)

    sum_op = op1 + op2
    psi = random_state(grid, 128)
    rho = wp.pure_density(psi)

    result = sum_op.apply_to_wave_function(psi.data)
    expected = op1.apply_to_wave_function(psi.data) + op2.apply_to_wave_function(psi.data)
    assert_allclose(result, expected, atol=1e-12)

    result = sum_op.apply_from_left(rho.data)
    expected = op1.apply_from_left(rho.data) + op2.apply_from_left(rho.data)
    assert_allclose(result, expected, atol=1e-12)

    result = sum_op.apply_from_right(rho.data)
    expected = op1.apply_from_right(rho.data) + op2.apply_from_right(rho.data)
    assert_allclose(result, expected, atol=1e-12)
