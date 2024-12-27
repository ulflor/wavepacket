import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.testing


def test_reject_invalid_arguments(grid_1d, grid_2d):
    op1 = wp.CartesianKineticEnergy(grid_1d, 0, 1.0)
    op2 = wp.CartesianKineticEnergy(grid_2d, 0, 1.0)

    with pytest.raises(wp.BadGridError):
        op1 + op2


def test_apply(grid_1d):
    op1 = wp.CartesianKineticEnergy(grid_1d, 0, 2.0)
    op2 = wp.CartesianKineticEnergy(grid_1d, 0, 5.0)

    sum_op = op1 + op2
    psi = wp.testing.random_state(grid_1d, 128)
    rho = wp.pure_density(psi)

    result = sum_op.apply_to_wave_function(psi.data)
    expected = op1.apply_to_wave_function(psi.data) + op2.apply_to_wave_function(psi.data)
    assert_allclose(result, expected, atol=1e-12, rtol=0)

    result = sum_op.apply_from_left(rho.data)
    expected = op1.apply_from_left(rho.data) + op2.apply_from_left(rho.data)
    assert_allclose(result, expected, atol=1e-12, rtol=0)

    result = sum_op.apply_from_right(rho.data)
    expected = op1.apply_from_right(rho.data) + op2.apply_from_right(rho.data)
    assert_allclose(result, expected, atol=1e-12, rtol=0)
