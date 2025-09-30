import pytest
from numpy.testing import assert_allclose

import wavepacket as wp


def test_reject_invalid_arguments(grid_1d, grid_2d):
    op1 = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)
    op2 = wp.operator.CartesianKineticEnergy(grid_2d, 0, 1.0)

    with pytest.raises(wp.BadGridError):
        op1 + op2

    with pytest.raises(wp.BadGridError):
        op1 * op2


def test_apply_sum(grid_1d, monkeypatch):
    op = wp.testing.DummyOperator(grid_1d)

    sum_op = op + op
    t = 7.0
    psi = wp.testing.random_state(grid_1d, 128)
    rho = wp.builder.pure_density(psi)

    monkeypatch.setattr(op,
                        "apply_to_wave_function", lambda data, _t: _t * data)
    result = sum_op.apply_to_wave_function(psi.data, t)
    expected = 2 * op.apply_to_wave_function(psi.data, t)
    assert_allclose(result, expected, atol=1e-12, rtol=0)

    monkeypatch.setattr(op,
                        "apply_from_left", lambda data, _t: 2.0 * _t * data)
    result = sum_op.apply_from_left(rho.data, t)
    expected = 2 * op.apply_from_left(rho.data, t)
    assert_allclose(result, expected, atol=1e-12, rtol=0)

    monkeypatch.setattr(op,
                        "apply_from_right", lambda data, _t: 3.0 * _t * data)
    result = sum_op.apply_from_right(rho.data, t)
    expected = 2 * op.apply_from_right(rho.data, t)
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_apply_product(grid_1d):
    # We expect these operators to not commute
    op1 = wp.operator.Potential1D(grid_1d, 0, lambda x: x ** 2)
    op2 = wp.operator.FbrOperator1D(grid_1d, 0, lambda x: x)

    product = op1 * op2

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.builder.pure_density(psi)

    psi_result = product.apply_to_wave_function(psi.data, 0.0)
    psi_expected = op1.apply_to_wave_function(op2.apply_to_wave_function(psi.data, 0.0), 0.0)
    assert_allclose(psi_expected, psi_result, atol=1e-12, rtol=0)

    rho_result_left = product.apply_from_left(rho.data, 0.0)
    rho_expected_left = op1.apply_from_left(op2.apply_from_left(rho.data, 0.0), 0.0)
    assert_allclose(rho_expected_left, rho_result_left, atol=1e-12, rtol=0)

    rho_result_right = product.apply_from_right(rho.data, 0.0)
    rho_expected_right = op2.apply_from_right(op1.apply_from_right(rho.data, 0.0), 0.0)
    assert_allclose(rho_expected_right, rho_result_right, atol=1e-12, rtol=0)
