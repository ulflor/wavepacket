import pytest
from numpy.testing import assert_allclose

import wavepacket as wp


def test_reject_invalid_arguments(grid_1d, grid_2d):
    op1 = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)
    op2 = wp.operator.CartesianKineticEnergy(grid_2d, 0, 1.0)

    with pytest.raises(wp.BadGridError):
        op1 + op2


def test_apply(grid_1d, monkeypatch):
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
