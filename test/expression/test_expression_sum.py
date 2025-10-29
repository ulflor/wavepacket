import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


def test_sum_requires_an_expression():
    with pytest.raises(wp.InvalidValueError):
        wp.expression.ExpressionSum([])


def test_forward_time_dependence(grid_1d):
    td = wp.expression.SchroedingerEquation(wp.operator.TimeDependentOperator(grid_1d, lambda t: t))
    ti = wp.expression.SchroedingerEquation(wp.operator.Constant(grid_1d, 1))

    assert not (ti + ti).time_dependent
    assert (ti + td).time_dependent


def test_forward_to_individual_liouvillians(grid_1d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1)
    td_op = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)
    psi = wp.testing.random_state(grid_1d, 5)

    eq_sum = wp.expression.SchroedingerEquation(op) + wp.expression.SchroedingerEquation(td_op)

    expected = -1j * (op + td_op).apply(psi, 5.0)
    got = eq_sum.apply(psi, 5.0)
    assert_close(got, expected, 1e-12)
