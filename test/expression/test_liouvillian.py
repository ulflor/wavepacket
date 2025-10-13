import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


def test_reject_invalid_states(grid_1d, grid_2d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)
    psi = wp.testing.random_state(grid_1d, 42)
    bad_state = wp.testing.random_state(grid_2d, 1)

    commutator = wp.expression.CommutatorLiouvillian(op)
    with pytest.raises(wp.BadStateError):
        commutator.apply(psi, 0.0)
    with pytest.raises(wp.BadGridError):
        commutator.apply(bad_state, 0.0)

    one_sided = wp.expression.OneSidedLiouvillian(op)
    with pytest.raises(wp.BadStateError):
        one_sided.apply(psi, 0.0)
    with pytest.raises(wp.BadGridError):
        one_sided.apply(bad_state, 0.0)


def test_apply_commutator(grid_1d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 2.0)
    commutator = wp.expression.CommutatorLiouvillian(op)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.builder.pure_density(psi)

    dot_rho = commutator.apply(rho, 0.0)

    # Follows from the derivation of the Liouville-von-Neumann equation
    dot_psi = -1j * op.apply(psi, 0.0)
    expected = wp.builder.direct_product(dot_psi, psi) + wp.builder.direct_product(psi, dot_psi)
    assert_close(dot_rho, expected, 1e-12)


def test_apply_left_sided_liouvillian(grid_1d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)

    psi = wp.testing.random_state(grid_1d, 43)
    rho = wp.builder.pure_density(psi)
    op_psi = op.apply(psi, 0.0)

    left_sided = wp.expression.OneSidedLiouvillian(op)
    left = left_sided.apply(rho, 0.0)
    expected = wp.builder.direct_product(op_psi, psi)
    assert_close(left, expected, 1e-12)

    right_sided = wp.expression.OneSidedLiouvillian(op, wp.expression.OneSidedLiouvillian.Side.RIGHT)
    right = right_sided.apply(rho, 0.0)
    expected = wp.builder.direct_product(psi, op_psi)
    assert_close(right, expected, 1e-12)


def test_propagate_time_correctly(grid_1d, monkeypatch):
    td = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 2.0)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.builder.pure_density(psi)

    commutator = wp.expression.CommutatorLiouvillian(op)
    td_commutator = wp.expression.CommutatorLiouvillian(op * td)
    assert_close(7.0 * commutator.apply(rho, 0.0), td_commutator.apply(rho, 7.0), 1e-12)

    one_sided = wp.expression.OneSidedLiouvillian(op)
    td_one_sided = wp.expression.OneSidedLiouvillian(op * td)
    assert_close(6.0 * one_sided.apply(rho, 0.0), td_one_sided.apply(rho, 6.0), 1e-12)
