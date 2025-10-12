import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


def test_reject_invalid_states(grid_1d, grid_2d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)
    commutator = wp.expression.CommutatorLiouvillian(op)
    psi = wp.testing.random_state(grid_1d, 42)

    with pytest.raises(wp.BadStateError):
        commutator.apply(psi, 0.0)

    bad_state = wp.testing.random_state(grid_2d, 1)
    with pytest.raises(wp.BadGridError):
        commutator.apply(bad_state, 0.0)


def test_apply_commutator(grid_1d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 2.0)
    eq = wp.expression.SchroedingerEquation(op)
    commutator = wp.expression.CommutatorLiouvillian(op)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.builder.pure_density(psi)

    dot_rho = commutator.apply(rho, 0.0)

    # Follows from the derivation of the Liouville-von-Neumann equation
    dot_psi = eq.apply(psi, 0.0)
    expected = wp.builder.direct_product(dot_psi, psi) + wp.builder.direct_product(psi, dot_psi)
    assert_close(dot_rho, expected, 1e-12)


def test_propagate_time_correctly(grid_1d, monkeypatch):
    td = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 2.0)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.builder.pure_density(psi)

    commutator = wp.expression.CommutatorLiouvillian(op)
    td_commutator = wp.expression.CommutatorLiouvillian(op * td)

    assert_close(7.0 * commutator.apply(rho, 0.0), td_commutator.apply(rho, 7.0), 1e-12)
