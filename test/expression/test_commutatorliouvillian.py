import pytest

import wavepacket as wp
import wavepacket.testing


def test_reject_invalid_states(grid_1d, grid_2d):
    op = wp.CartesianKineticEnergy(grid_1d, 0, 1.0)
    liouvillian = wp.CommutatorLiouvillian(op)
    psi = wp.testing.random_state(grid_1d, 42)

    with pytest.raises(wp.BadStateError):
        liouvillian.apply(psi, 0.0)

    bad_state = wp.testing.random_state(grid_2d, 1)
    with pytest.raises(wp.BadGridError):
        liouvillian.apply(bad_state, 0.0)


def test_apply_commutator(grid_1d):
    op = wp.CartesianKineticEnergy(grid_1d, 0, 2.0)
    eq = wp.SchroedingerEquation(op)
    liouvillian = wp.CommutatorLiouvillian(op)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.pure_density(psi)

    dot_rho = liouvillian.apply(rho, 0.0)

    # Follows from the derivation of the Liouville-von-Neumann equation
    dot_psi = eq.apply(psi, 0.0)
    expected = wp.direct_product(dot_psi, psi) + wp.direct_product(psi, dot_psi)
    wp.testing.assert_close(dot_rho, expected, 1e-12)


def test_propagate_time_correctly(grid_1d, monkeypatch):
    t = 42.0

    def check(data, time):
        assert time == t
        return data

    op = wp.testing.DummyOperator(grid_1d)
    monkeypatch.setattr(op, "apply_from_left", check)
    monkeypatch.setattr(op, "apply_from_right", check)

    eq = wp.CommutatorLiouvillian(op)
    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.pure_density(psi)
    eq.apply(rho, t)

