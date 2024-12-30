import pytest

import wavepacket as wp
import wavepacket.testing
from wavepacket.testing import assert_close


def test_reject_invalid_states(grid_1d, grid_2d):
    op = wp.testing.DummyOperator(grid_1d)
    eq = wp.SchroedingerEquation(op)
    psi = wp.testing.random_state(grid_1d, 42)

    rho = wp.pure_density(psi)
    with pytest.raises(wp.BadStateError):
        eq.apply(rho, 0.0)

    bad_state = wp.testing.random_state(grid_2d, 1)
    with pytest.raises(wp.BadGridError):
        eq.apply(bad_state, 0.0)


def test_equation(grid_1d, monkeypatch):
    op = wp.testing.DummyOperator(grid_1d)
    eq = wp.SchroedingerEquation(op)
    psi = wp.testing.random_state(grid_1d, 42)

    def apply(state, time):
        return time * state

    monkeypatch.setattr(op, 'apply_to_wave_function', apply)

    t = 17
    result = eq.apply(psi, t)

    assert_close(result, -1j * apply(psi, t), 1e-12)
