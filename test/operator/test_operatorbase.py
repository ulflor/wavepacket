import numpy as np
import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


def test_properties(grid_1d):
    op = wp.testing.DummyOperator(grid_1d)

    assert op.grid == grid_1d


def test_reject_invalid_states(grid_1d, monkeypatch):
    op = wp.testing.DummyOperator(grid_1d)

    bad_state = wp.grid.State(grid_1d, np.ones(1))
    with pytest.raises(wp.BadStateError):
        op.apply(bad_state, 0.0)

    other_grid = wp.grid.Grid(wp.grid.PlaneWaveDof(1, 2, 3))
    other_grid_state = wp.testing.random_state(other_grid, 1)
    with pytest.raises(wp.BadGridError):
        op.apply(other_grid_state, 0.0)


def test_apply_operator(grid_1d, monkeypatch):
    op = wp.testing.DummyOperator(grid_1d)
    monkeypatch.setattr(op,
                        "apply_to_wave_function", lambda data, _t: 2.0 * _t * data)
    monkeypatch.setattr(op,
                        "apply_from_left", lambda data, _t: 3.0 * _t * data)

    psi = wp.testing.random_state(grid_1d, 100)
    t = 7.0
    result = op.apply(psi, t)
    assert_close(result, 2.0 * t * psi, 1e-12)

    rho = wp.builder.pure_density(psi)
    result = op.apply(rho, t)
    assert_close(result, 3.0 * t * rho, 1e-12)
