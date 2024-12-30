import numpy as np
import pytest

import wavepacket as wp
import wavepacket.testing
from wavepacket.testing import assert_close


@pytest.fixture
def op(grid_1d) -> wp.testing.DummyOperator:
    return wp.testing.DummyOperator(grid_1d)


def test_properties(grid_1d, op):
    assert op.grid == grid_1d


def test_reject_invalid_states(grid_1d, op):
    bad_state = wp.State(grid_1d, np.ones(1))
    with pytest.raises(wp.BadStateError):
        op.apply(bad_state, 0.0)

    other_grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
    other_grid_state = wp.testing.random_state(other_grid, 1)
    with pytest.raises(wp.BadGridError):
        op.apply(other_grid_state, 0.0)


def test_apply_operator(grid_1d, op):
    psi = wp.testing.random_state(grid_1d, 100)
    t = 7.0
    result = op.apply(psi, t)
    assert_close(result, 2.0 * t * psi, 1e-12)

    rho = wp.pure_density(psi)
    result = op.apply(rho, t)
    assert_close(result, 3.0 * t * rho, 1e-12)
