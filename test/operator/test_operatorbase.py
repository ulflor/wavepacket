import numpy as np
import pytest

import wavepacket as wp
import wavepacket.testing
import wavepacket.typing as wpt
from wavepacket.testing import assert_close


class DummyOperator(wp.OperatorBase):
    def __init__(self, grid: wp.Grid):
        super().__init__(grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData) -> wpt.ComplexData:
        return 2.0 * psi

    def apply_from_left(self, rho: wpt.ComplexData) -> wpt.ComplexData:
        return 3.0 * rho

    def apply_from_right(self, rho: wpt.ComplexData) -> wpt.ComplexData:
        return 5.0 * rho


@pytest.fixture
def op(grid_1d) -> DummyOperator:
    return DummyOperator(grid_1d)


def test_properties(grid_1d, op):
    assert op.grid == grid_1d


def test_reject_invalid_states(grid_1d, op):
    bad_state = wp.State(grid_1d, np.ones(1))
    with pytest.raises(wp.BadStateError):
        op.apply(bad_state)

    other_grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
    other_grid_state = wp.testing.random_state(other_grid, 1)
    with pytest.raises(wp.BadGridError):
        op.apply(other_grid_state)


def test_apply_operator(grid_1d, op):
    psi = wp.testing.random_state(grid_1d, 100)
    result = op.apply(psi)
    assert_close(result, 2.0 * psi, 1e-12)

    rho = wp.pure_density(psi)
    result = op.apply(rho)
    assert_close(result, 3.0 * rho, 1e-12)


def test_summation(grid_1d, op):
    psi = wp.testing.random_state(grid_1d, 42)
    sum_op = op + op

    result = op.apply(psi)
    sum_result = sum_op.apply(psi)

    assert_close(sum_result, 2 * result, 1e-12)
