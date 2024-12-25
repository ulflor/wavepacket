import numpy as np
import pytest
import wavepacket as wp

from wavepacket.testing import random_state, assert_close
from wavepacket.typing import ComplexData


class DummyOperator(wp.OperatorBase):
    def __init__(self, grid: wp.Grid):
        super().__init__(grid)

    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        return 2.0 * psi

    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        return 3.0 * rho

    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        return 5.0 * rho


grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
op = DummyOperator(grid)


def test_properties():
    assert op.grid == grid


def test_reject_invalid_states():
    bad_state = wp.State(grid, np.ones(1))
    with pytest.raises(wp.BadStateError):
        op.apply(bad_state)

    other_grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
    other_grid_state = random_state(other_grid, 1)
    with pytest.raises(wp.BadGridError):
        op.apply(other_grid_state)


def test_apply_operator():
    psi = random_state(grid, 100)
    result = op.apply(psi)
    assert_close(result, 2.0 * psi, 1e-12)

    rho = wp.pure_density(psi)
    result = op.apply(rho)
    assert_close(result, 3.0 * rho, 1e-12)
