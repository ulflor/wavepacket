import numpy as np
import pytest
import wavepacket as wp

from wavepacket.testing import random_state, assert_close
from wavepacket.typing import ComplexData


class DummyOperator(wp.OperatorBase):
    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        return 2.0 * psi

    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        return 3.0 * rho

    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        return 5.0 * rho


grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
op = DummyOperator()


def test_reject_bad_state():
    bad_state = wp.State(grid, np.ones(1))
    with pytest.raises(wp.BadStateError):
        op.apply(bad_state)


def test_apply_operator():
    psi = random_state(grid, 100)
    result = op.apply(psi)
    assert_close(result, 2.0 * psi, 1e-12)

    rho = wp.pure_density(psi)
    result = op.apply(rho)
    assert_close(result, 3.0 * rho, 1e-12)
