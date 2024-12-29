import numpy as np
import pytest

import wavepacket as wp
import wavepacket.testing
from wavepacket.testing import assert_close


def test_reject_invalid_states(grid_1d, grid_2d):
    op = wp.CartesianKineticEnergy(grid_1d, 0, 1.0)
    eq = wp.SchroedingerEquation(op)
    psi = wp.testing.random_state(grid_1d, 42)

    rho = wp.pure_density(psi)
    with pytest.raises(wp.BadStateError):
        eq.apply(rho)

    bad_state = wp.testing.random_state(grid_2d, 1)
    with pytest.raises(wp.BadGridError):
        eq.apply(bad_state)


def test_equation(grid_1d):
    op = wp.Potential1D(grid_1d, 0, lambda x: 3 * np.ones(x.shape))
    eq = wp.SchroedingerEquation(op)
    psi = wp.testing.random_state(grid_1d, 42)

    result = eq.apply(psi)

    expected = -3j * psi
    assert_close(result, expected, 1e-12)
