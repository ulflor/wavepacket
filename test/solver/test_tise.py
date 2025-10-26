import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


def _standard_setup() -> wp.operator.OperatorBase:
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(-10, 10, 5),
                         wp.grid.SphericalHarmonicsDof(6, 1)])
    return (wp.operator.PlaneWaveFbrOperator(grid, 0, lambda p: p)
            + wp.operator.RotationalKineticEnergy(grid, 1, 10))


def test_return_eigenstates_and_values():
    op = _standard_setup()

    result = [x for x in wp.solver.diagonalize(op)]
    assert len(result) == op.grid.size

    # energies have no duplicates and are sorted
    energies = [e for (e, s) in result]
    for i in range(len(energies) - 1):
        assert energies[i + 1] - energies[i] > 1e-3

    # states are normal (duplicates are checked below)
    states = [s for (e, s) in result]
    for s in states:
        assert abs(wp.grid.trace(s) - 1) < 1e-12

    # results are eigenvectors
    for energy, state in result:
        applied = op.apply(state, 0.0)
        expected = energy * state

        assert_close(applied, expected, 1e-12)


def test_degenerate_states(grid_1d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)

    energies = [e for e, _ in wp.solver.diagonalize(op)]

    # every state except the first and possibly last is doubly degenerate
    assert abs(energies[0]) < 1e-12
    assert abs(energies[1] - energies[2]) < 1e-12


def test_time_forwarding(grid_1d):
    op = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)

    energies_t2 = [e for e, _ in wp.solver.diagonalize(op, 2.0)]

    assert abs(energies_t2[0] - 2.0) < 1e-12
