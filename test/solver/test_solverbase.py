import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


class DummySolver(wp.solver.SolverBase):
    def __init__(self, dt):
        super().__init__(dt)

    def step(self, state: wp.grid.State, t: float) -> wp.grid.State:
        return state + 1


def test_throw_on_invalid_timestep():
    with pytest.raises(wp.InvalidValueError):
        DummySolver(-1)

    with pytest.raises(wp.InvalidValueError):
        DummySolver(0)


def test_return_timestep():
    dt = 1.4
    solver = DummySolver(dt)

    assert solver.dt == dt


def test_propagate_edge_cases(grid_1d):
    solver = DummySolver(2.0)
    psi0 = wp.testing.random_state(grid_1d, 1)

    with pytest.raises(wp.InvalidValueError):
        for _ in solver.propagate(psi0, 0.0, -1):
            pass

    results = [x for x in solver.propagate(psi0, 0.0, 0)]
    assert len(results) == 1
    assert results[0][0] == 0.0
    assert_close(results[0][1], psi0)

    results = [x for x in solver.propagate(psi0, 0.0, 0, False)]
    assert len(results) == 0


def test_propagation(grid_1d):
    solver = DummySolver(2.0)
    psi0 = wp.testing.random_state(grid_1d, 1)

    results = [x for x in solver.propagate(psi0, 0.0, 3)]
    times = [x[0] for x in results]
    psis = [x[1] for x in results]

    assert times == [0.0, 2.0, 4.0, 6.0]
    for i in range(4):
        assert_close(psis[i], psi0 + i)
