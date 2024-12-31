import pytest

import wavepacket as wp
from wavepacket import State


class DummySolver(wp.SolverBase):
    def __init__(self, dt):
        super().__init__(dt)

    def step(self, state: State, t: float) -> State:
        pass


def test_throw_on_invalid_timestep():
    with pytest.raises(wp.InvalidValueError):
        DummySolver(-1)

    with pytest.raises(wp.InvalidValueError):
        DummySolver(0)


def test_return_timestep():
    dt = 1.4
    solver = DummySolver(dt)

    assert solver.dt == dt
