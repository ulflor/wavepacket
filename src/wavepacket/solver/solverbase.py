from abc import ABC, abstractmethod

import wavepacket as wp
from ..grid import State


class SolverBase(ABC):
    def __init__(self, dt: float):
        if dt <= 0:
            raise wp.InvalidValueError(f"Require positive timestep, got {dt}")

        self._dt = dt

    @property
    def dt(self) -> float:
        return self._dt

    @abstractmethod
    def step(self, state: State, t: float) -> State:
        pass

    def propagate(self, state0: State, t0: float,
                  num_steps: int, include_first: bool = True):
        if num_steps < 0:
            raise wp.InvalidValueError("Cannot propagate for negative number of steps.")

        if include_first:
            yield t0, state0

        state = state0
        for step in range(num_steps):
            t = t0 + step * self._dt
            state = self.step(state, t)
            yield t + self._dt, state
