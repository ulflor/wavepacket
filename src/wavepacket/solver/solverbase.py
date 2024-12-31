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
