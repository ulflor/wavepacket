from abc import ABC, abstractmethod

from ..grid import Grid, State
from ..typing import ComplexData
from ..utils import BadGridError, BadStateError


class OperatorBase(ABC):
    def __init__(self, grid: Grid):
        self._grid = grid

    def apply(self, state: State) -> State:
        if state.grid != self._grid:
            raise BadGridError("Grid of state does not match grid of operator.")

        if state.is_wave_function():
            return State(state.grid, self.apply_to_wave_function(state.data))
        elif state.is_density_operator():
            return State(state.grid, self.apply_from_left(state.data))
        else:
            raise BadStateError("Cannot apply the operator to an invalid state.")

    @property
    def grid(self):
        return self._grid

    @abstractmethod
    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        pass

    @abstractmethod
    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        pass

    @abstractmethod
    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        pass
