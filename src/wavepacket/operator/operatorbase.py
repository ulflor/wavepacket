import numpy as np

from abc import ABC, abstractmethod
from typing import Sequence

from ..grid import Grid, State
from ..typing import ComplexData
from ..utils import BadGridError, BadStateError, InvalidValueError


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

    def __add__(self, other):
        return OperatorSum([self, other])

    @abstractmethod
    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        pass

    @abstractmethod
    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        pass

    @abstractmethod
    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        pass


class OperatorSum(OperatorBase):
    def __init__(self, ops: Sequence[OperatorBase]):
        if not ops:
            raise InvalidValueError("OperatorSum needs at least one operator to sum.")
        for op in ops:
            if op.grid != ops[0].grid:
                raise BadGridError("All grids in a sum operator must be equal.")

        self._ops = ops
        grid = ops[0].grid
        super().__init__(grid)

    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        result = np.zeros(self.grid.shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_to_wave_function(psi)

        return result

    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_left(rho)

        return result

    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_right(rho)

        return result
