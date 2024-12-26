from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from ..grid import Grid, State


class OperatorBase(ABC):
    def __init__(self, grid: Grid):
        self._grid = grid

    def apply(self, state: State) -> State:
        if state.grid != self._grid:
            raise wp.BadGridError("Grid of state does not match grid of operator.")

        if state.is_wave_function():
            return State(state.grid, self.apply_to_wave_function(state.data))
        elif state.is_density_operator():
            return State(state.grid, self.apply_from_left(state.data))
        else:
            raise wp.BadStateError("Cannot apply the operator to an invalid state.")

    @property
    def grid(self):
        return self._grid

    def __add__(self, other):
        return OperatorSum([self, other])

    @abstractmethod
    def apply_to_wave_function(self, psi: wpt.ComplexData) -> wpt.ComplexData:
        pass

    @abstractmethod
    def apply_from_left(self, rho: wpt.ComplexData) -> wpt.ComplexData:
        pass

    @abstractmethod
    def apply_from_right(self, rho: wpt.ComplexData) -> wpt.ComplexData:
        pass


class OperatorSum(OperatorBase):
    def __init__(self, ops: Sequence[OperatorBase]):
        if not ops:
            raise wp.InvalidValueError("OperatorSum needs at least one operator to sum.")
        for op in ops:
            if op.grid != ops[0].grid:
                raise wp.BadGridError("All grids in a sum operator must be equal.")

        self._ops = ops
        grid = ops[0].grid
        super().__init__(grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData) -> wpt.ComplexData:
        result = np.zeros(self.grid.shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_to_wave_function(psi)

        return result

    def apply_from_left(self, rho: wpt.ComplexData) -> wpt.ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_left(rho)

        return result

    def apply_from_right(self, rho: wpt.ComplexData) -> wpt.ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_right(rho)

        return result
