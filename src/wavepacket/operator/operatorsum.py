import numpy as np

from typing import Sequence

from .operatorbase import OperatorBase
from ..typing import ComplexData
from ..utils import BadGridError, InvalidValueError


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
