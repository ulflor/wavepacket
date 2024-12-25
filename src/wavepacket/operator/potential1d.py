from .operatorbase import OperatorBase
from ..grid import Grid
from ..typing import ComplexData , RealGenerator


class Potential1D(OperatorBase):
    def __init__(self, grid: Grid, dof_index: int, generator: RealGenerator):
        data = generator(grid.dofs[dof_index].dvr_points)
        self._wf_data = grid.broadcast(data, dof_index)
        self._ket_data = grid.operator_broadcast(data, dof_index)
        self._bra_data = grid.operator_broadcast(data, dof_index, False)

    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        return self._wf_data * psi

    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        return self._ket_data * rho

    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        return self._bra_data * rho
