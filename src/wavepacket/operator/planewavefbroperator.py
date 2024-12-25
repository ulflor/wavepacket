import numpy as np

from .operatorbase import OperatorBase
from ..grid import Grid, PlaneWaveDof
from ..typing import ComplexData, Generator
from ..utils import InvalidValueError


class PlaneWaveFbrOperator(OperatorBase):
    def __init__(self, grid: Grid, dof_index: int, generator: Generator):
        if not isinstance(grid.dofs[dof_index], PlaneWaveDof):
            raise InvalidValueError(
                f"PlaneWaveFbrOperator requires a PlaneWaveDof, but got {grid.dofs[dof_index].__class__}")

        self._wf_index = dof_index
        self._ket_index = grid.normalize_index(dof_index)
        self._bra_index = self._ket_index + len(grid.dofs)

        # shifting the data here allows us to skip the fftshift on the input data in apply*()
        data = generator(grid.dofs[dof_index].fbr_points)
        shifted_data = np.fft.ifftshift(data)
        self._wf_data = grid.broadcast(shifted_data, dof_index)
        self._ket_data = grid.operator_broadcast(shifted_data, dof_index)
        self._bra_data = grid.operator_broadcast(shifted_data, dof_index, is_ket=False)

    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        psi_fft = np.fft.fft(psi, axis=self._wf_index)
        return np.fft.ifft(psi_fft * self._wf_data, axis=self._wf_index)

    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        rho_fft = np.fft.fft(rho, axis=self._ket_index)
        return np.fft.ifft(rho_fft * self._ket_data, axis=self._ket_index)

    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        rho_fft = np.fft.ifft(rho, axis=self._bra_index)
        return np.fft.fft(rho_fft * self._bra_data, axis=self._bra_index)
