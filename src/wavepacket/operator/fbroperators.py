import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from .operatorbase import OperatorBase
from ..grid import Grid


class PlaneWaveFbrOperator(OperatorBase):
    def __init__(self, grid: Grid, dof_index: int, generator: wpt.Generator):
        if not isinstance(grid.dofs[dof_index], wp.PlaneWaveDof):
            raise wp.InvalidValueError(
                f"PlaneWaveFbrOperator requires a PlaneWaveDof, but got {grid.dofs[dof_index].__class__}")

        self._wf_index = dof_index
        self._ket_index = grid.normalize_index(dof_index)
        self._bra_index = self._ket_index + len(grid.dofs)

        # shifting the data here allows us to skip the fftshift() on the input data in apply*()
        data = generator(grid.dofs[dof_index].fbr_points)
        shifted_data = np.fft.ifftshift(data)
        self._wf_data = grid.broadcast(shifted_data, dof_index)
        self._ket_data = grid.operator_broadcast(shifted_data, dof_index)
        self._bra_data = grid.operator_broadcast(shifted_data, dof_index, is_ket=False)

        super().__init__(grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        psi_fft = np.fft.fft(psi, axis=self._wf_index)
        return np.fft.ifft(psi_fft * self._wf_data, axis=self._wf_index)

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        rho_fft = np.fft.fft(rho, axis=self._ket_index)
        return np.fft.ifft(rho_fft * self._ket_data, axis=self._ket_index)

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        rho_fft = np.fft.ifft(rho, axis=self._bra_index)
        return np.fft.fft(rho_fft * self._bra_data, axis=self._bra_index)


class CartesianKineticEnergy(PlaneWaveFbrOperator):
    def __init__(self, grid: Grid, dof_index: int, mass: float):
        if mass <= 0:
            raise wp.InvalidValueError(f"Particle mass must be positive, but is {mass}")

        super().__init__(grid, dof_index, lambda fbr_points: fbr_points ** 2 / (2 * mass))
