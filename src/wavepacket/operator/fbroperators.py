import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from .operatorbase import OperatorBase
from ..grid import Grid


class PlaneWaveFbrOperator(OperatorBase):
    """
    Base class for operators that are diagonal in a plane wave basis.

    The :py:class:`wavepacket.grid.PlaneWaveDof` describes an expansion of
    the wave function in plane waves. In the corresponding FBR, derivatives
    are diagonal and essentially transform into a multiplication with the
    wave vector / FBR grid.

    A key difference to a generic FBR operator is that this operator
    does not apply the shift after the FFT, which increases performance.

    Parameters
    ----------
    grid : wp.grid.Grid
        The grid on which the operator is defined.
    dof_index : int
        The degree of freedom along which the operator is defined.
    generator : wpt.Generator
        A callable that gives the operator value for each FBR point.

    Raises
    ------
    wp.InvalidValueError
        If the supplied degree of freedom is not a plane wave expansion.
    """

    def __init__(self, grid: Grid, dof_index: int, generator: wpt.Generator):
        if not isinstance(grid.dofs[dof_index], wp.grid.PlaneWaveDof):
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
    """
    Convenience class that implements the common Cartesian kinetic energy operator.

    The form of the operator is :math:`-\\frac{1}{2m} \ \\frac{\partial^2}{\partial x^2}`.
    It requires the degree of freedom to be a :py:class:`wavepacket.grid.PlaneWaveDof`.

    Parameters
    ----------
    grid : wp.grid.Grid
        The grid on which the operator is defined.
    dof_index : inst
        Degree of freedom along which the operator acts
    mass : float
        The mass of the particle.

    Raises
    ------
    wp.InvalidValueError
        If the mass is not positive, or if the degree of freedom does not describe a
        plane wave expansion.
    """

    def __init__(self, grid: Grid, dof_index: int, mass: float):
        if mass <= 0:
            raise wp.InvalidValueError(f"Particle mass must be positive, but is {mass}")

        super().__init__(grid, dof_index, lambda fbr_points: fbr_points ** 2 / (2 * mass))
