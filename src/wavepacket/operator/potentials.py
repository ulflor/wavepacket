import wavepacket.typing as wpt
from .operatorbase import OperatorBase
from ..grid import Grid


class Potential1D(OperatorBase):
    """
    Operator that represents a real-valued one-dimensional potential.

    Within the DVR approximation [1]_, potential energy operators are
    diagonal in the DVR. That is, they are completely described by a value
    for each grid point.

    Parameters
    ----------
    grid : wp.grid.Grid
        The grid on which the operator is defined
    dof_index : int
        the index of the degree of freedom along which the potential is defined.
    generator : wpt.RealGenerator
        A callable that generates a potential energy value for each grid point of the respective DOF.

    References
    ----------
    .. [1] https://sourceforge.net/p/wavepacket/wiki/Numerics.DVR>
    """

    def __init__(self, grid: Grid, dof_index: int, generator: wpt.RealGenerator):
        data = generator(grid.dofs[dof_index].dvr_points)
        self._wf_data = grid.broadcast(data, dof_index)
        self._ket_data = grid.operator_broadcast(data, dof_index)
        self._bra_data = grid.operator_broadcast(data, dof_index, False)

        super().__init__(grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._wf_data * psi

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._ket_data * rho

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._bra_data * rho
