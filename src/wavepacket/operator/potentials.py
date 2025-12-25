import wavepacket.typing as wpt

from .operatorbase import OperatorBase
from ._clipping import clip_real
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
    cutoff: float, default=None
        If set, defines the maximum potential value. Values larger than the cutoff are set to the cutoff.
        For complex potentials, only the real part is truncated.

    References
    ----------
    .. [1] https://sourceforge.net/p/wavepacket/wiki/Numerics.DVR>
    """

    def __init__(self, grid: Grid, dof_index: int, generator: wpt.RealGenerator,
                 cutoff: float = None):
        data = generator(grid.dofs[dof_index].dvr_points).copy()

        if cutoff is not None:
            data = clip_real(data, upper=cutoff)

        self._wf_data = grid.broadcast(data, dof_index)
        self._ket_data = grid.operator_broadcast(data, dof_index)
        self._bra_data = grid.operator_broadcast(data, dof_index, False)

        super().__init__(grid)

    @property
    def time_dependent(self) -> bool:
        return False

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._wf_data * psi

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._ket_data * rho

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._bra_data * rho
