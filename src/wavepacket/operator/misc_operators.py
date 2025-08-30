import numpy as np
from typing import Sequence

import wavepacket as wp
import wavepacket.typing as wpt

from .operatorbase import OperatorBase
from ..grid import State, trace, orthonormalize


class Projection(OperatorBase):
    """
    Projects onto a subspace defined by one or more basis functions.

    This class represents an idempotent projection operator,
    :math:`\\hat P = \\sum_i |\\psi_i\\rangle\\langle\\psi_i|`,
    where the basis functions :math:`\\psi_i` span the subspace on
    which the operator projects.

    The basis functions need not be orthogonal, they are orthonormalized
    using :func:`wp.grid.orthonormalize`.

    Parameters
    ----------
    basis: wp.grid.State | Sequence[wp.grid.State]
      The basis functions :math:`\\psi_i` that span the subspace onto which the operator projects.
      Need not be orthogonal, but should be linearly independent.
      In case of one basis function, it can be supplied directly without a list.

    Raises
    ------
    wp.BadStateException
        Thrown if any basis function is not a wave function or has norm zero.

    wp.InvalidValueError
        Thrown if no basis functions are supplied.

    See Also
    --------
    wavepacket.grid.population: if you only want to calculate the population of some states.
    """

    def __init__(self, basis: State | Sequence[State]):
        if isinstance(basis, wp.grid.State):
            basis = [basis]

        if not basis:
            raise wp.InvalidValueError("Projection operator requires at least one state to project onto.")

        for state in basis:
            if not state.is_wave_function():
                raise wp.BadStateError("Can only project onto wave functions.")

            if trace(state) == 0:
                raise wp.BadStateError("Basis functions must not have norm zero.")

        orthonormal_basis = orthonormalize(basis)

        self._ket_nd = np.stack([s.data for s in orthonormal_basis])
        self._bra_nd = np.conj(self._ket_nd)
        self._ket_ravelled = np.reshape(self._ket_nd, (len(basis), basis[0].grid.size))
        self._bra_ravelled = np.conj(self._ket_ravelled)
        super().__init__(basis[0].grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.reshape(psi, self._grid.size)
        coefficients = np.tensordot(self._bra_ravelled, tmp, axes=(1, 0))
        return np.tensordot(self._ket_nd, coefficients, axes=(0, 0))

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        matrix_form = np.reshape(rho, (self._grid.size, self._grid.size))
        coefficients = np.tensordot(self._bra_ravelled, matrix_form, (1, 0))
        ket_projection = np.tensordot(self._ket_nd, coefficients, (0, 0))
        return np.reshape(ket_projection, self._grid.operator_shape)

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        matrix_form = np.reshape(rho, (self._grid.size, self._grid.size))
        coefficients = np.tensordot(matrix_form, self._ket_ravelled, (1, 1))
        bra_projection = np.tensordot(coefficients, self._bra_nd, (1, 0))
        return np.reshape(bra_projection, self._grid.operator_shape)
