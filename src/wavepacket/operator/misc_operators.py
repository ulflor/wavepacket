from collections.abc import Iterable
from typing import Final, override

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt

from .operatorbase import OperatorBase


class Projection(OperatorBase):
    """
    Projects onto a subspace defined by one or more basis functions.

    This class represents an idempotent projection operator,
    :math:`\\hat P = \\sum_i |\\psi_i\\rangle\\langle\\psi_i|`,
    where the basis functions :math:`\\psi_i` span the subspace on
    which the operator projects.

    The basis functions need not be orthogonal, they are orthonormalized
    using :func:`wavepacket.orthonormalize`.

    Parameters
    ----------
    basis: wp.grid.State | Sequence[wp.grid.State]
      The basis functions :math:`\\psi_i` that span the subspace onto which the operator projects.
      Need not be orthogonal, but should be linearly independent.
      In case of one basis function, it can be supplied directly without a list.

    Raises
    ------
    wp.BadStateException
        Raised if any basis function is not a wave function or has norm zero.

    wp.InvalidValueError
        Raised if no basis functions are supplied.

    wp.BadGridError
        Raised if the input states are not defined on the same grid.

    See Also
    --------
    wavepacket.population: if you only want to calculate the population of some states.
    """

    def __init__(self, basis: wp.grid.State | Iterable[wp.grid.State]) -> None:
        if isinstance(basis, wp.grid.State):
            basis_list = [basis]
        else:
            basis_list = list(basis)

        if not basis_list:
            raise wp.InvalidValueError(
                "Projection operator requires at least one state to project onto."
            )

        for state in basis_list:
            if not state.is_wave_function():
                raise wp.BadStateError("Can only project onto wave functions.")

            if wp.trace(state) == 0:
                raise wp.BadStateError("Basis functions must not have norm zero.")

        orthonormal_basis = wp.orthonormalize(basis_list)

        self._ket_nd = np.stack([s.data for s in orthonormal_basis])
        self._bra_nd = np.conj(self._ket_nd)
        self._ket_ravelled = np.reshape(
            self._ket_nd, (len(basis_list), basis_list[0].grid.size)
        )
        self._bra_ravelled = np.conj(self._ket_ravelled)
        super().__init__(basis_list[0].grid, False)

    @override
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.reshape(psi, self.grid.size)
        coefficients = np.tensordot(self._bra_ravelled, tmp, axes=(1, 0))
        return np.tensordot(self._ket_nd, coefficients, axes=(0, 0))

    @override
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        matrix_form = np.reshape(rho, (self.grid.size, self.grid.size))
        coefficients = np.tensordot(self._bra_ravelled, matrix_form, (1, 0))
        ket_projection = np.tensordot(self._ket_nd, coefficients, (0, 0))
        return np.reshape(ket_projection, self.grid.operator_shape)

    @override
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        matrix_form = np.reshape(rho, (self.grid.size, self.grid.size))
        coefficients = np.tensordot(matrix_form, self._ket_ravelled, (1, 1))
        bra_projection = np.tensordot(coefficients, self._bra_nd, (1, 0))
        return np.reshape(bra_projection, self.grid.operator_shape)


class Constant(OperatorBase):
    """
    Operator that encapsulates a simple constant.

    Normally, you will not use this operator directly. It is only a wrapper to enable numbers
    where an operator is expected, in particular when doing operator arithmetic.

    Parameters
    ----------
    grid: wp.grid.Grid
        The grid on which this operator is defined.
    value: complex
        The value that this operator wraps.

    Attributes
    ----------
    value: complex, readonly
        The wrapped value.
    """

    def __init__(self, grid: wp.grid.Grid, value: complex) -> None:
        self.value: Final[complex] = value
        super().__init__(grid, False)

    @override
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self.value * psi

    @override
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self.value * rho

    @override
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self.value * rho


class TensorOperator(OperatorBase):
    """
    Operator that wraps around a general tensor.

    This is one of the most general, but also slowest and most awkward operator.
    It essentially wraps around an operator matrix that is multiplied with a state
    vector or density matrix.

    In most cases, you do not set up such an operator manually, but it can be the
    result, for example, when transforming an operator into a subspace.

    Parameters
    ----------
    grid: wavepacket.grid.Grid
        The grid on which this operator is defined.
    tensor: wavepacket.testing.ComplexData
        The operator matrix in tensor form. A_{i1,i2, ... , j1,j2, ...} defines the
        mapping from the DVR basis function at (x_j1, x_j2, ...) to that at
        (x_i1, x_i2, ...).

    Raises
    ------
    wavepacket.InvalidValueError
        Raised if the input tensor has the wrong shape.
    """

    def __init__(self, grid: wp.grid.Grid, tensor: wpt.ComplexData) -> None:
        if tensor.shape != grid.operator_shape:
            raise wp.InvalidValueError("Input must be shaped like an operator.")

        self._matrix = np.reshape(tensor.copy(), (grid.size, grid.size))

        super().__init__(grid, False)

    @override
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.reshape(psi, self.grid.size)
        result = np.tensordot(self._matrix, tmp, axes=(1, 0))
        return np.reshape(result, self.grid.shape)

    @override
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.reshape(rho, (self.grid.size, self.grid.size))
        result = np.tensordot(self._matrix, tmp, axes=(1, 0))
        return np.reshape(result, self.grid.operator_shape)

    @override
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.reshape(rho, (self.grid.size, self.grid.size))
        result = np.tensordot(tmp, self._matrix, axes=(1, 0))
        return np.reshape(result, self.grid.operator_shape)
