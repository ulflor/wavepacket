import math
import numpy as np
from typing import Sequence

import wavepacket as wp
import wavepacket.typing as wpt

from .operatorbase import OperatorBase
from ..grid import State, trace


class Projection(OperatorBase):
    """
    Projects onto a given wave function.

    This class represents an idempotent projection operator,
    :math:`\\hat P = |\\psi\\rangle\\langle\\psi|`.

    Parameters
    ----------
    state: wp.grid.State
      The wave function :math:`\\psi` onto which we project the input.
      Does not need to be normalized.

    Raises
    ------
    wp.BadStateException
      Thrown if the state is not a wave function or has norm zero.
    """

    def __init__(self, state: State | Sequence[State]):
        if not state.is_wave_function():
            raise wp.BadStateError("Can only project onto a wave function.")

        if trace(state) == 0:
            raise wp.BadStateError("Wave function to project on must not have norm zero.")

        self._ket = state.data / math.sqrt(trace(state))
        self._bra = np.conj(self._ket)
        self._ket_vector = np.ravel(self._ket)
        self._bra_vector = np.ravel(self._bra)
        super().__init__(state.grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        coefficient = (self._bra * psi).sum()
        return coefficient * self._ket

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        matrix_form = np.reshape(rho, (self._grid.size, self._grid.size))
        coefficients = np.tensordot(self._bra_vector, matrix_form, (0, 0))
        return np.tensordot(self._ket, coefficients, ((), ()))

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        matrix_form = np.reshape(rho, (self._grid.size, self._grid.size))
        coefficients = np.tensordot(matrix_form, self._ket_vector, (1, 0))
        return np.tensordot(coefficients, self._bra, ((), ()))
