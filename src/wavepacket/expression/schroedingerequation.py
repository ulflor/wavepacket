from typing import override

import wavepacket as wp
from .expressionbase import ExpressionBase
from ..grid import State
from ..operator import OperatorBase


class SchroedingerEquation(ExpressionBase):
    """
    Expression wrapper for a Schrödinger equation.

    You should wrap a Hamiltonian in an object of this type
    so that the solvers can subsequently solve the resulting
    equation.

    Parameters
    ----------
    hamiltonian : wp.operator.OperatorBase
        The Hamiltonian that is wrapped by this class.

    Notes
    -----
    The Schrödinger equation is given by
    :math:`\dot \psi = -\imath \hat H \psi`,
    so this class only multiplies the wave function with the
    negative imaginary number, and wraps the input Hamiltonian
    into an expression so that solvers can work with it.
    """

    def __init__(self, hamiltonian: OperatorBase) -> None:
        self._hamiltonian = hamiltonian

        super().__init__(hamiltonian.time_dependent)

    @override
    def apply(self, psi: State, t: float) -> State:
        if psi.grid != self._hamiltonian.grid:
            raise wp.BadGridError("Input state has wrong grid.")

        if not psi.is_wave_function():
            raise wp.BadStateError("SchroedingerEquation requires a wave function.")

        return State(psi.grid, -1j * self._hamiltonian.apply_to_wave_function(psi.data, t))
