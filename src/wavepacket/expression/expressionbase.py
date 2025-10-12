from abc import ABC, abstractmethod
from typing import Optional

from ..grid import State


class ExpressionBase(ABC):
    """
    Base class for expressions.

    By deriving from this class and implementing the method
    :py:meth:`ExpressionBase.apply`, you can add custom expressions.

    Notes
    -----
    All differential equations have the form
    :math:`\dot \\rho = \mathcal{L}(\\rho)` (or equivalently
    :math:`\dot \psi = \hat H \psi`), that is, the left-hand side
    is just the time derivative. This matches the common convention for the
    Liouville von-Neumann equation, but differs from the usual
    form of the Schrödinger equation, where the imaginary factor
    is on the left-hand side of the equation.
    """

    @abstractmethod
    def apply(self, state: State, t: float) -> State:
        """
        Applies the expression to the input state and returns the result.

        Parameters
        ----------
        state : wp.grid.State
            The state on which the expression is applied.
        t : float
            The time at which the expression is evaluated.

        Raises
        ------
        wp.BadGridError
            If the grids of the state and the wrapped operator do not match.
        wp.BadStateError
            If the state is invalid or has the wrong time. For example, a Schroedinger equaiton
            makes little sense for a density operator.
        """
        pass
