from abc import ABC, abstractmethod

from ..grid import State


class ExpressionBase(ABC):
    """
    Base class for expressions.

    By deriving from this class and implementing the method
    :py:meth:`ExpressionBase.apply`, you can add custom expressions.

    Notes
    -----
    All differential equations have the form
    :math:`\dot x = E[x]`, where x is a density operator or
    wave function. Note that this differs from the common
    Schrödinger equation, we have moved the imaginary factor
    to the right-hand side of the equation.
    """

    @abstractmethod
    def apply(self, state: State, t: float) -> State:
        """
        Applies the expression at time `t` to the input state and returns the result.
        """
        pass
