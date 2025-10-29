from abc import ABC, abstractmethod
from typing import Self, Sequence

import numpy as np

import wavepacket as wp
from ..grid import State


class ExpressionBase(ABC):
    """
    Base class for expressions.

    By deriving from this class and implementing the method
    :py:meth:`ExpressionBase.apply`, you can add custom expressions.

    Attributes
    ----------
    time_dependent: bool
        If the expression is time-dependent or not. Some functionality may not work for time-dependent operators.

    Notes
    -----
    All differential equations have the form
    :math:`\\dot \\rho = \\mathcal{L}(\\rho)` (or equivalently
    :math:`\\dot \\psi = \\hat H \\psi`), that is, the left-hand side
    is just the time derivative. This matches the common convention for the
    Liouville von-Neumann equation, but differs from the usual
    form of the Schrödinger equation, where the imaginary factor
    is on the left-hand side of the equation.
    """

    def __add__(self, other: Self):
        return ExpressionSum([self, other])

    @property
    @abstractmethod
    def time_dependent(self) -> bool:
        """
        Returns if the expression is time-dependent or not.

        Some functionality, for example special solvers, may require time-independent expressions.
        This property typically evaluates the time-dependence of the underlying operator(s).
        """
        pass

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
            If the state is invalid or has the wrong time. For example, a Schrödinger equation
            makes little sense for a density operator.
        """
        pass


class ExpressionSum(ExpressionBase):
    """
    Sum of multiple expressions.

    You should rarely construct this class directly. It is created implicitly if you
    sum two or more expressions. Note that it is perfectly possible to construct an
    invalid sum, for example, by adding a SchroedingerEquation to some Liouvillian;
    the result will not accept any state without exceptions.

    Parameters
    ----------
    expressions : Sequence[wp.operator.ExpressionBase]
        The expressions that should be summed up.

    Raises
    ------
    wp.InvalidValueError
        if the list of expressions is empty.
    """

    def __init__(self, expressions: Sequence[ExpressionBase]):
        if not expressions:
            raise wp.InvalidValueError("ExpressionSum requires an expression.")

        self._expressions = expressions

    @property
    def time_dependent(self) -> bool:
        td_vals = [expr.time_dependent for expr in self._expressions]
        return any(td_vals)

    def apply(self, state: State, t: float) -> State:
        result = wp.grid.State(state.grid, np.zeros(state.data.shape))
        for expression in self._expressions:
            result = result + expression.apply(state, t)

        return result
