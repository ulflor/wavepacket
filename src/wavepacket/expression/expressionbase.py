from abc import ABC, abstractmethod
from typing import Final, Iterable, override

import numpy as np

import wavepacket as wp


class ExpressionBase(ABC):
    """
    Base class for expressions.

    By deriving from this class and implementing the method
    :py:meth:`ExpressionBase.apply`, you can add custom expressions.

    Parameters
    ----------
    time_dependent: bool
        Sets whether the expression is time-dependent or not.

    Attributes
    ----------
    time_dependent: bool, readonly
        If the expression is time-dependent or not.

        Some functionality, for example special solvers, may require time-independent expressions.
        This property typically evaluates the time-dependence of the underlying operator(s).

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

    def __init__(self, time_dependent: bool) -> None:
        self.time_dependent: Final[bool] = time_dependent

    def __add__(self, other: 'ExpressionBase') -> 'ExpressionBase':
        return ExpressionSum([self, other])

    @abstractmethod
    def apply(self, state: wp.grid.State, t: float) -> wp.grid.State:
        """
        Applies the expression to the input state and returns the result.

        Parameters
        ----------
        state : wp.grid.State
            The state on which the expression is applied.
        t : float
            The time at which the expression is evaluated.
            Only really needed for time-dependent expressions, but to keep the interface uniform,
            this parameter is required.

        Raises
        ------
        wp.BadGridError
            If the grids of the state and the wrapped operator do not match.
        wp.BadStateError
            If the state is invalid or has the wrong time. For example, a Schrödinger equation
            makes little sense for a density operator.
        """
        raise NotImplementedError()


class ExpressionSum(ExpressionBase):
    """
    Sum of multiple expressions.

    You should rarely construct this class directly. It is created implicitly if you
    sum two or more expressions. Note that it is perfectly possible to construct an
    invalid sum, for example, by adding a SchroedingerEquation to some Liouvillian;
    the result will not accept any state without exceptions.

    Parameters
    ----------
    expressions : Iterable[wp.operator.ExpressionBase]
        The expressions that should be summed up.

    Raises
    ------
    wp.InvalidValueError
        if the list of expressions is empty.
    """

    def __init__(self, expressions: Iterable[ExpressionBase]) -> None:
        if not expressions:
            raise wp.InvalidValueError("ExpressionSum requires an expression.")

        self._expressions = list(expressions)

        td_vals = [expr.time_dependent for expr in expressions]
        super().__init__(any(td_vals))

    @override
    def apply(self, state: wp.grid.State, t: float) -> wp.grid.State:
        result = wp.grid.State(state.grid, np.zeros(state.data.shape))
        for expression in self._expressions:
            result = result + expression.apply(state, t)

        return result
