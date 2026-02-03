from enum import Enum

import wavepacket as wp

from .expressionbase import ExpressionBase
from ..grid import State
from ..operator import OperatorBase


class CommutatorLiouvillian(ExpressionBase):
    """
    Represents a commutator expression in a Liouville von-Neumann equation.

    Given an operator `H`, this commutator expression is given by
    :math:`\\mathcal{L}(\\hat \\rho) = -\\imath (\\hat H \\hat \\rho - \\hat \\rho \\hat H)`.

    Parameters
    ----------
    op : wp.operator.OperatorBase
        The operator to commute with the density operator.

    Notes
    -----
    The extra factor of -i is added to ensure that the commutator can be directly
    plugged into a Liouville von-Neumann equation. defined as
    :math:`\\frac{\\partial \\hat \\rho}{\\partial t} = \\mathcal{L}(\\hat \\rho)`.
    If you need the raw commutator, you have to multiply the result with
    the imaginary number.
    """

    def __init__(self, op: OperatorBase) -> None:
        self._op = op

    @property
    def time_dependent(self) -> bool:
        return self._op.time_dependent

    def apply(self, rho: State, t: float) -> State:
        if rho.grid != self._op.grid:
            raise wp.BadGridError("Input state is defined on the wrong grid.")

        if not rho.is_density_operator():
            raise wp.BadStateError("CommutatorLiouvillian requires a density operator.")

        return State(rho.grid,
                     -1j * (self._op.apply_from_left(rho.data, t) - self._op.apply_from_right(rho.data, t)))


class OneSidedLiouvillian(ExpressionBase):
    """
    An expression that simply applies an operator from the left or right of a density operator.

    This operator is basically the same as the :py:class:`wp.expression.SchroedingerEquation`
    but applies the operator onto a density operator, and it lacks the factor of "-1j".

    This expression is found occasionally in the context of open quantum systems or when relaxing
    a density operator to a thermal state.

    Parameters
    ----------
    op : wp.operator.OperatorBase
        The operator to apply with the density operator.
    side: wp.expression.OneSidedLiouvillian.Side
        If the operator should be applied from the LEFT or RIGHT of the density operator.
    """

    class Side(Enum):
        LEFT = 0
        RIGHT = 1

    def __init__(self, op: OperatorBase, side: Side = Side.LEFT) -> None:
        self._op = op
        self._side = side

    @property
    def time_dependent(self) -> bool:
        return self._op.time_dependent

    def apply(self, rho: State, t: float) -> State:
        if rho.grid != self._op.grid:
            raise wp.BadGridError("Input state is defined on the wrong grid.")

        if not rho.is_density_operator():
            raise wp.BadStateError("CommutatorLiouvillian requires a density operator.")

        if self._side == OneSidedLiouvillian.Side.LEFT:
            result = self._op.apply_from_left(rho.data, t)
            return State(rho.grid, result)
        else:
            result = self._op.apply_from_right(rho.data, t)
            return State(rho.grid, result)
