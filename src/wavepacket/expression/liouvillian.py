import wavepacket as wp
from .expressionbase import ExpressionBase
from ..grid import State
from ..operator import OperatorBase


class CommutatorLiouvillian(ExpressionBase):
    """
    Represents a commutator expression in a Liouville von-Neumann equation.

    Given an operator `H`, this commutator expression is given by
    :math:`\mathcal{L}(\hat \\rho) = -\imath (\hat H \hat \\rho - \hat \\rho \hat H)`.

    Parameters
    ----------
    op: wp.operator.OperatorBase
        The operator to commute with the density operator.

    Notes
    -----
    The extra factor of -i is added to ensure that the commutator can be directly
    plugged into a Liouville von-Neumann equation. defined as
    :math:`\\frac{\partial \hat \\rho}{\partial t} = \mathcal{L}(\hat \\rho)`.
    If you need the raw commutator, you have to multiply the result with
    the imaginary number.
    """

    def __init__(self, op: OperatorBase):
        self._op = op

    def apply(self, rho: State, t: float) -> State:
        """
        Evaluates the commutator  for the given density operator and time.

        Parameters
        ----------
        rho: wp.grid.State
            The density operator to commute with
        t: float
            The time at which the operator is evaluated.

        Returns
        -------
        wp.grid.State
            The result of the commutator.

        Raises
        ------
        wp.BadGridError
            If the grids of the density operator and the wrapped operator do not match.
        wp.BadStateError
            If the input state is not a valid density operator.
        """
        if rho.grid != self._op.grid:
            raise wp.BadGridError("Input state is defined on the wrong grid.")

        if not rho.is_density_operator():
            raise wp.BadStateError("CommutatorLiouvillian requires a density operator.")

        return State(rho.grid,
                     -1j * (self._op.apply_from_left(rho.data, t) - self._op.apply_from_right(rho.data, t)))
