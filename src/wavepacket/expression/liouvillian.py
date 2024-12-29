import wavepacket as wp
from .expressionbase import ExpressionBase
from ..grid import State
from ..operator import OperatorBase


class CommutatorLiouvillian(ExpressionBase):
    def __init__(self, op: OperatorBase):
        self._op = op

    def apply(self, rho: State) -> State:
        if rho.grid != self._op.grid:
            raise wp.BadGridError("Input state is defined on the wrong grid.")

        if not rho.is_density_operator():
            raise wp.BadStateError("CommutatorLiouvillian requires a density operator.")

        return State(rho.grid,
                     -1j * (self._op.apply_from_left(rho.data) - self._op.apply_from_right(rho.data)))