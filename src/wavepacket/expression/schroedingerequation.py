import wavepacket as wp
from .expressionbase import ExpressionBase
from ..grid import State
from ..operator import OperatorBase


class SchroedingerEquation(ExpressionBase):
    def __init__(self, op: OperatorBase):
        self._op = op

    def apply(self, psi: State, t: float) -> State:
        if psi.grid != self._op.grid:
            raise wp.BadGridError("Input state has wrong grid.")

        if not psi.is_wave_function():
            raise wp.BadStateError("SchroedingerEquation requires a wave function.")

        return wp.State(psi.grid, -1j * self._op.apply_to_wave_function(psi.data, t))
