import numpy as np

import wavepacket as wp
from .operatorbase import OperatorBase
from ..grid import State


def expectation_value(op: OperatorBase, state: State, t: float = None) -> complex:
    """
    Calculates the expectation value of an operator given a state.

    Parameters
    ----------
    op: wp.operator.OperatorBase
        The operator whose expectation value is calculated.
    state: wp.grid.State
        The wave function or density operator that is used for the calculation.
    t: float, default=None
        The time at which the operator should be evaluated. Ignored for
        time-independent operators.
    """
    new = op.apply(state, t)

    if state.is_wave_function():
        return np.vdot(state.data, new.data)
    else:
        matrix_data = np.reshape(new.data, [new.grid.size, new.grid.size])
        return np.trace(matrix_data)
