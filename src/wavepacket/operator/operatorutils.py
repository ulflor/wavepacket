import numpy as np

import wavepacket as wp
from .operatorbase import OperatorBase
from ..grid import State


def expectation_value(op: OperatorBase, state: State, t: float = None) -> complex:
    # Default: t = None, no time-dependence
    new = op.apply(state, t)

    if state.is_wave_function():
        return np.vdot(state.data, new.data)
    else:
        matrix_data = np.reshape(new.data, [new.grid.size, new.grid.size])
        return np.trace(matrix_data)
