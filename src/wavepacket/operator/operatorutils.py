import numpy as np

from .operatorbase import OperatorBase
from ..grid import State, trace


def expectation_value(op: OperatorBase, state: State) -> complex:
    new = op.apply(state)

    if state.is_wave_function():
        return np.vdot(state.data, new.data)
    else:
        return trace(new)
