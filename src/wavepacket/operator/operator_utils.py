from typing import Optional

import numpy as np

import wavepacket as wp
from .operatorbase import OperatorBase
from ..grid import State


def expectation_value(op: OperatorBase, state: State, t: Optional[float] = None) -> complex:
    """
    Calculates the expectation value of an operator for a given state.

    Parameters
    ----------
    op : wp.operator.OperatorBase
        The operator whose expectation value is calculated.
    state : wp.grid.State
        The wave function or density operator that is used for the calculation.
    t : float, optional
        The time at which the operator should be evaluated.
         Only required for time-dependent operators.

    Raises
    ------
    InvalidValueError
        If a time-dependent operator was supplied, but no time value was given.
    """
    if op.time_dependent and t is None:
        raise wp.InvalidValueError("You must supply a time value for time-dependent operators.")

    new = op.apply(state, t)

    if state.is_wave_function():
        return np.vdot(state.data, new.data)
    else:
        matrix_data = np.reshape(new.data, [new.grid.size, new.grid.size])
        return np.trace(matrix_data)
