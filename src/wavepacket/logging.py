import math
import numbers

from .grid import State, trace
from .operator import Potential1D, expectation_value


def log(t: numbers.Real, state: State, precision: int = 6) -> None:
    """
    Prints some data about the state for inspection.

    The idea is that you call this function during every solver step and get
    a log with the most important values about the propagation, for example
    the state trace (if it deviates from one, this may be caused by poor convergence).

    Parameters
    ----------
    t: float
        The time at which you log.
    state: wp.grid.State
        The state to log.
    precision: int, default=6
        How many decimal places should be printed.
    """
    print(f"\n\nt = {float(t):.{precision}},     trace = {trace(state):.{precision}}\n")

    for index, dof in enumerate(state.grid.dofs):
        x = Potential1D(state.grid, 0, lambda dvr_grid: dvr_grid)
        x2 = Potential1D(state.grid, 0, lambda dvr_grid: dvr_grid ** 2)

        x_expect = expectation_value(x, state).real
        x_expect2 = expectation_value(x2, state).real

        print(f"<x_{index}> = {x_expect:.{precision}}  =/- {math.sqrt(x_expect2 - x_expect ** 2):.{precision}}")
