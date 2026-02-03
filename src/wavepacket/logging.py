import math

from .grid import State, trace
from .operator import Potential1D, expectation_value


def log(t: float, state: State, precision: int = 6) -> None:
    """
    Prints some data about the state for inspection.

    The idea is that you call this function during every solver step and get
    a log with the most important values about the propagation, for example
    the state trace (if it deviates from one, this may be caused by poor convergence).

    Parameters
    ----------
    t : float
        The time at which you log.
    state : wp.grid.State
        The state to log.
    precision : int, default=6
        How many decimal places should be printed.
    """
    print(f"\n\nt = {float(t):.{precision}},     trace = {trace(state):.{precision}}\n")

    for index, dof in enumerate(state.grid.dofs):
        x = Potential1D(state.grid, 0, lambda dvr_grid: dvr_grid)

        x_avg = expectation_value(x, state).real
        x2_avg = expectation_value(x * x, state).real

        # In exotic cases, the error dx**2 can become negative, so we trade
        # correctness for robustness here by taking its absolute value.
        print(f"<x_{index}> = {x_avg:.{precision}}  =/- {math.sqrt(abs(x2_avg - x_avg ** 2)):.{precision}}")
