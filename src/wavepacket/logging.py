import math

import wavepacket as wp


def log(t: float, state: wp.grid.State, precision: int = 6) -> None:
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
    print(f"\n-----------------------------------------------\n"
          f"t = {float(t):.{precision}},     trace = {wp.trace(state):.{precision}}\n")

    normalized_state = wp.normalize(state)
    for index, dof in enumerate(state.grid.dofs):
        x = wp.operator.Potential1D(state.grid, 0, lambda dvr_grid: dvr_grid)

        x_avg = wp.expectation_value(x, normalized_state).real
        x2_avg = wp.expectation_value(x * x, normalized_state).real

        # In exotic cases, the error dx**2 can become negative, so we trade
        # correctness for robustness here by taking its absolute value.
        print(f"<x_{index}> = {x_avg:.{precision}}  =/- {math.sqrt(abs(x2_avg - x_avg ** 2)):.{precision}}")
