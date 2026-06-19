import math

import wavepacket as wp


def _truncate(value: float, truncation: float | None) -> float:
    if truncation is None or abs(value) >= truncation:
        return value
    else:
        return 0.0


def _print_expectation_values(state: wp.grid.State, precision: int, truncate: float) -> None:
    normalized_state = wp.normalize(state)
    for index, dof in enumerate(state.grid.dofs):
        x = wp.operator.Potential1D(state.grid, index, lambda dvr_grid: dvr_grid)

        x_val = wp.expectation_value(x, normalized_state).real
        x2_val = wp.expectation_value(x * x, normalized_state).real

        x_avg = _truncate(x_val, truncate)
        dx = _truncate(math.sqrt(x2_val - x_avg**2), truncate)

        # In exotic cases, the error dx**2 can become negative, so we trade
        # correctness for robustness here by taking its absolute value.
        print(f"    <x_{index}> = {x_avg:.{precision}}  +/- {dx:.{precision}}")


def log(
    t: float, state: wp.grid.State, precision: int = 6, truncate: float | None = None
) -> None:
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
    truncate: float | None, default=None
        If set, set all calculated values smaller than this boundary to zero.
        The main use case is for tests; values that are approximately zero can
        differ in their actual value based on numpy version etc. This causes
        regression tests to fail for uninteresting reasons.
    """
    print(
        f"\nt = {float(t):.{precision}},     trace = {wp.trace(state):.{precision}}\n"
        f"===================================================\n"
    )

    channel_dof = state.grid.get_single_channel_dof()
    if channel_dof is None:
        _print_expectation_values(state, precision, truncate)
    else:
        transform = wp.grid.ChannelProjectionTransformation(state.grid)
        for channel in range(channel_dof.size):
            channel_state = transform.transform(state, channel=channel)
            channel_trace = wp.trace(channel_state)

            if channel_trace == 0.0 or (truncate is not None and channel_trace < truncate):
                continue

            print(
                f"\n  channel '{channel_dof.names[channel]}',      trace = {channel_trace:.{precision}}"
                "\n  ---------------------------------------------------\n"
            )
            _print_expectation_values(channel_state, precision, truncate)
