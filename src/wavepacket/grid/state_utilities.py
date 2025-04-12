import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from .grid import Grid
from .state import State


def _take_diagonal(data: wpt.ComplexData, grid: Grid) -> wpt.RealData:
    # note: the diagonal should be real and positive but for numerical issues
    matrix_form = np.reshape(data, (grid.size, grid.size))
    diagonal = np.abs(np.diag(matrix_form))
    return np.reshape(diagonal, grid.shape)


def dvr_density(state: State) -> wpt.RealData:
    """
    Returns the density of the input state at the DVR grid points.

    The density is returned as a real-valued coefficient array
    with the same shape as the underlying grid. The density is mostly a
    dead end: useful for plotting and inspection, but not for
    further computations.

    Parameters
    ----------
    state : wp.grid.State
        The state (wave function or density operator) whose density should be computed.

    Returns
    -------
    wpt.RealData
        The coefficient array of the density values at the DVR grid points.

    Raises
    ------
    wp.BadStateError
        If the supplied state is neither a wave function nor a density operator.
    """
    if state.is_wave_function():
        data = state.data
        for index, dof in enumerate(state.grid.dofs):
            data = dof.to_dvr(data, index)

        return np.abs(data * data)
    elif state.is_density_operator():
        data = state.data
        grid = state.grid

        # both, the bra and the ket indices are converted
        for index, dof in enumerate(grid.dofs):
            data = dof.to_dvr(data, index)
            data = dof.to_dvr(data, index + len(grid.dofs))

        return _take_diagonal(data, grid)
    else:
        raise wp.BadStateError("Input is not a valid state.")


def trace(state: State) -> float:
    """
    Returns the trace of the supplied input state.

    For density operators, this is the usual trace norm, i.e., sum over the
    diagonal elements. For wave functions, it is the square of the usual 
    L2 norm.

    Parameters
    ----------
    state : wp.grid.State
        The state (wave function or density operator) for which to calculate the trace.

    Returns
    -------
    float
        The trace of the input state.

    Raises
    ------
    wp.BadStateError
        If the supplied state is neither a wave function nor a density operator.
    """
    if state.is_wave_function():
        return np.abs(state.data ** 2).sum()
    elif state.is_density_operator():
        diagonal = _take_diagonal(state.data, state.grid)
        return diagonal.sum()
    else:
        raise wp.BadStateError("Input is not a valid state.")
