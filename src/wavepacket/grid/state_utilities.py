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
    if state.is_wave_function():
        return np.sum(np.abs(state.data ** 2))
    elif state.is_density_operator():
        diagonal = _take_diagonal(state.data, state.grid)
        return np.sum(diagonal)
    else:
        raise wp.BadStateError("Input is not a valid state.")
