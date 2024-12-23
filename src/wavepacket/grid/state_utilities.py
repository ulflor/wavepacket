import numpy as np

from .state import State
from ..typing import RealData
from ..utils import BadStateError


def dvr_density(state: State) -> RealData:
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

        matrix_form = np.reshape(data, (grid.size, grid.size))
        density = np.abs(np.diag(matrix_form))  # the diagonal is real up to numerical issues
        return np.reshape(density, grid.shape)
    else:
        raise BadStateError("Input is not a valid state.")
