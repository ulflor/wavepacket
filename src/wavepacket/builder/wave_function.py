from collections.abc import Sequence
from typing import Iterable

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from ..grid import Grid, State


def product_wave_function(grid: Grid,
                          generators: wpt.Generator | Sequence[wpt.Generator],
                          normalize: bool = True) -> State:
    """
    Builds a product wave function from a set of one-dimensional wave functions.

    Parameters
    ----------

    grid: wp.grid.Grid
           The grid on which the product wave function is assembled
    generators: Sequence[wp.typing.Generator]
                A list of callables that specifies the wave function
                along each degree of freedom. The `generators` return
                the one-dimensional functions in the DVR, i.e., raw function
                values at the grid points.
    normalize: bool, default=true
               If the norm is non-zero and this value is set, the resulting
               product wave function is normalized, otherwise the product
               is returned directly.

    Returns
    -------
    wp.grid.State
        The product wave function in the Wavepacket-default weighted DVR.

    Raises
    ------
    wp.InvalidValueError
        If the number of generators does not match the grid dimensions.

    numpy exceptions
        These may be thrown if the generators return wave functions
        with invalid shapes.
    """
    generator_list = generators
    if not isinstance(generator_list, Iterable):
        generator_list = [generators]

    if len(generator_list) != len(grid.dofs):
        raise wp.InvalidValueError(
            "To build a wave function, you need as many generators as degrees of freedoms."
            f"Given {len(generator_list)} generators for {len(grid.dofs)} DOFs.")

    result_data = np.ones(grid.shape, dtype=complex)
    for dof_index, generator in enumerate(generator_list):
        dof = grid.dofs[dof_index]
        array = generator(dof.dvr_points)
        array = dof.from_dvr(array, 0)

        result_data *= grid.broadcast(array, dof_index)

    result = State(grid, result_data)
    norm = np.sqrt(wp.grid.trace(result))
    if normalize and norm > 0:
        return result / norm
    else:
        return result
