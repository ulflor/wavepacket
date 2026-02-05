from collections.abc import Sequence
import math

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt


def product_wave_function(grid: wp.grid.Grid,
                          generators: wpt.Generator | Sequence[wpt.Generator],
                          normalize: bool = True) -> wp.grid.State:
    """
    Builds a product wave function from a set of one-dimensional wave functions.

    Parameters
    ----------
    grid : wp.grid.Grid
           The grid on which the product wave function is assembled
    generators : wpt.Generator | Sequence[wp.typing.Generator]
                One or more callables that specifies the wave function
                along each degree of freedom. The `generators` return
                the one-dimensional functions in the DVR, i.e., raw function
                values at the grid points.
    normalize : bool, default=true
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
    """
    generator_list = generators
    if not isinstance(generator_list, Sequence):
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

    result = wp.grid.State(grid, result_data)
    norm = np.sqrt(wp.grid.trace(result))
    if normalize and norm > 0:
        return result / norm
    else:
        return result


def random_wave_function(grid: wp.grid.Grid,
                         generator: np.random.Generator) -> wp.grid.State:
    """
    Generates a random wave function.

    The output is a state in the weighted DVR, whose coefficients in (unweighted) DVR
    are complex numbers uniformly distributed on the unit circle.

    Parameters
    ----------
    grid: wp.grid.Grid
        The grid for which the random wave function is created.
    generator: np.random.Generator
        The Numpy generator that creates the random values.

    Examples
    --------
    You typically need to generate several random wave functions for a simulation.
    In such a case, it is advantageous to recycle the random number generator after initial seeding.
    That way, a single seed number allows reproduction of the random numbers.

    >>> rng = np.random.default_rng(42)
    >>> psi = random_wave_function(grid, rng)
    >>> psi2 = random_wave_function(grid, rng)
    """
    data = np.ones(grid.shape) * np.exp(2 * math.pi * 1j * generator.random(grid.shape))
    for index, dof in enumerate(grid.dofs):
        data = dof.from_dvr(data, index)

    return wp.grid.State(grid, data)


def zero_wave_function(grid: wp.grid.Grid) -> wp.grid.State:
    """
    Returns a wave function whose coefficients are constant zero.

    These states sometimes occur as initial states in perturbation theory approaches.
    """
    return wp.grid.State(grid, np.zeros(grid.shape))
