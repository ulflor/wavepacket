import math

import numpy as np
from typing import Sequence

import wavepacket as wp
import wavepacket.typing as wpt
from .grid import Grid
from .state import State


def _take_diagonal(data: wpt.ComplexData, grid: Grid) -> wpt.RealData:
    # note: the diagonal should be real and positive but for numerical issues
    matrix_form = np.reshape(data, (grid.size, grid.size))
    diagonal = np.abs(np.diag(matrix_form))
    return np.reshape(diagonal, grid.shape)


def _normalize(u: wpt.ComplexData) -> wpt.ComplexData:
    return u / math.sqrt(np.abs(u ** 2).sum())


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


def fbr_density(state: State) -> wpt.RealData:
    """
    Returns the FBR density of the input state at the FBR grid points.

    The density is returned as a real-valued coefficient array
    with the same shape as the underlying grid. The density is mostly a
    dead end: useful for plotting and inspection, but not for
    further computations.

    Note that this function returns the FBR density, i.e., the coefficients
    or diagonal when expanding the wave function or density operator in the
    underlying grid's basis.

    Parameters
    ----------
    state : wp.grid.State
        The state (wave function or density operator) whose FBR density should be computed.

    Returns
    -------
    wpt.RealData
        The coefficient array of the FBR density values at the FBR grid points.

    Raises
    ------
    wp.BadStateError
        If the supplied state is neither a wave function nor a density operator.
    """
    if state.is_wave_function():
        data = state.data
        for index, dof in enumerate(state.grid.dofs):
            data = dof.to_fbr(data, index)

        return np.abs(data * data)
    elif state.is_density_operator():
        data = state.data
        grid = state.grid

        # both, the bra and the ket indices are converted
        for index, dof in enumerate(grid.dofs):
            data = dof.to_fbr(data, index)
            data = dof.to_fbr(data, index + len(grid.dofs), is_ket=False)

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


def orthonormalize(states: Sequence[State]) -> list[State]:
    """
    Orthogonalizes and normalizes a set of linearly independent wave functions.

    This function does a simple Gram-Schmidt orthogonalization followed by a normalization
    of the resulting orthogonal vectors. As such, it is not efficient for a large number
    of vectors, and produces artefacts for linearly dependent vectors. In short,
    it should be good for common use cases, but should only be used with sanitized input.

    Parameters
    ----------
    states: Sequence[wp.grid.State]
        The wave functions to orthonormalize. Must be linearly independent.
        May be empty or contain only one element, in which case the normalized wave function is returned.

    Returns
    -------
    list[State]
        A set of wave functions that span the same subspace as the input, but are normalized
        and orthogonal.

    Raises
    ------
    wp.BadGridError
        Raised if the input states are defined on different grids.

    wp.BadStateError
        Raised if any input state is either not a wave function or has zero trace.
    """
    if not states:
        return []

    grid = states[0].grid
    for state in states:
        if not state.is_wave_function():
            raise wp.BadStateError("Orthonormalization available only for wave functions.")

        if state.grid is not grid:
            raise wp.BadGridError("Cannot orthonormalize states on different grids.")

        if trace(state) == 0:
            raise wp.BadStateError("Cannot orthonormalize state with norm zero.")

    result: wpt.ComplexData = []
    for state in states:
        a = state.data
        state.data.sum()
        for b in result:
            a = _normalize(a)
            overlap = (np.conj(b) * a).sum()
            a = a - overlap * b
        result.append(_normalize(a))

    return [State(grid, v) for v in result]
