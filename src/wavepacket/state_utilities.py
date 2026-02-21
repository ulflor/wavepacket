from collections.abc import Sequence
import math

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt


def _take_diagonal(data: wpt.ComplexData, grid: wp.grid.Grid) -> wpt.RealData:
    # note: the diagonal should be real and positive but for numerical issues
    matrix_form = np.reshape(data, (grid.size, grid.size))
    diagonal = np.abs(np.diag(matrix_form))
    return np.reshape(diagonal, grid.shape)


def _normalize(u: wpt.ComplexData) -> wpt.ComplexData:
    return u / math.sqrt(np.abs(u ** 2).sum())


def dvr_density(state: wp.grid.State) -> wpt.RealData:
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


def fbr_density(state: wp.grid.State) -> wpt.RealData:
    """
    Returns the FBR density of the input state at the FBR grid points.

    The density is returned as a real-valued coefficient array
    with the same shape as the underlying grid. The density is mostly a
    dead end: useful for plotting and inspection, but not for
    further computations.

    Note that this function returns the FBR density, i.e., the coefficients
    or diagonal when expanding the wave function or density operator in the
    underlying grid's basis, respectively.

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


def trace(state: wp.grid.State) -> float:
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


def normalize(state: wp.grid.State) -> wp.grid.State:
    """
    Normalizes the input state.

    Parameters
    ----------
    state : wp.grid.State
        The state (wave function or density operator) that should be normalized.

    Returns
    -------
    wp.grid.State
        The normalized input state.

    Raises
    ------
    wp.BadStateError
        If the supplied state is neither a wave function nor a density operator.

    """
    if state.is_wave_function():
        norm = np.sqrt(trace(state))
        return state / norm
    elif state.is_density_operator():
        return state / trace(state)
    else:
        raise wp.BadStateError("Input is not a valid state.")


def orthonormalize(states: Sequence[wp.grid.State]) -> list[wp.grid.State]:
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

    return [wp.grid.State(grid, v) for v in result]


def population(state: wp.grid.State, target: wp.grid.State) -> float:
    """
    Calculates how much a target state is populated in a given state.

    The return value is simply the absolute square of the scalar product of
    the two states. The target state is normalized before the calculation.
    This operation can be thought of as a shortcut for creating a
    :py:class:`wp.operator.Projection` with the target and calculating
    the :py:func:`wp.operator.expectation_value` of the input state.

    Parameters
    ----------
    state: wp.grid.State
        The state that is projected onto the target state

    target: wp.grid.State
        The wave function onto which the input state is projected.
        This function is normalized before use.

    Raises
    ------
    wp.BadGridError
        Raised if the input and the target state are defined on different grids.

    wp.BadStateError
        Raised if the target is not a wave function.

    See Also
    --------
    wavepacket.operator.Projection: For advanced calculations
    """
    if not target.is_wave_function():
        raise wp.BadStateError("Projection requires a wave function as target.")

    if target.grid != state.grid:
        raise wp.BadGridError("Target wave function must be the same grid as the state to project.")

    target_trace = trace(target)
    if state.is_wave_function():
        coefficient = (np.conj(target.data) * state.data).sum()
        return coefficient ** 2 / target_trace
    elif state.is_density_operator():
        matrix_form = np.reshape(state.data, [state.grid.size, state.grid.size])
        flat_target = np.ravel(target.data)
        left_summation = np.tensordot(np.conj(flat_target), matrix_form, axes=(0, 0))
        return np.tensordot(left_summation, flat_target, axes=(0, 0)).sum() / target_trace
    else:
        raise wp.BadStateError("Input is not a valid state.")
