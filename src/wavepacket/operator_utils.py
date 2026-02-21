from typing import Iterator

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt


def expectation_value(op: wp.operator.OperatorBase,
                      state: wp.grid.State, t: float | None = None) -> complex:
    """
    Calculates the expectation value of an operator for a given state.

    Parameters
    ----------
    op : wp.operator.OperatorBase
        The operator whose expectation value is calculated.
    state : wp.grid.State
        The wave function or density operator that is used for the calculation.
    t : float | None
        The time at which the operator should be evaluated.
         Only required for time-dependent operators.

    Raises
    ------
    InvalidValueError
        If a time-dependent operator was supplied, but no time value was given.
    """
    if op.time_dependent and t is None:
        raise wp.InvalidValueError("You must supply a time value for time-dependent operators.")

    new = op.apply(state, t)

    if state.is_wave_function():
        return np.vdot(state.data, new.data)
    else:
        matrix_data = np.reshape(new.data, [new.grid.size, new.grid.size])
        return np.trace(matrix_data)


def diagonalize(op: wp.operator.OperatorBase,
                t: float | None = None) -> Iterator[tuple[float, wpt.ComplexData]]:
    """
    Calculates the eigenstates and -values of an operator.

    See :doc:`/tutorials/eigenstates` for a discussion of this topic.

    This function is a wrapper around `numpy.linalg.eigh` that calculates
    a matrix representation of the operator transforms the calculated eigenstates
    into a :py:class:`wp.grid.State` for easier consumption, and provides a generator
    for looping instead of a matrix with all eigenvalues in one go.
    This function diagonalizes a full, dense operator matrix, so it requires

    Typically, you solve the eigenproblem for time-independent operators,
    but you can also calculate instantaneous states and energies by specifying a time value.

    Parameters
    ----------
    op: wp.operator.OperatorBase
        The operator whose eigenstates and -values are calculated.

    t: float | None = None
        The time at which the operator is evaluated.
        Required only for time-dependent operators.

    Yields
    ------
    Tuples consisting of the eigenenergy and the eigenstate of the operator.
    The output is sorted by the eigenvalues.

    Raises
    ------
    wp.InvalidValueError
        If no time was supplied for a time-dependent operator

    Examples
    --------
    Iterate over the eigenvalues and -vectors

    >>> hamiltonian = ...
    >>> for energy, state in wp.diagonalize(hamiltonian):
    >>>     print(f'E = {energy}, trace norm = {wp.trace(state)}')
    """
    if op.time_dependent and t is None:
        raise wp.InvalidValueError("Time-dependent operators require a time whn to diagonalize.")

    grid = op.grid

    # construct a matrix representation in weighted DVR of the operator
    rho = wp.builder.unit_density(grid)
    rho = op.apply(rho, t)
    matrix = np.reshape(rho.data, [grid.size, grid.size])

    # Diagonalize and convert the result
    vals, vecs = np.linalg.eigh(matrix)
    for i in range(vals.size):
        psi_data = np.reshape(vecs[:, i], grid.shape)
        psi = wp.grid.State(grid, psi_data)
        yield vals[i], psi
