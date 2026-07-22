from typing import Iterator

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt


def expectation_value(
    op: wp.operator.OperatorBase, state: wp.grid.State, t: float | None = None
) -> complex:
    """
    Calculates the expectation value of an operator for a given state.

    Parameters
    ----------
    op: wp.operator.OperatorBase
        The operator whose expectation value is calculated.
    state: wp.grid.State
        The wave function or density operator that is used for the calculation.
    t: float | None
        The time at which the operator should be evaluated.
         Only required for time-dependent operators.

    Raises
    ------
    InvalidValueError
        If a time-dependent operator was supplied, but no time value was given.
    """
    if op.time_dependent and t is None:
        raise wp.InvalidValueError(
            "You must supply a time value for time-dependent operators."
        )

    new = op.apply(state, t)

    if state.is_wave_function():
        return np.vdot(state.data, new.data)
    else:
        matrix_data = np.reshape(new.data, [new.grid.size, new.grid.size])
        return np.trace(matrix_data)


def diagonalize(
    op: wp.operator.OperatorBase, t: float | None = None
) -> Iterator[tuple[float, wpt.ComplexData]]:
    """
    Calculates the eigenstates and -values of an operator.

    See :doc:`/tutorials/eigenstates` for a discussion of this topic.

    This function is a wrapper around `numpy.linalg.eigh` that calculates
    a matrix representation of the operator transforms the calculated eigenstates
    into a :py:class:`wavepacket.grid.State` for easier consumption, and provides a generator
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
        raise wp.InvalidValueError(
            "Time-dependent operators require a time whn to diagonalize."
        )

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


def transform_operator(
    op: wp.operator.OperatorBase, transform: wp.grid.TransformationBase, **kwargs
) -> wp.operator.TensorOperator:
    """
    Transforms an operator with a given transformation.

    You plug in an operator from the transformation's source grid, and obtain
    an operator in the transformation's target grid. Be aware that this function
    does o sophisticated magic. It merely assembles the matrix form of the operator,
    transforms it using the given transformation, and wraps the result in a
    :py:class:`wavepacket.operator.TensorOperator`.

    As a result, the function needs to construct a (large) operator matrix, may
    prouce a less efficient operator, and works only for certain transformations,
    in particular those that are effectively a projection (e.g.,
    :py:class:`wavepacket.grid.ChannelTransformation`). If you attempt a partial
    trace of the operator matrix, you might get unintended results.

    Parameters
    ----------
    op: wavepacket.operator.OperatorBase
        The operator that should be transformed.
    transform: wavepacket.grid.TransformationBase
        The transformation that should be applied to the operator.
    kwargs:
        Any additional parameters that need to be passed to the transformation.
        An example is the "channel" argument for the
        :py:class:`wavepacket.grid.ChannelTransformation`.

    Returns
    -------
    wavepacket.operator.TensorOperator
        The transformed operator defined on the target grid.

    Raises
    ------
    wavepacket.BadGridError
        Raised if the operator is not defined on the transformations's source grid.
    """
    if op.grid is not transform.source_grid:
        raise wp.BadGridError("Grids of operator and transformation do not match.")

    unit_density = wp.builder.unit_density(op.grid)
    op_matrix = op.apply(unit_density, 0)
    transformed_matrix = transform.transform(op_matrix, **kwargs)

    return wp.operator.TensorOperator(transformed_matrix.grid, transformed_matrix.data)
