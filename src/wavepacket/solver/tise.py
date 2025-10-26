import numpy as np

import wavepacket as wp

from ..operator.operatorbase import OperatorBase


def diagonalize(op: OperatorBase, t: float = 0.0):
    """
    Calculates the eigenstates and -values of an operator.

    See :doc:`tutorials/eigenstates` for a discussion of this topic.

    This function is a wrapper around `numpy.linalg.eigh` that calculates
    a matrix representation of the operator transforms the calculated eigenstates
    into a :py:class:`wp.grid.State` for easier consumption, and provides a generator
    for looping instead of a matrix with all eigenvalues in one go.
    This function diagonalises a full, dense operator matrix, so it requires  

    Typically, you solve the eigenproblem for time-independent operators,
    but you can also calculate instantaneous states and energies by specifying a time value.

    Parameters
    ----------
    op: wp.operator.OperatorBase
        The operator whose eigenstates and -values are calculated.

    t: float = 0.0
        The time at which the operator is evaluated. Can be ignored for time-independent operators.

    Yields
    -------
    Tuples consisting of the eigenenergy and the eigenstate of the operator.
    The output is sorted by the eigenvalues.
    """

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
