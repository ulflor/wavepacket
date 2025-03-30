import numpy as np

import wavepacket as wp
from ..grid import State


def pure_density(psi: State) -> State:
    """
    Given an input wave function, create the corresponding pure density operator.

    This function only performs the direct product, it does not apply
    further modifications like normalizations.

    Parameters
    ----------
    psi: wp.grid.State
         The input wave function

    Returns
    -------
    wp.grid.State
        The corresponding density operator.

    Raises
    ------
    wp.BadStateError
        If the input is not a valid wave function.
    """
    return direct_product(psi, psi)


def direct_product(ket: State, bra: State) -> State:
    """
    Returns a direct product of wave functions as density operators.

    Given two wave functions :math:`\psi, \phi`, this function returns the
    density operator resulting from the direct product
    :math:`| \psi >< \phi |` This can be useful to build up a
    density operator piecewise.

    Parameters
    ----------
    ket: wp.grid.State
         The ket state :math:`\psi`
    bra: wp.grid.State
         The bra state :math:`\phi`. Note that the function performs a
         complex conjugation of this state prior to multiplication.

    Returns
    -------
    wp.grid.State
        The direct product of the two states.

    Raises
    ------
    wp.BadStateError
        If one of the input states is not a valid wave function.
    wp.BadGridError
        If the input states are defined on different grids.
    """
    if not ket.is_wave_function() or not bra.is_wave_function():
        raise wp.BadStateError("Density operator can only be constructed from wave functions.")

    if ket.grid != bra.grid:
        raise wp.BadGridError("Grid for bra and ket states does not match")

    rho_matrix = np.outer(ket.data, np.conj(bra.data))

    return State(ket.grid, np.reshape(rho_matrix, ket.grid.operator_shape))
