import numpy as np

import wavepacket as wp
from ..grid import State


def pure_density(psi: State) -> State:
    return direct_product(psi, psi)


def direct_product(ket: State, bra: State) -> State:
    if not ket.is_wave_function() or not bra.is_wave_function():
        raise wp.BadStateError("Density operator can only be constructed from wave functions.")

    if ket.grid != bra.grid:
        raise wp.BadGridError("Grid for bra and ket states does not match")

    rho_matrix = np.outer(ket.data, np.conj(bra.data))

    return State(ket.grid, np.reshape(rho_matrix, ket.grid.operator_shape))
