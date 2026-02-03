import numpy as np

import wavepacket as wp


def random_state(grid: wp.grid.Grid, seed: int) -> wp.grid.State:
    """
    Creates a random wave function.

    Note that this function does not employ high-quality randomization,
    it is only meant to create states without accidental symmetries or lots of code.
    """
    rng = np.random.default_rng(seed)
    data = rng.random(grid.shape) + 1j * rng.random(grid.shape)

    return wp.grid.State(grid, data)
