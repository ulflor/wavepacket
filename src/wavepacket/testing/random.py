import numpy as np

import wavepacket as wp


def random_state(grid, seed) -> wp.State:
    # Note: We do not require high-quality randomization here,
    # we just want to avoid accidental symmetries and cumbersome setup.
    rng = np.random.default_rng(seed)
    data = rng.random(grid.shape) + 1j * rng.random(grid.shape)

    return wp.State(grid, data)
