import numpy.testing

import wavepacket as wp


def assert_close(actual: wp.grid.State, expected: wp.grid.State, diff: float = 0) -> None:
    """
    Assertion helper: Verify that two states are on the same grid and similar to each other.

    Parameters
    ----------
    actual : wp.grid.State
        The expected state
    expected : wp.grid.State
        The state to be tested.
    diff : float
        The maximum absolute difference between any coefficients of the two states.
    """
    assert actual.grid == expected.grid
    numpy.testing.assert_allclose(actual.data, expected.data, rtol=0, atol=diff)
