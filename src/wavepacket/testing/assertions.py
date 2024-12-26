import numpy.testing

import wavepacket as wp


def assert_close(actual: wp.State, expected: wp.State, diff: float = 0) -> None:
    assert actual.grid == expected.grid
    numpy.testing.assert_allclose(actual.data, expected.data, rtol=0, atol=diff)
