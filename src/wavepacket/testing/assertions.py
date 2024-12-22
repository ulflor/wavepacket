import wavepacket as wp
from numpy.testing import assert_allclose


def assert_close(actual: wp.State, expected: wp.State, diff: float = 0) -> None:
    assert actual.grid == expected.grid
    assert_allclose(actual.data, expected.data, rtol=0, atol=diff)
