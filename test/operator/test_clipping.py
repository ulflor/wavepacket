import numpy as np

import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
from wavepacket.operator._clipping import clip_real


@pytest.fixture
def input_data() -> wp.typing.ComplexData:
    return np.linspace(-5, 5, 20) + 1j * np.linspace(-10, 10, 20)


def test_clipping(input_data):
    clipped_data = clip_real(input_data, -6, 6)

    assert_allclose(clipped_data, input_data, rtol=0)


def test_clip_lower_bound(input_data):
    lower_bound = 0.0
    clipped_data = clip_real(input_data, lower_bound)

    assert_allclose(np.imag(clipped_data), np.imag(input_data), rtol=0)

    real_input = np.real(input_data)
    real_clipped = np.real(clipped_data)
    for index in range(input_data.size):
        assert real_clipped[index] == max(lower_bound, real_input[index])


def test_clip_upper_bound(input_data):
    upper_bound = 0.0
    clipped_data = clip_real(input_data, upper=upper_bound)

    assert_allclose(np.imag(clipped_data), np.imag(input_data), rtol=0)

    real_input = np.real(input_data)
    real_clipped = np.real(clipped_data)
    for index in range(input_data.size):
        assert real_clipped[index] == min(upper_bound, real_input[index])


def test_clip_real_data():
    data = np.arange(0.5, 10.5)
    clipped_data = clip_real(data, 2, 7)

    assert data.dtype == np.float64
    assert clipped_data.dtype == np.float64

    assert np.all(clipped_data <= 7)
    assert np.all(clipped_data >= 2)
