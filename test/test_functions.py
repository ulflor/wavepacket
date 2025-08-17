import math
import pytest

import wavepacket as wp

from numpy.testing import assert_allclose


def test_sin_square_errors():
    with pytest.raises(wp.InvalidValueError):
        wp.SinSquare(5.0, -1)

    with pytest.raises(wp.InvalidValueError):
        wp.SinSquare(3.0, 0.0)


def test_sin_square_values():
    functor = wp.SinSquare(5.0, 2.0)

    assert functor(2.9) == 0
    assert_allclose(functor(4), math.sqrt(0.5), rtol=0, atol=1e-12)
    assert functor(5) == 1
    assert_allclose(functor(6), math.sqrt(0.5), rtol=0, atol=1e-12)
    assert functor(7.1) == 0


def test_soft_rectangular_errors():
    with pytest.raises(wp.InvalidValueError):
        wp.SoftRectangularFunction(5.0, -1.0, 1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.SoftRectangularFunction(5.0, 0.0, 1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.SoftRectangularFunction(5.0, 1.0, -1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.SoftRectangularFunction(5.0, 1.0, 0.0)

    with pytest.raises(wp.InvalidValueError):
        wp.SoftRectangularFunction(5.0, 0.0)


def test_soft_rectangular_values():
    functor = wp.SoftRectangularFunction(5.0, 2.0, 1.0)

    assert functor(1.9) == 0.0
    assert functor(2) == 0.0
    assert 0 <= functor(2.3) <= 1
    assert functor(3) == 1.0
    assert 1 == functor(3.1) and 1 == functor(6.8)
    assert functor(7) == 1.0
    assert 0 <= functor(7.5) <= 1
    assert functor(8) == 0.0
    assert functor(8.5) == 0.0

