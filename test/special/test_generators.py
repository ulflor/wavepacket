import math
import numpy as np
import pytest
import wavepacket as wp

from numpy.testing import assert_allclose


def test_reject_non_positive_widths():
    with pytest.raises(wp.InvalidValueError):
        wp.special.Gaussian(rms=-1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.special.Gaussian(rms=0.0)

    with pytest.raises(wp.InvalidValueError):
        wp.special.Gaussian(fwhm=-1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.special.Gaussian(fwhm=0.0)


def test_setting_rms_or_fwhm():
    with pytest.raises(wp.BadFunctionCall):
        wp.special.Gaussian(rms=1.0, fwhm=1.0)

    with pytest.raises(wp.BadFunctionCall):
        wp.special.Gaussian()


def test_gaussian():
    # The exact functional form is tested in the demos,
    # here we only verify the approximate scaling
    points = np.linspace(0, 10, 101)

    generator = wp.special.Gaussian(x=5, fwhm=4.0)
    vals = generator(points)

    assert_allclose(np.max(vals), 1.0, atol=1e-2)
    assert_allclose(vals[70], 0.5, atol=1e-2)

    generator = wp.special.Gaussian(x=5, rms=4.0 / np.sqrt(8 * np.log(2)))
    vals2 = generator(points)
    assert_allclose(vals, vals2, atol=1e-12, rtol=0)


def test_spherical_harmonic_rejects_invalid_values():
    with pytest.raises(wp.InvalidValueError):
        wp.special.SphericalHarmonic(-1, 0)

    with pytest.raises(wp.InvalidValueError):
        wp.special.SphericalHarmonic(5, -6)

    with pytest.raises(wp.InvalidValueError):
        wp.special.SphericalHarmonic(5, 7)


def test_spherical_harmonics():
    theta = np.linspace(0, math.pi, 10)

    h_20 = wp.special.SphericalHarmonic(2, 0)
    expected = 0.25 * math.sqrt(5 / math.pi) * (3 * np.cos(theta) ** 2 - 1)
    assert_allclose(h_20(theta), expected, atol=1e-12, rtol=0)

    h_21 = wp.special.SphericalHarmonic(2, 1)
    expected = - 0.5 * math.sqrt(15 / 2 / math.pi) * np.sin(theta) * np.cos(theta)
    assert_allclose(h_21(theta), expected, atol=1e-12, rtol=0)

    h_33 = wp.special.SphericalHarmonic(3, -3)
    expected = 1 / 8 * math.sqrt(35 / math.pi) * np.sin(theta) ** 3
    assert_allclose(h_33(theta), expected, atol=1e-12, rtol=0)
