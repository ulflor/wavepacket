import numpy as np
import pytest
import wavepacket as wp

from numpy.testing import assert_allclose


def test_reject_non_positive_widths():
    with pytest.raises(wp.InvalidValueError):
        wp.Gaussian(rms=-1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.Gaussian(rms=0.0)

    with pytest.raises(wp.InvalidValueError):
        wp.Gaussian(fwhm=-1.0)

    with pytest.raises(wp.InvalidValueError):
        wp.Gaussian(fwhm=0.0)


def test_setting_rms_or_fwhm():
    with pytest.raises(wp.BadFunctionCall):
        wp.Gaussian(rms=1.0, fwhm=1.0)

    with pytest.raises(wp.BadFunctionCall):
        wp.Gaussian()


def test_gaussian():
    # The exact functional form is tested in the demos,
    # here we only verify the approximate scaling
    points = np.linspace(0, 10, 101)

    generator = wp.Gaussian(x=5, fwhm=2.0)
    vals = generator(points)

    assert_allclose(np.max(vals), 1.0, atol=1e-2)
    assert_allclose(vals[70], 0.5, atol=1e-2)

    generator = wp.Gaussian(x=5, rms=2.0 / np.sqrt(8 * np.log(2)))
    vals2 = generator(points)
    assert_allclose(vals, vals2, atol=1e-12, rtol=0)
