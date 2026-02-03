import math

import numpy as np
import scipy

import wavepacket as wp
import wavepacket.typing as wpt


class Gaussian(wpt.Generator):
    """
    Callable that defines a one-dimensional Gaussian function.

    This callable can be supplied wherever a callable is required. An example
    would be an initial wave function for :py:func:`wavepacket.builder.product_wave_function`,
    or a potential wrapped in a :py:class:`wavepacket.operator.Potential1D`.

    Parameters
    ----------
    x : float, default=0
        The center of the Gaussian.
    p : float, default=0
        The momentum of the Gaussian.
    rms, fwhm : float
        You must specify the width of the Gaussian using exactly one of these values,
        either through the root-mean-square width, or the full-width-at-half-maximum.

    Raises
    ------
    wp.InvalidValueError
        If the width of the Gaussian is not positive.
    wp.BadFunctionCall
        If both rms and fwhm have either been set or not supplied.

    Notes
    -----
    Up to scaling, the functional form of the Gaussian is
    :math:`f(x) = e^{-(x-x_0)^2 / 2 \sigma^2 + \imath p (x-x_0)}`.
    Here, sigma is the rms width, which is connected to the FWHM by
    :math:`\sigma = \mathrm{FWHM} / \sqrt{8 \ln 2}`.
    """

    def __init__(self, x: float = 0.0, p: float = 0.0,
                 rms: float | None = None, fwhm: float | None = None):
        if rms is not None and rms <= 0:
            raise wp.InvalidValueError(f"RMS width of Gaussian is {rms}, but should be positive.")

        if fwhm is not None and fwhm <= 0:
            raise wp.InvalidValueError(f"FWHM of Gaussian is {rms}, but should be positive.")

        if fwhm is not None and rms is not None:
            raise wp.BadFunctionCall("Only one of RMS width or FWHM must be set, not both.")
        if fwhm is None and rms is None:
            raise wp.BadFunctionCall("One of RMS width or FWHM must be set.")

        self._x = x
        self._p = p
        if rms:
            self._rms = rms
        else:
            self._rms = fwhm / np.sqrt(8 * np.log(2))

    def __call__(self, x: wpt.RealData) -> wpt.ComplexData:
        shifted = x - self._x
        arg = - shifted ** 2 / (2 * self._rms ** 2) + 1j * self._p * shifted
        return np.exp(arg)


class PlaneWave(wpt.Generator):
    """
    Callable that defines a plane wave.

    You will typically use this callable for initial states. There are often
    better options, especially if your FBR already defines a plane wave basis,
    but sometimes you may just want to represent a reasonable plane wave and
    not count indices to get the correct wave vector.

    Parameters
    ----------
    k : float
        The wave vector of the plane wave.
    """

    def __init__(self, k: float):
        self._k = k

    def __call__(self, x: wpt.RealData) -> wpt.ComplexData:
        return np.exp(1j * self._k * x)


class SphericalHarmonic(wpt.RealGenerator):
    """
    Callable that returns a spherical harmonic Y_l^m(theta, phi=0).

    Usually, this callable will be used for initial states. Note that the
    phi-dependence of a spherical harmonic is trivial exp(i m phi), and
    usually not needed (we fix m and the phi-integration yields a constant).
    For this reason, the functor takes only the theta-values as single parameters,
    and returns the spherical harmonic at phi = 0.

    Parameters
    ----------
    l : int
        The rotational quantum number / angular momentum
    m : int
        The minor rotational quantum number -l <= m <= l

    Raises
    ------
    wp.InvalidValueError
        If l is negative or if (-l <= m <= l) does not hold.
    """

    def __init__(self, l: int, m: int):
        if l < 0:
            raise wp.InvalidValueError(f"Angular momentum must not be negative, but is '{l}'.")

        if m > l or m < -l:
            raise wp.InvalidValueError(
                f"Quantum number m must fulfill -l <= m <= l, but we have {-l} <= {m} <= {l}.")

        self._l = l
        self._m = m

    def __call__(self, theta: wpt.RealData) -> wpt.RealData:
        factor = math.sqrt((2 * self._l + 1) * math.factorial(self._l - self._m) /
                           (4 * math.pi * math.factorial(self._l + self._m)))
        legendre = scipy.special.lpmv(self._m, self._l, np.cos(theta))

        return factor * legendre
