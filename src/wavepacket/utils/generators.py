from typing import Optional

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt


class Gaussian(wpt.Generator):
    def __init__(self, x: float = 0.0, p: float = 0.0,
                 rms: Optional[float] = None, fwhm: Optional[float] = None):
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
            self._rms = fwhm / np.sqrt(2 * np.log(2))

    def __call__(self, x: wpt.RealData) -> wpt.ComplexData:
        shifted = x - self._x
        arg = - shifted ** 2 / (2 * self._rms ** 2) + 1j * self._p * shifted
        return np.exp(arg)


class PlaneWave:
    def __init__(self, k: float):
        self._k = k

    def __call__(self, x: wpt.RealData) -> wpt.ComplexData:
        return np.exp(1j * self._k * x)
