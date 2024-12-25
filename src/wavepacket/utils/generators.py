import numpy as np

from typing import Optional

from ..typing import RealData, ComplexData, Generator
from .exceptions import BadFunctionCall, InvalidValueError


class Gaussian(Generator):
    def __init__(self, x: float = 0.0, p: float = 0.0,
                 rms: Optional[float] = None, fwhm: Optional[float] = None):
        if rms is not None and rms <= 0:
            raise InvalidValueError(f"RMS width of Gaussian is {rms}, but should be positive.")

        if fwhm is not None and fwhm <= 0:
            raise InvalidValueError(f"FWHM of Gaussian is {rms}, but should be positive.")

        if fwhm is not None and rms is not None:
            raise BadFunctionCall("Only one of RMS width or FWHM must be set, not both.")
        if fwhm is None and rms is None:
            raise BadFunctionCall("One of RMS width or FWHM must be set.")

        self._x = x
        self._p = p
        if rms:
            self._rms = rms
        else:
            self._rms = fwhm / np.sqrt(2 * np.log(2))

    def __call__(self, x: RealData) -> ComplexData:
        shifted = x - self._x
        arg = - shifted ** 2 / (2 * self._rms ** 2) + 1j * self._p * shifted
        return np.exp(arg)


class PlaneWave:
    def __init__(self, k: float):
        self._k = k

    def __call__(self, x: RealData) -> ComplexData:
        return np.exp(1j * self._k * x)
