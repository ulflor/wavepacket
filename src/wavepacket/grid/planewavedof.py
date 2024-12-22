import math

import numpy as np

from .dofbase import DofBase
from ..typing import ComplexData, RealData
from ..utils import InvalidValueError


def _broadcast(data: ComplexData, ndim, index: int) -> ComplexData:
    shape = np.ones(ndim, dtype=int)
    shape[index] = len(data)

    return np.reshape(data, shape)


class PlaneWaveDof(DofBase):
    def __init__(self, xmin: float, xmax: float, n: int):
        if xmin > xmax:
            raise InvalidValueError("Range should be positive")

        if n <= 0:
            raise InvalidValueError(f"Number of grid points must be positive, but is {n}")

        dx = (xmax - xmin) / n

        dvr = np.linspace(xmin, xmax, n, endpoint=False)
        if n % 2 == 0:
            fbr = np.linspace(-math.pi / dx, math.pi / dx, n, endpoint=False)
        else:
            fbr = np.linspace(-math.pi / dx * (n - 1) / n, math.pi / dx * (n - 1) / n, n)

        super().__init__(dvr, fbr)

        self._sqrt_weights: RealData = math.sqrt(dx) * np.ones(n)
        self._phase: ComplexData = np.exp(-1j * fbr * xmin) / np.sqrt(n)
        self._conj_phase: ComplexData = n * np.conj(self._phase)

    def to_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        if is_ket:
            phase = _broadcast(self._phase, data.ndim, index)
            transformed = np.fft.fftshift(np.fft.fft(data, axis=index), axes=index)
        else:
            phase = _broadcast(self._conj_phase, data.ndim, index)
            transformed = np.fft.fftshift(np.fft.ifft(data, axis=index), axes=index)

        return phase * transformed

    def from_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        if is_ket:
            phase = _broadcast(self._conj_phase, data.ndim, index)
            untransformed = phase * data
            return np.fft.ifft(np.fft.ifftshift(untransformed, axes=index), axis=index)
        else:
            phase = _broadcast(self._phase, data.ndim, index)
            untransformed = phase * data
            return np.fft.fft(np.fft.ifftshift(untransformed, axes=index), axis=index)

    def to_dvr(self, data: ComplexData, index: int) -> ComplexData:
        conversion_factor = _broadcast(self._sqrt_weights, data.ndim, index)
        return data / conversion_factor

    def from_dvr(self, data: ComplexData, index: int) -> ComplexData:
        conversion_factor = _broadcast(self._sqrt_weights, data.ndim, index)
        return data * conversion_factor
