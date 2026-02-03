import math

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from .dofbase import DofBase
from ._utils import broadcast


class PlaneWaveDof(DofBase):
    """
    One-dimensional plane wave basis expansion.

    This expansion is a good base choice for non-rotational degrees of freedom.
    The DVR grid consists of equally-spaced grid points from xmin to (xmax - dx),
    the FBR grid are the wave vectors of the plane waves centered around 0.
    In the FBR, derivatives become simple multiplications wih the wave vectors, so
    this grid allows a simple representation of kinetic energy operators.

    The transformation between DVR and FBR uses an FFT, so the performance is ok, even
    though you might need more grid points than with more suitable expansions.

    Be aware that this degree of freedom implicitly uses periodic boundary conditions
    in real space. That is, if the wave function leaves the grid on one side, it reenters
    the grid on the other side. This problem can only be mitigated with negative imaginary
    potentials. Periodic boundary conditions also hold in the FBR, causing aliasing.
    Nevertheless, the monitoring of convergence is rather simple, see [1]_.

    Parameters
    ----------
    xmin : float
        The start of the grid.
    xmax : float
        The end of the grid. Note that the last grid point is at xmax - dx.
    n : int
        The number of grid points.

    Raises
    ------
    wavepacket.InvalidValueError
        If the length of the grid is negative or the number of grid points is non-positive.

    References
    ----------
    .. [1] https://sourceforge.net/p/wavepacket/cpp/blog/2020/11/convergence-1-equally-space-grids
    """

    def __init__(self, xmin: float, xmax: float, n: int) -> None:
        if xmin > xmax:
            raise wp.InvalidValueError("Range should be positive")

        if n <= 0:
            raise wp.InvalidValueError(f"Number of grid points must be positive, but is {n}")

        dx = (xmax - xmin) / n

        dvr = np.linspace(xmin, xmax, n, endpoint=False, dtype=np.float64)
        if n % 2 == 0:
            fbr = np.linspace(-math.pi / dx, math.pi / dx, n, endpoint=False, dtype=np.float64)
        else:
            fbr = np.linspace(-math.pi / dx * (n - 1) / n, math.pi / dx * (n - 1) / n, n, dtype=np.float64)

        super().__init__(dvr, fbr)

        self._sqrt_weights: wpt.RealData = math.sqrt(dx) * np.ones(n)
        self._phase: wpt.ComplexData = np.exp(-1j * fbr * xmin) / np.sqrt(n)
        self._conj_phase: wpt.ComplexData = n * np.conj(self._phase)

    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        if is_ket:
            phase = broadcast(self._phase, data.ndim, index)
            transformed = np.fft.fftshift(np.fft.fft(data, axis=index), axes=index)
        else:
            phase = broadcast(self._conj_phase, data.ndim, index)
            transformed = np.fft.fftshift(np.fft.ifft(data, axis=index), axes=index)

        return phase * transformed

    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        if is_ket:
            phase = broadcast(self._conj_phase, data.ndim, index)
            untransformed = phase * data
            return np.fft.ifft(np.fft.ifftshift(untransformed, axes=index), axis=index)
        else:
            phase = broadcast(self._phase, data.ndim, index)
            untransformed = phase * data
            return np.fft.fft(np.fft.ifftshift(untransformed, axes=index), axis=index)

    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        conversion_factor = broadcast(self._sqrt_weights, data.ndim, index)
        return data / conversion_factor

    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        conversion_factor = broadcast(self._sqrt_weights, data.ndim, index)
        return data * conversion_factor
