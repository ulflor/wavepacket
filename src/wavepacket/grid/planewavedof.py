import math
import numpy as np

from .dofbase import DofBase
from ..utils import InvalidValueError


class PlaneWaveDof(DofBase):
    def __init__(self, xmin: float, xmax: float, n: int):
        if xmin > xmax:
            raise InvalidValueError("Range should be positive")

        if n <= 0:
            raise InvalidValueError(f"Number of grid points must be positive, but is {n}")

        dx = (xmax - xmin) / n
        self._weights = dx * np.ones(n)

        dvr = np.linspace(xmin, xmax, n, endpoint=False)
        if n % 2 == 0:
            fbr = np.linspace(-math.pi / dx, math.pi / dx, n, endpoint=False)
        else:
            fbr = np.linspace(-math.pi / dx * (n - 1) / n, math.pi / dx * (n - 1) / n, n)

        super().__init__(dvr, fbr)