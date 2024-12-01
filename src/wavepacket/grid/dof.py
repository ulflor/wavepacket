import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from ..exceptions import InvalidValueError


class DofType(Enum):
    PLANE_WAVE = 1
    OTHER = 2


@dataclass(frozen=True)
class DegreeOfFreedom:
    type: DofType
    dvr: npt.NDArray[np.floating]
    fbr: npt.NDArray[np.floating]
    weights: npt.NDArray[np.floating]


def plane_wave_dof(xmin: float, xmax: float, n: int) -> DegreeOfFreedom:
    if xmin >= xmax:
        raise InvalidValueError(f"Range [{xmin}, {xmax}] should be strictly monotonic increasing.")

    if n <= 0:
        raise InvalidValueError(f"Number of grid points must be positive, but is '{n}'.")

    dx = (xmax - xmin) / n

    dvr = np.linspace(xmin, xmax, n, endpoint=False)
    if n % 2 == 0:
        fbr = np.linspace(-math.pi / dx, math.pi / dx, n, endpoint=False)
    else:
        fbr = np.linspace(-math.pi / dx * (n - 1) / n, math.pi / dx * (n - 1) / n, n)
    weights = dx * np.ones(n)

    return DegreeOfFreedom(DofType.PLANE_WAVE, dvr, fbr, weights)