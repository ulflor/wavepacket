from typing import Callable, TypeAlias

import numpy as np
import numpy.typing as npt

RealData: TypeAlias = npt.NDArray[np.float64]
ComplexData: TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.complex128]

Generator: TypeAlias = Callable[[RealData], ComplexData]
RealGenerator: TypeAlias = Callable[[RealData], RealData]
