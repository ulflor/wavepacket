import numpy as np
import numpy.typing as npt

from typing import TypeAlias

RealData: TypeAlias = npt.NDArray[np.float64]
ComplexData: TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.complex128]
