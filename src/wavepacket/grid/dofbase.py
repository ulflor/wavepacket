import numpy as np
import numpy.typing as npt

from ..utils import InvalidValueError


class DofBase:
    def __init__(self, dvr_array: npt.NDArray[np.floating], fbr_array: npt.NDArray[np.floating]):
        if len(dvr_array) == 0 or len(fbr_array) == 0:
            raise InvalidValueError("Degrees of freedom may not be empty.")

        if dvr_array.ndim != 1 or fbr_array.ndim != 1:
            raise InvalidValueError("A degree of freedom represents only one-dimensional data.")

        self._dvr_array = dvr_array
        self._fbr_array = fbr_array

    @property
    def dvr_array(self) -> npt.NDArray[np.floating]:
        return self._dvr_array

    @property
    def fbr_array(self) -> npt.NDArray[np.floating]:
        return self._fbr_array

    def to_fbr(self, data: npt.NDArray[complex | float], index: int, is_ket: bool = True)\
            -> npt.NDArray[complex | float]:
        pass

    def from_fbr(self, data: npt.NDArray[complex | float], index: int, is_ket: bool = True)\
            -> npt.NDArray[complex | float]:
        pass

    def to_dvr(self, data: npt.NDArray[complex | float], index: int) -> npt.NDArray[complex | float]:
        pass

    def from_dvr(self, data: npt.NDArray[complex | float], index: int) -> npt.NDArray[complex | float]:
        pass

