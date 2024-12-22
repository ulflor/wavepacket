from abc import abstractmethod, ABC

from ..typing import ComplexData, RealData
from ..utils import InvalidValueError


class DofBase(ABC):
    def __init__(self, dvr_array: RealData, fbr_array: RealData):
        if len(dvr_array) == 0 or len(fbr_array) == 0:
            raise InvalidValueError("Degrees of freedom may not be empty.")

        if dvr_array.ndim != 1 or fbr_array.ndim != 1:
            raise InvalidValueError("A degree of freedom represents only one-dimensional data.")

        self._dvr_array = dvr_array
        self._fbr_array = fbr_array

    @property
    def dvr_array(self) -> RealData:
        return self._dvr_array

    @property
    def fbr_array(self) -> RealData:
        return self._fbr_array

    @abstractmethod
    def to_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        pass

    @abstractmethod
    def from_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        pass

    @abstractmethod
    def to_dvr(self, data: ComplexData, index: int) -> ComplexData:
        pass

    @abstractmethod
    def from_dvr(self, data: ComplexData, index: int) -> ComplexData:
        pass
