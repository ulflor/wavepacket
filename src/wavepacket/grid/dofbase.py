from abc import abstractmethod, ABC

from ..typing import ComplexData, RealData
from ..utils import InvalidValueError


class DofBase(ABC):
    def __init__(self, dvr_points: RealData, fbr_points: RealData):
        if len(dvr_points) == 0 or len(fbr_points) == 0:
            raise InvalidValueError("Degrees of freedom may not be empty.")

        if dvr_points.ndim != 1 or fbr_points.ndim != 1:
            raise InvalidValueError("A degree of freedom represents only one-dimensional data.")

        if dvr_points.size != fbr_points.size:
            raise InvalidValueError("The DVR and FBR grids must have the same size.")

        self._dvr_points = dvr_points
        self._fbr_points = fbr_points

    @property
    def dvr_points(self) -> RealData:
        return self._dvr_points

    @property
    def fbr_points(self) -> RealData:
        return self._fbr_points

    @property
    def size(self) -> int:
        return self._dvr_points.size

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
