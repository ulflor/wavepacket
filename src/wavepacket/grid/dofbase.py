from abc import abstractmethod, ABC

import wavepacket as wp
import wavepacket.typing as wpt


class DofBase(ABC):
    def __init__(self, dvr_points: wpt.RealData, fbr_points: wpt.RealData):
        if len(dvr_points) == 0 or len(fbr_points) == 0:
            raise wp.InvalidValueError("Degrees of freedom may not be empty.")

        if dvr_points.ndim != 1 or fbr_points.ndim != 1:
            raise wp.InvalidValueError("A degree of freedom represents only one-dimensional data.")

        if dvr_points.size != fbr_points.size:
            raise wp.InvalidValueError("The DVR and FBR grids must have the same size.")

        self._dvr_points = dvr_points
        self._fbr_points = fbr_points

    @property
    def dvr_points(self) -> wpt.RealData:
        return self._dvr_points

    @property
    def fbr_points(self) -> wpt.RealData:
        return self._fbr_points

    @property
    def size(self) -> int:
        return self._dvr_points.size

    @abstractmethod
    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        pass

    @abstractmethod
    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        pass

    @abstractmethod
    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        pass

    @abstractmethod
    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        pass
