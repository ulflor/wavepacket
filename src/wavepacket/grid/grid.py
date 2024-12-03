import numpy as np
import numpy.typing as npt

from .dof import DegreeOfFreedom
from ..exceptions import InvalidValueError


class Grid:
    def __init__(self, degrees_of_freedom: DegreeOfFreedom | list[DegreeOfFreedom]) -> None:
        if not degrees_of_freedom:
            raise InvalidValueError("Grid requires at least one degree of freedom.")

        try:
            for dof in degrees_of_freedom:
                self._validate_degree_of_freedom(dof)

            self._dof = list(degrees_of_freedom)
            self._shape = tuple([dof.dvr.size for dof in degrees_of_freedom])
        except TypeError:
            self._validate_degree_of_freedom(degrees_of_freedom)

            self._dof = [degrees_of_freedom]
            self._shape = degrees_of_freedom.dvr.shape

    @property
    def dof(self) -> list[DegreeOfFreedom]:
        return self._dof

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def operator_shape(self) -> tuple[int, ...]:
        return self._shape + self._shape

    def broadcast(self, data: npt.NDArray[np.floating], index: int) -> npt.NDArray[np.floating]:
        dim = np.ones(len(self._dof), dtype=np.integer)
        dim[index] = self._dof[index].dvr.size

        return np.reshape(data, dim)

    def operator_broadcast(self, data: npt.NDArray[np.floating], index: int) -> npt.NDArray[np.floating]:
        dim = np.ones(2*len(self._dof), dtype=np.integer)
        dim[index] = self._dof[index % len(self._dof)].dvr.size

        return np.reshape(data, dim)

    @staticmethod
    def _validate_degree_of_freedom(dof: DegreeOfFreedom) -> None:
        if dof.dvr.size == 0:
            raise InvalidValueError(f"A degree of freedom has zero points.")

        if dof.dvr.ndim != 1:
            raise InvalidValueError(f"Grids should be one-dimensional.")

        if dof.dvr.shape != dof.fbr.shape or dof.dvr.shape != dof.weights.shape:
            raise InvalidValueError(f"DVR, FBR and weights grid must have same size. "
                                    f"Got {dof.dvr.size}, {dof.fbr.size}, {dof.weights.size}, respectively.")
