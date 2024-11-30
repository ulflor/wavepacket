from .dof import DegreeOfFreedom
from ..exceptions import InvalidValueError


class Grid:
    def __init__(self, degrees_of_freedom: list[DegreeOfFreedom]) -> None:
        if not degrees_of_freedom:
            raise InvalidValueError("Grid requires at least one degree of freedom.")

        for dof in degrees_of_freedom:
            if dof.dvr.size == 0:
                raise InvalidValueError(f"A degree of freedom has zero points.")

            if dof.dvr.ndim != 1:
                raise InvalidValueError(f"Grids should be one-dimensional.")

            if dof.dvr.shape != dof.fbr.shape or dof.dvr.shape != dof.weights.shape:
                raise InvalidValueError(f"DVR, FBR and weights grid must have same size. "
                                        f"Got {dof.dvr.size}, {dof.fbr.size}, {dof.weights.size}, respectively.")

        self._dof = list(degrees_of_freedom)
        self._shape = tuple([dof.dvr.size for dof in degrees_of_freedom])

    @property
    def dof(self) -> list[DegreeOfFreedom]:
        return self._dof

    @property
    def shape(self) -> tuple[int,...]:
        return self._shape

    @property
    def operator_shape(self) -> tuple[int,...]:
        return self._shape + self._shape