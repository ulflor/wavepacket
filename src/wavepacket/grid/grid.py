import numpy as np
from .dof import DegreeOfFreedom
from ..exceptions import InvalidValueError


class Grid:
    def __init__(self, degrees_of_freedom: list[DegreeOfFreedom]):
        if not degrees_of_freedom:
            raise InvalidValueError("Grid requires at least one degree of freedom.")

        for dvr, fbr, weights in degrees_of_freedom:
            if dvr.size != fbr.size or dvr.size != weights.size:
                raise InvalidValueError(f"DVR, FBR and weights grid must have same size. "
                                        f"Got {dvr.size}, {fbr.size}, {weights.size}, respectively.")

        for dvr, _, _ in degrees_of_freedom:
            if dvr.size == 0:
                raise InvalidValueError(f"A degree of freedom has zero points.")