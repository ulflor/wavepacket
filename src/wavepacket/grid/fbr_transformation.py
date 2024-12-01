from enum import Enum

from ..exceptions import InvalidValueError
from ..state import State
from .dof import DofType
from .grid import Grid


class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2


class FbrTransformation:
    def transform_ket(self, state: State, direction: Direction) -> State:
        pass

    def transform_bra(self, state: State, direction: Direction) -> State:
        pass


class PlaneWaveTransformation(FbrTransformation):
    def __init__(self, grid: Grid, dof_index: int):
        dof = grid.dof[dof_index]
        if dof.type != DofType.PLANE_WAVE:
            raise InvalidValueError("Plane Wave Transformation requires plane wave degree of freedom.")