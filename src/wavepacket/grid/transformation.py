from abc import ABC, abstractmethod
from typing import Final

from .grid import Grid
from .state import State


class TransformationBase(ABC):
    """
    Base class for transformations of states between different grids.

    Transformations take a state defined on one grid, modify it to match another grid, and return the modified state.
    For example, you might want to trace out a degree of freedom or project out an electronic state.

    Attributes
    ----------
    source_grid: Grid, readonly
        The grid from which we transform.
    target_grid: Grid, readonly
        The grid onto which we transform.
    """

    def __init__(self, source_grid: Grid, target_grid: Grid):
        self.source_grid: Final[Grid] = source_grid
        self.target_grid: Final[Grid] = target_grid

    @abstractmethod
    def transform(self, state: State, **kwargs) -> State:
        """
        Transforms a state from the source to the target grid.

        Depending on the exact transformation, there may be restrictions on the type of the input or
        output state. For example, when you trace out some degrees of freedom, the result is a
        density operator, even if the input is a wave function.

        Specific transformations may allow further parametrization, the corresponding keyword arguments
        are documented in the specific classes.
        """
        raise NotImplementedError
