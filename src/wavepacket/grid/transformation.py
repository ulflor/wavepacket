from abc import ABC, abstractmethod
from typing import Final

import numpy as np

import wavepacket as wp

from .grid import Grid
from .state import State


class TransformationBase(ABC):
    """
    Base class for transformations of states between different grids.

    Transformations take a state defined on one grid, modify it to match another grid, and return the modified state.
    They usually encapsulate the code to create the new grid and the transformation logic.
    Examples are the tracing out of degrees of freedom, or a projection into a subspace.

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


class PartialTraceTransformation(TransformationBase):
    """
    Traces out all degrees of freedom except one.

    This transformation can be used to calculate reduced density matrices.
    You specify a degree of freedom, and all other degrees of freedom are traced out, leaving you with a
    reduced density operator along only the specified degree of freedom.
    It follows that this transformation accepts both wave functions and density operators as input,
    but produces only density operators as output.

    Parameters
    ----------
    source_grid: Grid
        The grid from which we transform.
    dof_index: int
        The index of the degree of freedom that is preserved.
        Can be negative in which case it is counted from the end as usual with Python indices.

    Attributes
    ----------
    source_grid: Grid, readonly
        The grid from which we transform.
    target_grid: Grid, readonly
        The grid onto which we transform.

    Raises
    ------
    wavepacket.BadGridError
        Raised if the source grid is one-dimensional. Tracing out degrees of freedom makes no sense in that case.
    IndexError
        Raised if the index of the preserved degree of freedom is invalid.
    """

    def __init__(self, source_grid: Grid, dof_index: int):
        if len(source_grid.dofs) == 1:
            raise wp.BadGridError("Partial trace over one degree of freedom is not possible.")

        dof_index = source_grid.normalize_index(dof_index)
        target_grid = Grid(source_grid.dofs[dof_index])

        self._dof_index = dof_index
        super().__init__(source_grid, target_grid)

    def transform(self, state: State, **kwargs) -> State:
        """
        Transforms the input state into a reduced density operator in the target grid.

        Parameters
        ----------
        state: State
            The state to transform.
            Must be a wave function or a density operator defined on the source_grid.
        kwargs
            Unused. This transformation is not parametrized.

        Raises
        ------
        wavepacket.BadGridError
            Raised if the state is defined on the wrong grid.
        wavepacket.BadStateError
            Raised if the input state is not a valid wave function or density operator.
        """
        if state.grid != self.source_grid:
            raise wp.BadGridError(
                "Input state must be defined on the transformation's source grid."
            )

        grid_rank = len(self.source_grid.dofs)

        if state.is_wave_function():
            psi = np.swapaxes(state.data, 0, self._dof_index)
            summation_axes = [x for x in range(1, grid_rank)]

            result = np.tensordot(psi, np.conj(psi), (summation_axes, summation_axes))
            return State(self.target_grid, result)
        elif state.is_density_operator():
            # make the relevant dof the first and squash all the other indices
            dof_size = self.source_grid.shape[self._dof_index]
            other_size = self.source_grid.size // dof_size

            tmp = np.swapaxes(state.data, 0, self._dof_index)
            tmp = np.swapaxes(tmp, grid_rank, grid_rank + self._dof_index)
            flattened = np.reshape(tmp, (dof_size, other_size, dof_size, other_size))

            result = np.trace(flattened, axis1=1, axis2=3)
            return State(self.target_grid, result)
        else:
            raise wp.BadStateError("Cannot transform invalid state.")
