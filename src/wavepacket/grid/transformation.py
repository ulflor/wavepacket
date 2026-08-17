import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Final, override

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
        The grid onto which we transform. Usually, this grid is created by the transformation.
    """

    def __init__(self, source_grid: Grid, target_grid: Grid):
        self.source_grid: Final[Grid] = source_grid
        self.target_grid: Final[Grid] = target_grid

    @abstractmethod
    def transform(self, state: State, **kwargs: Any) -> State:
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

    @override
    def transform(self, state: State, **kwargs: Any) -> State:
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


class ChannelProjectionTransformation(TransformationBase):
    """
    Transformation that strips the channel degree of freedom and projects.

    This transformation does two things: It projects the state onto a specific channel,
    similar to {py:class}`wavepacket.operator.Channel`, and then it removes the channel
    degree of freedom.

    Note that the actual transformation _requires_ a channel onto which to project.

    The purpose here is to get rid of the channel degree of freedom, but still retain the rest.
    This is relevant for example for plotting, where you want to plot everything but the
    channel degree of freedom.

    Parameters
    ----------
    grid: wavepacket.grid.Grid
        The source grid from which to transform.

    Raises
    ------
    wavepacket.BadGridError
        Raised if the grid has no channel degree of freedom, multiple of them,
        or no other degree of freedom.
    """

    def __init__(self, grid: Grid):
        channel_dof = grid.get_single_channel_dof()
        if channel_dof is None:
            raise wp.BadGridError("Transformation requires a grid with a single channel dof.")
        if len(grid.dofs) == 1:
            raise wp.BadGridError("Transformation requires more than one degree of freedom.")

        dof_index = grid.dofs.index(channel_dof)
        before = list(grid.dofs[:dof_index])
        after = list(grid.dofs[dof_index + 1 :])

        points_before = math.prod([dof.size for dof in before])  # math.prod([]) == 1 !
        points = channel_dof.size
        points_after = math.prod([dof.size for dof in after])
        self._fixed_shape = (points_before, points, points_after)

        super().__init__(grid, wp.grid.Grid(before + after))

    @override
    def transform(self, state: State, **kwargs: Any) -> State:
        """
        Transforms the wave function

        Parameters
        ----------
        state: wavepacket.grid.State
            the input state to transform
        **kwargs:
            This function requires the argument "channel" to be set.
            It must be a valid index or name of the channel onto which
            the state is projected.

        Returns
        -------
        wavepacket.grid.State
            The projected state in the target grid (removed channel degree of freedom).

        Raises
        ------
        wavepacket.BadGridError
            Raised if the state is not defined on the transformation's source grid.
        wavepacket.BadStateError
            Raised if the input state is not a valid state (wave function or density operator)
        wavepacket.BadFunctionCall
            Raised if the channel argument is missing.
        wavepacket.InvalidValueError
            Raised if the channel argument does not describe a valid channel.
        """
        if state.grid is not self.source_grid:
            raise wp.BadGridError("State is defined on wrong grid.")

        if "channel" not in kwargs:
            raise wp.BadFunctionCall(
                "Transformation requires a 'channel' onto which to project."
            )

        channel = kwargs["channel"]
        # Note: We checked already that the source_grid has channel != None
        channel_index = self.source_grid.get_single_channel_dof().get_index(channel)  # type: ignore

        if state.is_wave_function():
            reshaped = np.reshape(state.data, self._fixed_shape)
            projected = reshaped[:, channel_index, :]
            result = np.reshape(projected, self.target_grid.shape)

            return wp.grid.State(self.target_grid, result)
        elif state.is_density_operator():
            reshaped = np.reshape(state.data, self._fixed_shape + self._fixed_shape)
            projected = reshaped[:, channel_index, :, :, channel_index, :]
            result = np.reshape(projected, self.target_grid.operator_shape)

            return wp.grid.State(self.target_grid, result)
        else:
            raise wp.BadStateError("Invalid state cannot be transformed.")


class SubspaceTransformation(TransformationBase):
    """
    Transform into a subspace spanned by some basis vectors.

    Note that the subspace does not retain any structure; wave functions
    are just a set of coefficients of the basis vectors with no useful
    further representation, e.g., for plotting.

    Parameters
    ----------
    subspace: list[wavepacket.grid.State]
        The list of vectors that span the subspace. They are orthonormalized
        internally. However, they must be linearly independent, otherwise the
        orthonormalization picks up noise as an additional basis vector.

    Raises
    ------
    wp.InvalidValueError
        Raised if the subspace parameter is not defined or empty, or if
        any wave function has norm zero.
    wp.BadGridError
        Raised if the subspace vectors are defined on different grids.
    wp.BadStateError
        Raised if any subspace vector is not a wave function.
    """

    def __init__(self, subspace: Sequence[State]) -> None:
        if not subspace:
            raise wp.InvalidValueError("Subspace definition required for transformation.")

        grid = subspace[0].grid
        if any(s.grid != grid for s in subspace):
            raise wp.BadGridError("Subspace vectors are not defined on a common grid.")
        if any(not s.is_wave_function() for s in subspace):
            raise wp.BadStateError("Subspace definition must only contain wave functions.")
        if any(wp.trace(s) == 0 for s in subspace):
            raise wp.InvalidValueError("Basis functions must have finite norm.")

        basis = [np.ravel(s.data) for s in wp.orthonormalize(subspace)]
        self._from_right = np.stack(basis, axis=1)
        self._from_left = self._from_right.T.conj()

        target_grid = wp.grid.Grid(wp.grid.ChannelDof(len(subspace)))
        super().__init__(grid, target_grid)

    @override
    def transform(self, state: State, **kwargs: Any) -> State:
        if state.grid is not self.source_grid:
            raise wp.BadGridError("State is defined on wrong grid.")

        if state.is_wave_function():
            tmp = state.data.ravel()
            transformed = np.tensordot(self._from_left, tmp, axes=(1, 0))
        elif state.is_density_operator():
            tmp = np.reshape(state.data, (self.source_grid.size, self.source_grid.size))
            left_side = np.tensordot(self._from_left, tmp, axes=(1, 0))
            transformed = np.tensordot(left_side, self._from_right, axes=(1, 0))
        else:
            raise wp.BadStateError("Invalid state cannot be transformed.")

        return State(self.target_grid, transformed)

    def transform_back(self, state: State) -> State:
        if state.grid is not self.target_grid:
            raise wp.BadGridError("State is defined on wrong grid.")

        if state.is_wave_function():
            tmp = np.tensordot(state.data, self._from_right, axes=(0, 1))
            transformed = np.reshape(tmp, self.source_grid.shape)
        elif state.is_density_operator():
            tmp = np.tensordot(self._from_right, state.data, axes=(1, 0))
            tmp = np.tensordot(tmp, self._from_left, axes=(1, 0))
            transformed = np.reshape(tmp, self.source_grid.operator_shape)
        else:
            raise wp.BadStateError("Invalid state cannot be transformed.")

        return wp.grid.State(self.source_grid, transformed)
