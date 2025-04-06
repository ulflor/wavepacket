from abc import ABC, abstractmethod
from collections.abc import Sequence
import typing

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from ..grid import Grid, State


class OperatorBase(ABC):
    """
    Base class of an operator.

    An operator can be applied to a wave function or from the left or right to a
    density operator. It is defined on a grid, and can only operate on states on that
    grid.

    Parameters
    ----------
    grid: wp.grid.Grid
        The grid on which the operator is defined.
        Particular operators may require additional parameters.

    Attributes
    ----------
    grid
    """

    def __init__(self, grid: Grid):
        self._grid = grid

    @property
    def grid(self):
        """
        Returns the grid on which the operator is defined.
        """
        return self._grid

    def apply(self, state: State, t: float) -> State:
        """
        Applies the operator onto a wave function or a density operator from the left.

        This is a convenience function if you just want to apply the operator without
        detailed knowledge of the state.

        Parameters
        ----------
        state: wp.grid.State
            The state that the operator is applied on.
        t: float
            The time at which the operator is applied. Only relevant for time-dependent
            operators.

        Returns
        -------
        wp.grid.State
            The result of applying the operator on the state.

        Raises
        ------
        wp.BadGridError
            If the state's grid does not match the grid of the operator.
        wp.BadStateError
            If the state is neither a wave function nor a density operator.

        """
        if state.grid != self._grid:
            raise wp.BadGridError("Grid of state does not match grid of operator.")

        if state.is_wave_function():
            return State(state.grid, self.apply_to_wave_function(state.data, t))
        elif state.is_density_operator():
            return State(state.grid, self.apply_from_left(state.data, t))
        else:
            raise wp.BadStateError("Cannot apply the operator to an invalid state.")

    def __add__(self, other: typing.Self) -> typing.Self:
        """
        Adds two operators and returns the result as a :py:class:`wavepacket.grid.OperatorSum`.
        """
        return OperatorSum([self, other])

    @abstractmethod
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        """
        Applies the operator on a wave function.

        This function is mainly for Wavepacket-internal use. It ignores most error
        handling (usually done by the wrapping :py:class:`wavepacket.expression.ExpressionBase`),
        and operates directly on coefficients to avoid the creation of temporary states.

        Parameters
        ----------
        psi: wpt.ComplexData
            The coefficients describing the wave function on which the operator acts.
        t: float
            The time at which the operator should be evaluated.

        Returns
        -------
        wpt.ComplexData
            The coefficients of the resulting wave function.
        """
        pass

    @abstractmethod
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        """
        Applies the operator on a density operator from the left.

        This function is mainly for Wavepacket-internal use. It ignores most error
        handling (usually done by the wrapping :py:class:`wavepacket.expression.ExpressionBase`),
        and operates directly on coefficients to avoid the creation of temporary states.

        Parameters
        ----------
        rho: wpt.ComplexData
            The coefficients describing the density operator on which the operator acts.
        t: float
            The time at which the operator should be evaluated.

        Returns
        -------
        wpt.ComplexData
            The coefficients of the resulting density operator.
        """
        pass

    @abstractmethod
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        """
        Applies the operator on a density operator from the right.

        This function is mainly for Wavepacket-internal use. It ignores most error
        handling (usually done by the wrapping :py:class:`wavepacket.expression.ExpressionBase`),
        and operates directly on coefficients to avoid the creation of temporary states.

        Parameters
        ----------
        rho: wpt.ComplexData
            The coefficients describing the density operator on which the operator acts.
        t: float
            The time at which the operator should be evaluated.

        Returns
        -------
        wpt.ComplexData
            The coefficients of the resulting density operator.
        """
        pass


class OperatorSum(OperatorBase):
    """
    An operator that represents the sum of multiple other operators.

    You do not normally construct this operator directly. It is the result of
    adding two or more operators together. All functionality is simply forwarded
    to the individual operators.

    Parameters
    ----------
    ops: Sequence[wp.operator.OperatorBase]
        The operators that should be summed up.

    Raises
    ------
    wp.BadGridError
        If the operators are defined on different grids.
    """

    def __init__(self, ops: Sequence[OperatorBase]):
        if not ops:
            raise wp.InvalidValueError("OperatorSum needs at least one operator to sum.")
        for op in ops:
            if op.grid != ops[0].grid:
                raise wp.BadGridError("All grids in a sum operator must be equal.")

        self._ops = ops
        grid = ops[0].grid
        super().__init__(grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = np.zeros(self.grid.shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_to_wave_function(psi, t)

        return result

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_left(rho, t)

        return result

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_right(rho, t)

        return result
