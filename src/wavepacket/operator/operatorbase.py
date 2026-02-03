from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

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
    grid : wp.grid.Grid
        The grid on which the operator is defined.
        Particular operators may require additional parameters.

    Attributes
    ----------
    grid: wp.grid.Grid
        The grid on which the operator is defined.
    time_dependent: bool
        If the operator is time-dependent or not. Some functionality may not work for time-dependent operators.
    """

    def __init__(self, grid: Grid) -> None:
        self._grid = grid

    @property
    def grid(self) -> Grid:
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
        state : wp.grid.State
            The state that the operator is applied on.
        t : float
            The time at which the operator is applied.
            Only really needed for time-dependent operators, but to keep the interface uniform,
            this parameter is required.

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

    def __neg__(self) -> 'OperatorBase':
        return self * wp.operator.Constant(self._grid, -1)

    def __add__(self, other: 'OperatorBase | complex') -> 'OperatorBase':
        if not isinstance(other, OperatorBase):
            other = wp.operator.Constant(self._grid, other)
        return OperatorSum([self, other])

    def __radd__(self, other: complex) -> 'OperatorBase':
        return self + other

    def __sub__(self, other: 'OperatorBase | complex') -> 'OperatorBase':
        return self + (-1) * other

    def __rsub__(self, other: complex) -> 'OperatorBase':
        return other + (-1) * self

    def __mul__(self, other: 'OperatorBase | complex') -> 'OperatorBase':
        if not isinstance(other, OperatorBase):
            other = wp.operator.Constant(self._grid, other)
        return OperatorProduct([self, other])

    def __rmul__(self, other: 'OperatorBase | complex') -> 'OperatorBase':
        return self * other

    @property
    @abstractmethod
    def time_dependent(self) -> bool:
        """
        Returns if the operator is time-dependent or not
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        """
        Applies the operator on a wave function.

        This function is mainly for Wavepacket-internal use. It ignores most error
        handling (usually done by the wrapping :py:class:`wavepacket.expression.ExpressionBase`),
        and operates directly on coefficients to avoid the creation of temporary states.

        Parameters
        ----------
        psi : wpt.ComplexData
            The coefficients describing the wave function on which the operator acts.
        t : float
            The time at which the operator should be evaluated.
            Only really needed for time-dependent operators, but to keep the interface uniform,
            this parameter is required.

        Returns
        -------
        wpt.ComplexData
            The coefficients of the resulting wave function.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        """
        Applies the operator on a density operator from the left.

        This function is mainly for Wavepacket-internal use. It ignores most error
        handling (usually done by the wrapping :py:class:`wavepacket.expression.ExpressionBase`),
        and operates directly on coefficients to avoid the creation of temporary states.

        Parameters
        ----------
        rho : wpt.ComplexData
            The coefficients describing the density operator on which the operator acts.
        t : float
            The time at which the operator should be evaluated.
            Only really needed for time-dependent operators, but to keep the interface uniform,
            this parameter is required.

        Returns
        -------
        wpt.ComplexData
            The coefficients of the resulting density operator.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        """
        Applies the operator on a density operator from the right.

        This function is mainly for Wavepacket-internal use. It ignores most error
        handling (usually done by the wrapping :py:class:`wavepacket.expression.ExpressionBase`),
        and operates directly on coefficients to avoid the creation of temporary states.

        Parameters
        ----------
        rho : wpt.ComplexData
            The coefficients describing the density operator on which the operator acts.
        t : float
            The time at which the operator should be evaluated.
            Only really needed for time-dependent operators, but to keep the interface uniform,
            this parameter is required.

        Returns
        -------
        wpt.ComplexData
            The coefficients of the resulting density operator.
        """
        raise NotImplementedError()


class OperatorSum(OperatorBase):
    """
    An operator that represents the sum of multiple operators.

    You do not normally construct this operator directly. It is the result of
    adding two or more operators together. All functionality is simply forwarded
    to the individual operators.

    Parameters
    ----------
    ops : Sequence[wp.operator.OperatorBase]
        The operators that should be summed up.

    Raises
    ------
    wp.BadGridError
        If the operators are defined on different grids.
    """

    def __init__(self, ops: Sequence[OperatorBase]) -> None:
        if not ops:
            raise wp.InvalidValueError("OperatorSum needs at least one operator to sum.")
        for op in ops:
            if op.grid != ops[0].grid:
                raise wp.BadGridError("All grids in a sum operator must be equal.")

        self._ops = ops
        grid = ops[0].grid
        super().__init__(grid)

    @property
    @override
    def time_dependent(self) -> bool:
        td_vals = [op.time_dependent for op in self._ops]
        return any(td_vals)

    @override
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = np.zeros(self.grid.shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_to_wave_function(psi, t)

        return result

    @override
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_left(rho, t)

        return result

    @override
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = np.zeros(self.grid.operator_shape, dtype=np.complex128)
        for op in self._ops:
            result += op.apply_from_right(rho, t)

        return result


class OperatorProduct(OperatorBase):
    """
    An operator that represents the product of multiple operators.

    You do not normally construct this operator directly. It is the result of
    multiplying (concatenating) two or more operators. All functionality is simply forwarded
    to the individual operators in the correct order.

    Parameters
    ----------
    ops : Sequence[wp.operator.OperatorBase]
        The operators that should be concatenated.

    Raises
    ------
    wp.BadGridError
        If the operators are defined on different grids.
    """

    def __init__(self, ops: Sequence[OperatorBase]) -> None:
        if not ops:
            raise wp.InvalidValueError("OperatorSum needs at least one operator to sum.")
        for op in ops:
            if op.grid != ops[0].grid:
                raise wp.BadGridError("All grids in a sum operator must be equal.")

        self._ops = ops
        super().__init__(ops[0].grid)

    @property
    @override
    def time_dependent(self) -> bool:
        td_vals = [op.time_dependent for op in self._ops]
        return any(td_vals)

    @override
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = psi
        for op in reversed(self._ops):
            result = op.apply_to_wave_function(result, t)
        return result

    @override
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = rho
        for op in reversed(self._ops):
            result = op.apply_from_left(result, t)
        return result

    @override
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        result = rho
        for op in self._ops:
            result = op.apply_from_right(result, t)
        return result
