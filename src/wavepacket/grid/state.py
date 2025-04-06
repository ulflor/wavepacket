import numbers
from dataclasses import dataclass
import typing

import wavepacket as wp
import wavepacket.typing as wpt
from .grid import Grid


@dataclass(frozen=True)
class State:
    """
    This class holds the definition of a specific quantum state.

    A state can be a wave function or a density operator. Invalid states can be constructed,
    but are not used. A state is comprised of three parts:

    1. The grid on which the state is defined.
    2. The expansion coefficients.
    3. The representation / basis for the expansion.
       This is always weighted DVR, see :doc:`/representations`.

    Once constructed, a State is immutable. It supports elementary arithmetic operations,
    for more complex operations, you need to extract the coefficients, transform them,
    and wrap them in another state.

    The technical reason for a state class is to store grids together with the coefficients,
    which makes the Wavepacket API less error-prone.

    Attributes
    ----------
    grid: wp.grid.Grid
        The grid for which the state is defined.
    data: wpt.ComplexData
        The coefficients of the state.
    """
    grid: Grid
    data: wpt.ComplexData

    def is_wave_function(self) -> bool:
        """
        Returns whether the state represents a wave function.
        """
        return self.data.shape == self.grid.shape

    def is_density_operator(self) -> bool:
        """
        Returns whether the state represents a density operator.
        """
        return self.data.shape == self.grid.operator_shape

    def __add__(self, other: typing.Self | numbers.Number) -> typing.Self:
        if isinstance(other, State):
            self._check_states(other)
            return State(self.grid, self.data + other.data)
        else:
            return State(self.grid, self.data + other)

    def __radd__(self, other: numbers.Number) -> typing.Self:
        return self + other

    def __sub__(self, other: typing.Self | numbers.Number) -> typing.Self:
        if isinstance(other, State):
            self._check_states(other)
            return State(self.grid, self.data - other.data)
        else:
            return State(self.grid, self.data - other)

    def __rsub__(self, other: numbers.Number) -> typing.Self:
        return State(self.grid, other - self.data)

    def __mul__(self, other: numbers.Number) -> typing.Self:
        return State(self.grid, self.data * other)

    def __rmul__(self, other: numbers.Number) -> typing.Self:
        return self * other

    def __truediv__(self, other: numbers.Number) -> typing.Self:
        if other == 0.0:
            raise ZeroDivisionError("State cannot be divided by zero.")

        return State(self.grid, self.data / other)

    def __neg__(self) -> typing.Self:
        return State(self.grid, -self.data)

    def _check_states(self, other: typing.Self) -> None:
        if self.grid != other.grid:
            raise wp.BadGridError("Binary operations with states on different grids are not supported.")

        if self.data.shape != other.data.shape:
            raise wp.BadStateError(
                "Binary operations can only be performed for two wave_functions or density operators.")
