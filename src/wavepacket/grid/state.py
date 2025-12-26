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

    A state can be a wave function or a density operator.
    While invalid states can be constructed, they have no use.
    A state is comprised of three parts:

    1. The grid on which the state is defined.
    2. The expansion coefficients.
    3. The representation / basis for the expansion.
       This is always weighted DVR, see :doc:`/representations`.

    Once constructed, a State is meant to be immutable. It supports elementary
    arithmetic operations, however, to allow easy composition.

    The technical reason for a state class is to store grids together with the coefficients,
    which greatly simplifies the Wavepacket API.

    Attributes
    ----------
    grid
        The grid on which the state is defined.
    data
        The coefficients of the state.

    Examples
    --------
    States are usually generated using functions in the wavepacket.builder package.
    Those functions require a grid.
    >>> psi = wp.builder.product_wave_function(grid, wp.Gaussian(rms=0.5))
    >>> zero = wp.builder.zero_wave_function(grid)
    >>> rho = wp.builder.pure_density(psi)

    Ordinary arithmetic is possible.
    >>> psi2 = psi + zero + 1.0

    You can access the low-level details of the state.
    >>> import numpy
    >>> assert psi2.grid = grid
    >>> assert numpy.all(zero.data == 0.0)

    States are input to and output from most operations
    >>> wp.grid.trace(psi)
    1.0
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
