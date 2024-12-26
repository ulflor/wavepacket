import numbers
from dataclasses import dataclass

import wavepacket as wp
import wavepacket.typing as wpt
from .grid import Grid


@dataclass(frozen=True)
class State:
    # Note: The State is _always_ in the default representation, never FBR or DVR
    grid: Grid
    data: wpt.ComplexData

    def is_wave_function(self) -> bool:
        return self.data.shape == self.grid.shape

    def is_density_operator(self) -> bool:
        return self.data.shape == self.grid.operator_shape

    def __add__(self, other):
        if isinstance(other, State):
            self._check_states(other)
            return State(self.grid, self.data + other.data)
        else:
            return State(self.grid, self.data + other)

    def __radd__(self, other: numbers.Number):
        return self + other

    def __sub__(self, other):
        if isinstance(other, State):
            self._check_states(other)
            return State(self.grid, self.data - other.data)
        else:
            return State(self.grid, self.data - other)

    def __rsub__(self, other: numbers.Number):
        return State(self.grid, other - self.data)

    def __mul__(self, other: numbers.Number):
        return State(self.grid, self.data * other)

    def __rmul__(self, other: numbers.Number):
        return self * other

    def __truediv__(self, other: numbers.Number):
        if other == 0.0:
            raise ZeroDivisionError("State cannot be divided by zero.")

        return State(self.grid, self.data / other)

    def __neg__(self):
        return State(self.grid, -self.data)

    def _check_states(self, other):
        if self.grid != other.grid:
            raise wp.BadGridError("Binary operations with states on different grids are not supported.")

        if self.data.shape != other.data.shape:
            raise wp.BadStateError("Binary operations can only be performed for two wave_functions or density operators.")
