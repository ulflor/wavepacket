import numbers
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from .grid import Grid
from ..exceptions import BadGridError, BadStateError


@dataclass(frozen=True)
class State:
    grid: Grid
    data: npt.NDArray[np.complexfloating | np.floating]

    def is_wave_function(self) -> bool:
        return self.data.shape == self.grid.shape

    def is_density_operator(self) -> bool:
        return self.data.shape == self.grid.operator_shape

    def __add__(self, state_or_number):
        if isinstance(state_or_number, numbers.Number):
            return State(self.grid, self.data + state_or_number)
        else:
            State._check(self, state_or_number)
            return State(self.grid, self.data + state_or_number.data)

    def __radd__(self, number):
        assert isinstance(number, numbers.Number)

        return State(self.grid, self.data + number)

    def __sub__(self, state_or_number):
        if isinstance(state_or_number, numbers.Number):
            return State(self.grid, self.data - state_or_number)
        else:
            State._check(self, state_or_number)
            return State(self.grid, self.data - state_or_number.data)

    def __rsub__(self, number):
        assert isinstance(number, numbers.Number)

        return State(self.grid, number - self.data)

    def __mul__(self, number):
        assert isinstance(number, numbers.Number)

        return State(self.grid, self.data * number)

    def __rmul__(self, number):
        assert isinstance(number, numbers.Number)

        return self * number

    def __truediv__(self, number):
        assert isinstance(number, numbers.Number)

        return State(self.grid, self.data / number)

    @staticmethod
    def _check(state1, state2) -> None:
        if state1.grid != state2.grid:
            raise BadGridError("Binary operations between states with different grids not supported.")

        if state1.data.shape != state2.data.shape:
            raise BadStateError("Binary operations between states of different size not supported.")

