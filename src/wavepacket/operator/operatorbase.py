from abc import ABC, abstractmethod

from ..grid import State
from ..typing import ComplexData
from ..utils import BadStateError


class OperatorBase(ABC):
    def apply(self, state: State) -> State:
        if state.is_wave_function():
            return State(state.grid, self.apply_to_wave_function(state.data))
        elif state.is_density_operator():
            return State(state.grid, self.apply_from_left(state.data))
        else:
            raise BadStateError("Cannot apply the operator to an invalid state.")

    @abstractmethod
    def apply_to_wave_function(self, psi: ComplexData) -> ComplexData:
        pass

    @abstractmethod
    def apply_from_left(self, rho: ComplexData) -> ComplexData:
        pass

    @abstractmethod
    def apply_from_right(self, rho: ComplexData) -> ComplexData:
        pass
