from abc import ABC, abstractmethod

from ..grid import State


class ExpressionBase(ABC):
    @abstractmethod
    def apply(self, state: State, t: float) -> State:
        pass
