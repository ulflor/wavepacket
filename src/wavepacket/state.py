import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from .grid import Grid


@dataclass(frozen=True)
class State:
    grid: Grid
    data: npt.NDArray[np.complexfloating | np.floating]

    def is_wave_function(self) -> bool:
        return self.data.shape == self.grid.shape

    def is_density_operator(self) -> bool:
        return self.data.shape == self.grid.operator_shape