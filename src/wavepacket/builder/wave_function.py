import math
import numpy as np
from typing import Callable, Iterable, TypeAlias, Sequence

from ..grid import Grid, State, trace
from ..typing import ComplexData, RealData
from ..utils import InvalidValueError

Generator: TypeAlias = Callable[[RealData], ComplexData]


def build_product_wave_function(grid: Grid,
                                generators: Generator | Sequence[Generator],
                                normalize: bool = True) -> State:
    generator_list = generators
    if not isinstance(generator_list, Iterable):
        generator_list = [generators]

    if len(generator_list) != len(grid.dofs):
        raise InvalidValueError("To build a wave function, you need as many generators as degrees of freedoms."
                                f"Given {len(generator_list)} generators for {len(grid.dofs)} DOFs.")

    result_data = np.ones(grid.shape, dtype=complex)
    for dof_index, generator in enumerate(generator_list):
        dof = grid.dofs[dof_index]
        array = generator(dof.dvr_points)
        array = dof.from_dvr(array, 0)

        result_data *= grid.broadcast(array, dof_index)

    result = State(grid, result_data)
    norm = math.sqrt(trace(result))
    if normalize and norm > 0:
        return result / norm
    else:
        return result
