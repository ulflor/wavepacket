from typing import Sequence

import numpy as np

from .dofbase import DofBase
from ..typing import ComplexData
from ..utils import InvalidValueError


class Grid:
    def __init__(self, dofs: Sequence[DofBase] | DofBase):
        if dofs is None:
            raise InvalidValueError("A grid needs at least one Degree of freedom defined.")

        if isinstance(dofs, DofBase):
            self._dofs = [dofs]
        else:
            self._dofs = list(dofs)

        self._shape = tuple(len(dof.dvr_array) for dof in self._dofs)

    @property
    def shape(self) -> tuple[int, ...]:
        # note: dvr shape
        return self._shape

    @property
    def operator_shape(self) -> tuple[int, ...]:
        return self._shape + self._shape

    def __getitem__(self, index: int) -> DofBase:
        return self._dofs[index]

    def broadcast(self, data: ComplexData, index: int) -> ComplexData:
        # Note: rather slow, only use for precomputation
        new_shape = len(self._dofs) * [1]
        new_shape[index] = len(self._dofs[index].dvr_array)
        return np.reshape(data, new_shape)

    def operator_broadcast(self, data: ComplexData, dof_index: int, is_ket: bool = True) -> ComplexData:
        # Note: rather slow, only use for precomputation
        new_shape = (2 * len(self._dofs)) * [1]

        if is_ket:
            shape_index = dof_index
            if dof_index < 0:
                shape_index -= len(self._dofs)
        else:
            shape_index = dof_index + len(self._dofs)

        new_shape[shape_index] = len(self._dofs[dof_index].dvr_array)
        return np.reshape(data, new_shape)