from collections.abc import Sequence
import math

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from .dofbase import DofBase


class Grid:
    def __init__(self, dofs: Sequence[DofBase] | DofBase):
        if dofs is None:
            raise wp.InvalidValueError("A grid needs at least one Degree of freedom defined.")

        if isinstance(dofs, DofBase):
            self._dofs = [dofs]
        else:
            self._dofs = list(dofs)

        self._shape = tuple(dof.size for dof in self._dofs)

    @property
    def shape(self) -> tuple[int, ...]:
        # note: dvr shape
        return self._shape

    @property
    def operator_shape(self) -> tuple[int, ...]:
        return self._shape + self._shape

    @property
    def dofs(self) -> Sequence[DofBase]:
        return self._dofs

    @property
    def size(self) -> int:
        return math.prod(dof.size for dof in self._dofs)

    def normalize_index(self, index: int) -> int:
        if index < -len(self._dofs) or index >= len(self._dofs):
            raise IndexError("Index of degree of freedom out of bounds.")

        if index < 0:
            return index + len(self._dofs)
        else:
            return index

    def broadcast(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        # Note: rather slow, only use for precomputation
        new_shape = len(self._dofs) * [1]
        new_shape[index] = self._dofs[index].size
        return np.reshape(data, new_shape)

    def operator_broadcast(self, data: wpt.ComplexData, dof_index: int, is_ket: bool = True) -> wpt.ComplexData:
        # Note: rather slow, only use for precomputation
        new_shape = (2 * len(self._dofs)) * [1]

        shape_index = self.normalize_index(dof_index)
        if not is_ket:
            shape_index += len(self._dofs)

        new_shape[shape_index] = self._dofs[dof_index].size
        return np.reshape(data, new_shape)
