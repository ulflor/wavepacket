from collections.abc import Sequence
import math
from typing import Iterable

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt
from .dofbase import DofBase


class Grid:
    """
    Definition of a one- or multidimensional grid.

    This class collects multiple :py:class:`wavepacket.grid.DofBase`-derived objects,
    each corresponding to a one-dimensional basis expansion, and represents the resulting
    one- or multidimensional grid that you can operate with.

    Parameters
    ----------
    dofs : Sequence[wp.grid.DofBase] | DofBase
        The degree(s) of freedom that make up the grid. The order determines the order of
        the coefficient array indices.

    Attributes
    ----------
    shape
        The shape of a NumPy array that describes a wave function
    operator_shape
        The shape of a NumPy array that describes an operator
    dofs
        A list of degrees of freedom that describe the degrees of freedom of the grid
    size
        The total number of grid points

    Raises
    ------
    wp.InvalidValueError
        If no degrees of freedom are supplied.
    """

    def __init__(self, dofs: Iterable[DofBase] | DofBase) -> None:
        if dofs is None:
            raise wp.InvalidValueError("A grid needs at least one Degree of freedom defined.")

        if isinstance(dofs, DofBase):
            self._dofs = [dofs]
        else:
            self._dofs = list(dofs)

        self._shape = tuple(dof.size for dof in self._dofs)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The dimensions of the grid, for use in e.g. array creation.
        """
        # note: dvr shape
        return self._shape

    @property
    def operator_shape(self) -> tuple[int, ...]:
        """
        The dimensions of an operator matrix on this grid.

        An example of such a matrix would be the coefficient array for a density operator.
        The operator dimensions are the dimensions of the grid concatenated with themselves.
        For example, a grid with dimensions (5, 4) has operator dimensions (5, 4, 5, 4).
        """
        return self._shape + self._shape

    @property
    def dofs(self) -> list[DofBase]:
        """
        The vector of degrees of freedom that make up the grid.
        """
        return list(self._dofs)

    @property
    def size(self) -> int:
        """
        The total number of grid points.
        """
        return math.prod(dof.size for dof in self._dofs)

    def normalize_index(self, index: int) -> int:
        """
        Maps indices on the interval [0, rank_of_grid].

        Meant for Wavepacket-internal use.

        For operator matrices, we want to address bra and ket indices separately,
        and with normal Python convention, i.e., -n meaning the n-th entry from the end.
        However, an operator matrix has 2N dimensions, with the first N denoting bra indices,
        and the last N denoting ket indices. Then, the arithmetic becomes cumbersome unless we
        first map the input index onto the range [0,N].
        """
        if index < -len(self._dofs) or index >= len(self._dofs):
            raise IndexError("Index of degree of freedom out of bounds.")

        if index < 0:
            return index + len(self._dofs)
        else:
            return index

    def broadcast(self, data: wpt.AnyData, index: int) -> wpt.AnyData:
        """
        Transforms a 1D array into a more suitable form for scaling.

        Meant for Wavepacket-internal use. Note that this function is rather slow,
        and should not be used in tight loops.

        This function has a very specific purpose. Imagine, you have a grid
        with shape (5, 4, 3). On this grid, you have a wave function specified by a coefficient
        array "a" of the same shape. Now you define a potential along the second degree of freedom only.
        The potential is given by a one-dimensional array "V" of size 4.
        If you want to apply this potential to the wave function within the DVR approximation,
        the new coefficients are given as :math:`b_{ijk} = V_j a_{ijk}`.

        Unfortunately, Numpy does not offer a function for this scaling operation.
        What we can do instead is to blow up the array V into a 3D array
        of shape (1, 4, 1). then you can map the above multiplication onto Numpy's
        broadcasting rules. This reshaping is done by this function.
        """
        # Note: rather slow, only use for precomputation
        new_shape = len(self._dofs) * [1]
        new_shape[index] = self._dofs[index].size
        return np.reshape(data, new_shape)

    def operator_broadcast(self, data: wpt.AnyData, dof_index: int, is_ket: bool = True) -> wpt.AnyData:
        """
        Similar to broadcast, but blows up the array into a form suitable for multiplication with operators.

        Meant for Wavepacket-internal use. Note that this function is rather slow,
        and should not be used in tight loops.

        See :py:meth:`broadcast` for a description of the problem. For the (5, 4, 3)
        grid shape, this function would blow up the potential array into a shape
        (1, 4, 1, 1, 1, 1) or (1, 1, 1, 1, 4, 1). The is_ket parameter switches between the
        two variants, we call the first three indices "ket" and the latter three "bra" indices.
        """
        new_shape = (2 * len(self._dofs)) * [1]

        shape_index = self.normalize_index(dof_index)
        if not is_ket:
            shape_index += len(self._dofs)

        new_shape[shape_index] = self._dofs[dof_index].size
        return np.reshape(data, new_shape)
