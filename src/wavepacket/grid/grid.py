import math
from collections.abc import Sequence
from typing import Final, Iterable

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt

from .channel_dof import ChannelDof
from .dofbase import DofBase


class Grid:
    """
    Definition of a one- or multidimensional grid.

    This class collects multiple :py:class:`wavepacket.grid.DofBase`-derived objects,
    each corresponding to a one-dimensional basis expansion, and represents the resulting
    one- or multidimensional grid that you can operate with.

    Parameters
    ----------
    dofs : Iterable[wp.grid.DofBase] | DofBase
        The degree(s) of freedom that make up the grid. The order determines the order of
        the coefficient array indices.
    names: Iterable[str] | str | None, optional
        If supplied, a list of names for the individual degrees of freedom. These names can
        be used instead of indices to reference the degrees of freedom later.

    Attributes
    ----------
    shape: tuple[int, ...], readonly
        The shape of a NumPy array that describes a wave function.
    operator_shape: tuple[int, ...], readonly
        The shape of a NumPy array that describes an operator

        An example of such a matrix would be the coefficient array for a density operator.
        The operator dimensions are the dimensions of the grid concatenated with themselves.
        For example, a grid with dimensions (5, 4) has operator dimensions (5, 4, 5, 4).
    size: int, readonly
        The total number of grid points
    ndim: int, readonly
        The number of dimensions / degrees of freedom of the grid.
    dofs: Sequence[wavepacket.grid.DofBase], readonly
        A list of degrees of freedom that describe the degrees of freedom of the grid
    dof_names: Sequence[str], readonly
        A list of names for the degrees of freedom. If not supplied in __init__(), these
        are just generic names "0", "1", ...

    Raises
    ------
    wp.InvalidValueError
        If no degrees of freedom are supplied.
    """

    def __init__(
        self, dofs: Iterable[DofBase] | DofBase, names: Iterable[str] | str | None = None
    ) -> None:
        if dofs is None:
            raise wp.InvalidValueError("A grid needs at least one degree of freedom defined.")

        if isinstance(dofs, DofBase):
            dofs = [dofs]

        self.shape: Final[tuple[int, ...]] = tuple(dof.size for dof in dofs)
        self.operator_shape: Final[tuple[int, ...]] = self.shape + self.shape
        self.size: Final[int] = math.prod(dof.size for dof in dofs)
        self.dofs: Final[Sequence[wp.grid.DofBase]] = list(dofs)
        self.ndim: Final[int] = len(dofs)

        # names need dofs for processing (we want a len() function)
        if names is None:
            names = [str(n) for n in range(self.ndim)]
        if isinstance(names, str):
            names = [names]

        self.dof_names: Final[Sequence[str]] = list(names)

        if len(self.dof_names) != len(self.dofs):
            raise wp.InvalidValueError(
                "Number of names differs from number of degrees of freedom."
            )
        if any(not n for n in self.dof_names):
            raise wp.InvalidValueError("Empty names are not allowed.")
        if len(set(self.dof_names)) != len(self.dof_names):
            raise wp.InvalidValueError("Duplicate names are not allowed.")

        # fast, convenient lookup
        self._index_lookup = {n: n for n in range(-self.ndim, self.ndim)}
        self._index_lookup.update({name: index for index, name in enumerate(self.dof_names)})

    def get_index(self, index: wpt.IndexOrName) -> int:
        """
        Translates a dof index or name into the dof index.

        This piece of machinery allows functions to conveniently refer to a
        degree of freedom using either its index or its name. (Valid) indices
        are returned untouched, for names the index is returned.

        Invalid names or indices raise an exception.
        """
        try:
            return self._index_lookup[index]
        except KeyError:
            raise wp.InvalidValueError(
                f"Index '{index}' does not refer to an existing degree of freedom."
            )

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
        if index < -self.ndim or index >= self.ndim:
            raise IndexError("Index of degree of freedom out of bounds.")

        if index < 0:
            return index + self.ndim
        else:
            return index

    def broadcast(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
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
        new_shape = self.ndim * [1]
        new_shape[index] = self.dofs[index].size
        return np.reshape(data, new_shape)

    def operator_broadcast(
        self, data: wpt.ComplexData, dof_index: int, is_ket: bool = True
    ) -> wpt.ComplexData:
        """
        Similar to broadcast, but blows up the array into a form suitable for multiplication with operators.

        Meant for Wavepacket-internal use. Note that this function is rather slow,
        and should not be used in tight loops.

        See :py:meth:`broadcast` for a description of the problem. For the (5, 4, 3)
        grid shape, this function would blow up the potential array into a shape
        (1, 4, 1, 1, 1, 1) or (1, 1, 1, 1, 4, 1). The is_ket parameter switches between the
        two variants, we call the first three indices "ket" and the latter three "bra" indices.
        """
        new_shape = (2 * self.ndim) * [1]

        shape_index = self.normalize_index(dof_index)
        if not is_ket:
            shape_index += self.ndim

        new_shape[shape_index] = self.dofs[dof_index].size
        return np.reshape(data, new_shape)

    def get_single_channel_dof(self) -> ChannelDof | None:
        """
        Returns the single degree of freedom if the grid has one or None.

        This is a convenience shorthand mainly for some functionality (e.g., plotting)
        to check if special handling of channels needs to be done, and to simplify the
        query for the channels.

        Returns
        -------
        wavepacket.grid.ChannelDof
            The degree of freedom that describes the channel.
            If the grid has no or multiple channel degrees of freedom, this function returns None.
        """
        channel_dofs = [dof for dof in self.dofs if isinstance(dof, ChannelDof)]
        if len(channel_dofs) == 1:
            return channel_dofs[0]
        else:
            return None
