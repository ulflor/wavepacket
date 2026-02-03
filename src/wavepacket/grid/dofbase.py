"""
Definition of an abstract Degree of Freedom (DOF).
"""

from abc import abstractmethod, ABC

import wavepacket as wp
import wavepacket.typing as wpt


class DofBase(ABC):
    """
    Abstract base class of a one-dimensional basis expansion.

    We assemble a multidimensional grid as direct product of one-dimensional grids.
    To distinguish these two types of grids, we call the one-dimensional grids DOF
    (Degree of Freedom). They are defined by a basis expansion (the FBR), its corresponding
    grid definition in real space (the DVR, see :doc:`/representations`),
    the transformation between the two representations, and
    some descriptive quantum numbers for the FBR.

    Note that the transformation functions are highly flexible, but awkward and error-prone
    to use, so they should generally be avoided outside of Wavepacket-internal code.
    In general, you should use convenience functions instead that perform the required
    transformation behind the scenes, such as :py:func:`wavepacket.grid.dvr_density`.

    Parameters
    ----------
    dvr_points : real-valued array_like
        The grid points in real space.
    fbr_points : real-valued array_like
        This is typically used to assign useful numbers to the underlying basis.
        For example, in a plane-wave expansion, we would use the wave vectors as fbr_grid.
        The size must match that of the `dvr_points`.

    Attributes
    ----------
    dvr_points
    fbr_points
    size

    Raises
    ------
    wp.InvalidValueError
        If the input grids are empty, multidimensional, or if the FBR and DVR grid do not match in size.

    See Also
    --------
    wavepacket.grid.Grid : The multidimensional grid formed from one or more degrees of freedom.
    wavepacket.grid.PlaneWaveDof: An implementation of a degree of freedom
                             based on a plane wave expansion.
    """

    def __init__(self, dvr_points: wpt.RealData, fbr_points: wpt.RealData):
        if len(dvr_points) == 0 or len(fbr_points) == 0:
            raise wp.InvalidValueError("Degrees of freedom may not be empty.")

        if dvr_points.ndim != 1 or fbr_points.ndim != 1:
            raise wp.InvalidValueError("A degree of freedom represents only one-dimensional data.")

        if dvr_points.size != fbr_points.size:
            raise wp.InvalidValueError("The DVR and FBR grids must have the same size.")

        self._dvr_points = dvr_points
        self._fbr_points = fbr_points

    @property
    def dvr_points(self) -> wpt.RealData:
        """
        Numpy array that gives the grid points in real space as supplied in the constructor.
        """
        return self._dvr_points

    @property
    def fbr_points(self) -> wpt.RealData:
        """
        Numpy array that gives the grid points of the underlying basis as supplied in the constructor.
        """
        return self._fbr_points

    @property
    def size(self) -> int:
        """
        The size of the grid (number of elements of `dvr_points`/`fbr_points`)
        """
        return self._dvr_points.size

    @abstractmethod
    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        """
        Translates a dimension of the input coefficients from the Wavepacket-default "weighted DVR"
        into the FBR.

        This function is not meant for public use. It does not handle errors explicitly,
        and is just awkward to use; you need to reach through the state abstraction and transform
        each index correctly.

        Parameters
        ----------
        data : wp.typing.ComplexData
            The input coefficients of the state to transform.
        index : int
            The index of the coefficient array that should be transformed.
        is_ket : bool, default=True
            If the index is the coefficient for a ket state (True) or a bra state.

        Returns
        -------
        wpt.ComplexData
            The appropriately transformed coefficients. You will generally not wrap the results
            into a :py:class:`wavepacket.grid.State`, because that class implicitly assumes a
            weighted DVR transformation.
        """
        raise NotImplementedError()

    @abstractmethod
    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        """
        Translates a dimension of the input coefficients from the FBR  into the Wavepacket-default
        "weighted DVR"

        This function is not meant for public use. It does not handle errors explicitly,
        and is just awkward to use; you need to reach through the state abstraction and transform
        each index correctly.

        Parameters
        ----------
        data : wp.typing.ComplexData
            The input coefficients of the state to transform.
        index : int
            The index of the coefficient array that should be transformed.
        is_ket : bool, default=True
            If the index is the coefficient for a ket state (True) or a bra state.

        Returns
        -------
        wpt.ComplexData
            The appropriately transformed coefficients. You generally need to wrap the result in
            a :py:class:`wavepacket.grid.State` before further use.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        """
        Translates a dimension of the input coefficients from the Wavepacket-default "weighted DVR"
        into the DVR.

        This function is not meant for public use. It does not handle errors explicitly,
        and is just awkward to use; you need to reach through the state abstraction and transform
        each index correctly.

        Parameters
        ----------
        data : wp.typing.ComplexData
            The input coefficients of the state to transform.
        index : int
            The index of the coefficient array that should be transformed.

        Returns
        -------
        wpt.ComplexData
            The appropriately transformed coefficients. You will generally not wrap the results
            into a :py:class:`wavepacket.grid.State`, because that class implicitly assumes a
            weighted DVR transformation.
        """
        raise NotImplementedError()

    @abstractmethod
    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        """
        Translates a dimension of the input coefficients from the DVR into the  Wavepacket-default "weighted DVR".

        This function is not meant for public use. It does not handle errors explicitly,
        and is just awkward to use; you need to reach through the state abstraction and transform
        each index correctly.

        Parameters
        ----------
        data : wp.typing.ComplexData
            The input coefficients of the state to transform.
        index : int
            The index of the coefficient array that should be transformed.

        Returns
        -------
        wpt.ComplexData
            The appropriately transformed coefficients. They need to be wrapped in a
            :py:class:`wavepacket.grid.State` before further use.
        """
        raise NotImplementedError()
