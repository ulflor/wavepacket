"""
Definition of an abstract Degree of Freedom (DOF).
"""

from abc import abstractmethod, ABC

import wavepacket as wp
import wavepacket.typing as wpt


class DofBase(ABC):
    """
    Abstract base class of a one-dimensional grid.

    We assemble a multidimensional grid as direct product of one-dimensional grids.
    To distinguish these two forms of grids, we call the one-dimensional grids DOF
    (Degree of Freedom). They are defined by a grid definition in real space (which we
    call DVR) and some descriptive values for the underlying basis (the FBR).

    You need to instantiate some DOFs and assemble a multidimensional grid from
    one or more of those. The provided functionality is accessible in a more
    convenient way by higher-level helper methods, for example
    :py:func:`wavepacket.grid.dvr_density`.

    Parameters
    ----------
    dvr_points: real-valued array_like
        The grid points in real space.
    fbr_points: real-valued array_like
        This is typically used to assign useful numbers to the underlying basis.
        For example, in a plane-wave expansion, we would use the wave vectors as fbr_grid.
        The size must match that of the dvr_points.

    Attributes
    ----------
    dvr_points: real-valued NumPy array
        The grid points in real space given in the constructor.
    fbr_points: real-valued NumPy array
        The grid points of the underlying basis as given in the constructor.
    size: int
        The size of the grid (number of elements of `dvr_points`/`fbr_points`)

    See Also
    --------
    wavepacket.Grid : The multidimensional grid formed from one or more degrees of freedom.
    wavepacket.PlaneWaveDof: An implementation of a degree of freedom
                             based on a plane wave expansion.

    Notes
    -----
    We always define a grid based on the pseudo-spectral representation, also known as
    Discrete Variable Representation (DVR) in the literature. The idea is to expand a
    wave function or density operator in a basis, but also allow a lossless representation
    on certain points in real space. A hopefully comprehensive summary can be found on
    the `Wavepacket wiki <https://sf.net/p/wavepacket/wiki/Numerics.DVR>`_
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
        """
        return self._dvr_points

    @property
    def fbr_points(self) -> wpt.RealData:
        return self._fbr_points

    @property
    def size(self) -> int:
        return self._dvr_points.size

    @abstractmethod
    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        pass

    @abstractmethod
    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        pass

    @abstractmethod
    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        pass

    @abstractmethod
    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        pass
