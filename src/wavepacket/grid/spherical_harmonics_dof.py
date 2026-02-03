from typing import Final, override

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt

from ._utils import broadcast
from .dofbase import DofBase


class SphericalHarmonicsDof(DofBase):
    """
    One-dimensional expansion in rotational eigenstates / spherical harmonics.

    This expansion is obviously useful for systems that involve rotations.
    It takes a fixed magnetic quantum number m, because no DVR is known for
    arbitrary rotations. Wave functions are given at points :math:`(\\theta_i, \\phi=0)``,
    and the weights include the integration over :math:`\phi`.
    The FBR is an expansion in spherical harmonics :math:`Y_{lm}`,
    and the azimuthal quantum numbers :math:`l` form the FBR grid.

    Parameters
    ----------
    lmax: int
        The maximum azimuthal quantum number that is expressed by the grid.
    m: int
        The magnetic quantum number

    Raises
    ------
    wavepacket.InvalidError
        If the magnetic quantum number is too large (i.e., if lmax < abs(m))
    """

    def __init__(self, lmax: int, m: int) -> None:
        if lmax < abs(m):
            raise wp.InvalidValueError("Maximum angular momentum too small, grid has size 0.")

        self.m: Final[int] = m

        dvr_points, weights = _quadrature(lmax, m)
        fbr_points = np.linspace(abs(m), lmax, lmax - abs(m) + 1)

        super().__init__(dvr_points, fbr_points)
        self._sqrt_weights = np.sqrt(weights)

        harmonics = np.stack([wp.SphericalHarmonic(L, m)(dvr_points) for L in range(abs(m), lmax + 1)], 1)
        self._fbr2weighted = self.from_dvr(harmonics, 0)

    @override
    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        swapped_data = np.swapaxes(data, 0, index)
        result = np.tensordot(self._fbr2weighted, swapped_data, axes=(0, 0))
        return np.swapaxes(result, 0, index)

    @override
    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        swapped_data = np.swapaxes(data, 0, index)
        result = np.tensordot(self._fbr2weighted, swapped_data, axes=(1, 0))
        return np.swapaxes(result, 0, index)

    @override
    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        conversion_factor = broadcast(self._sqrt_weights, data.ndim, index)
        return data / conversion_factor

    @override
    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        conversion_factor = broadcast(self._sqrt_weights, data.ndim, index)
        return data * conversion_factor


def _quadrature(lmax: int, m: int) -> tuple[wpt.RealData, wpt.RealData]:
    """
    Performs the Gaussian quadrature for spherical harmonics with constant m.
    Returns two grids of grid points theta_i ands weights w_i such that summing over
    the gid points with the weights gives the same result as integrating products
    of spherical harmonics.
    """

    # The Gaussian quadrature for spherical harmonics may not be immediately apparent
    # and uses one piece corkscrew logic, so just a brief recap.
    #
    # Our goal is to get grid points theta_i (/phi_i) and weights w_i such that for two
    # spherical harmonics with k,l <= l_max,
    #
    # A_kl = \int_0^pi sin(theta) dtheta \int_0^{2pi} dphi Y_k^m^*(theta, phi) Y_l^m(theta, phi)
    #      = \sum_i w_i Y_k^m*(theta_i, phi_i) Y_l^m(theta_i, phi_i)
    #
    # (While the result is trivial, A_kl = \delta_kl, we can extend such integrations to
    # sums of spherical harmonics, i.e., rotational wave functions and to the DVR method).
    # Note that we keep the quantum number m constant, because the general problem is much
    # more complex, probably not even solvable.
    # First, we note that for constant m, the Condon-Shortley phase cancels, and the
    # phi-integration just yields 2pi, so we get
    #
    # A_kl = 2pi * \int_0^pi sin(theta) dtheta Y_k^m(theta, 0) Y_l^m(theta, 0)
    #
    # and we can just choose an arbitrary phi_i = 0. Next, we insert the definition of spherical
    # harmonics, and make a transition of the integral dtheta -> dx = d(cos(theta)).
    # This yields
    #
    # A_kl \propto \int_-1^1 (1-x^2)^m d_m/dx^m[P_k(x)]  d^m/dx^m[P_l(x)] dx
    #
    # in terms of Legendre polynomials P_l. There are some k,l,m-dependent prefactors,
    # but we will not need them in the end, so we just drop them here.
    #
    # d^m / dx^m [P_l(x)] = (1 * 3 * ... * (2m-1)) * C_{l-m}^{m+1/2}(x)
    #
    # in terms of ultra spherical (Gegenbauer) polynomials. Insertion yields our final result
    #
    # A_{kl} \propto \int_-1^1 (1-x^2)^m C_{k-m}^{m+1/2}(x) C_{l-m}^{m+1/2}(x) dx
    #
    # What have we achieved? Our original scalar product of spherical harmonics is, up
    # to normalization factors, the same as a product of Gegenbauer polynomials with their
    # weight function (1-x^2)^m, where the maximum polynomial order of the polynomials is
    # (l_max - m).
    #
    # Such integrals of polynomials have a (Gaussian) quadrature formula See, e.g., Tannor
    # "Introduction to quantum mechanics", section 11.3. The calculation proceeds in three steps:
    #
    # 1. Find normalized polynomials Z_{i}(x) = C_i^{m+1/2}(x) / N_{i,m} such that
    #    \int_1^1 (1-x^2)^m Z_i(x) Z_j(x) dx = \delta_{ij}
    # 2. With these normalized polynomials, calculate the matrix
    #    X_ij = \int_-1^1 (1-x^2)^m Z_i(x) * x * Z_j(x) dx
    #    and diagonalize it
    # 3. The eigenvalues are the grid points, the weights are given by the first components of
    #    the eigenvalues v as w_a = (v^{(a)}_0)^2 * \int_{-1}^1 (1-x^2)^m dx
    #
    # We do use a few tricks here that will take time to wrap your head around:
    # - We do not try to deal with normalized Gegenbauer polynomials; the resulting intermediate
    #   formulas are just ugly. Instead, we note that the original integration of the spherical
    #   harmonics can be recast as the required integral of normalized Gegenbauer polynomials,
    #   and just calculate X_ij by adding cos(theta) to the integrand and using spherical harmonics
    #   recursion relations.
    # - Watch out because the indices and the order of polynomials / number of grid points always
    #   use l-m, not l.
    # - We eventually want to absorb the factor (1-x^2)^m into the definition of our wave function /
    #   spherical harmonic, so we need to divide the weights that we get out by (1-x^2)^m.

    m = abs(m)
    num_points = lmax - m + 1

    # Construct the matrix X
    n = np.arange(1.0, num_points)
    off_diagonal = np.sqrt(n * (n + 2 * m) / (2 * n + 2 * m - 1) / (2 * n + 2 * m + 1))
    matrix = np.diagflat(off_diagonal, 1) + np.diagflat(off_diagonal, -1)

    result = np.linalg.eig(matrix)

    points = np.acos(result.eigenvalues)
    weights = result.eigenvectors[0, :] ** 2

    # Sorting is wrong, we need to fix that
    sort_indices = np.argsort(points)
    points = points[sort_indices]
    weights = weights[sort_indices]

    # The weights need two fixes.
    # First, we want to include the weight function (1-x^2)^m in the integrand, so we need to remove those again.
    # Second, there are some constant factors missing. To ge those, we simply demand that
    # \int |Y_mm(theta, phi)|^2 sin(theta) dtheta dphi = sum_k w_k * Y_mm(theta_k, 0) == 1
    y_mm = wp.SphericalHarmonic(m, m)(points)
    weights /= (1 - np.cos(points) ** 2) ** m
    weights /= np.sum(weights * y_mm ** 2)

    return points, weights
