import math
from typing import Callable

import wavepacket as wp
import wavepacket.typing as wpt

from .operatorbase import OperatorBase


class TimeDependentOperator(OperatorBase):
    """
    Wrapper for general time-dependent functions, usually laser fields.

    This operator gets a function f(t), and simply multiplies the input state
    with the function. It gets a full function f(t), which may also be
    complex.

    Parameters
    ----------
    grid: wp.grid.Grid
        The grid on which this operator acts.
    func: Callable[[float], float]
        Functions that returns a potentially complex value for a given input time.
    """

    def __init__(self, grid: wp.grid.Grid, func: Callable[[float], complex]) -> None:
        super().__init__(grid)

        self._func = func

    @property
    def time_dependent(self) -> bool:
        return True

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._func(t) * psi

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._func(t) * rho

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        return self._func(t).conjugate() * rho


class LaserField(TimeDependentOperator):
    """
    Operator that describes a common laser field.

    A usual laser field is defined as :math:`E(t) = E_0 f(t) \\cos(\\omega t + \\phi)`
    in terms of the maximum field E_0, the pulse shape f, angular frequency omega and
    initial phase phi.

    This class is meant as a shorthand for common models. If you want to model more complex
    laser fields, define them in a separate function and supply it directly to the
    :py:class:`wavepacket.operator.TimeDependentOperator` base class.

    Parameters
    ----------
    grid: wp.grid.Grid
        The grid on which this operator acts.
    max_field: float
        The maximum field strength E_0
    shape: Callable[[float], float]
        The pulse shape f(t). For easier use, it should be normalized to a maximum value of 1.
        Wavepacket comes with several predefined shapes, such as :py:class:`wavepacket.Lorentzian`.
    omega: float
        The angular frequency omega
    phi: float, default=0
        The initial phase of the pulse
    """

    def __init__(self, grid: wp.grid.Grid, max_field: float, shape: Callable[[float], float],
                 omega: float, phi: float = 0) -> None:
        super().__init__(grid, lambda t: max_field * shape(t) * math.cos(omega * t + phi))
