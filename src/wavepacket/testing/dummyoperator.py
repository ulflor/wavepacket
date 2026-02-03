from typing import override

import wavepacket as wp
import wavepacket.typing as wpt
from ..grid import Grid
from ..operator import OperatorBase


class DummyOperator(OperatorBase):
    """
    Empty operator that throws when it is applied.

    Used for testing where we need an operator, but do not get
    as far as actually doing something.
    """

    def __init__(self, grid: Grid) -> None:
        super().__init__(grid)

    @property
    @override
    def time_dependent(self) -> bool:
        return False

    @override
    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        raise wp.BadFunctionCall("Should be patched.")

    @override
    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        raise wp.BadFunctionCall("Should be patched.")

    @override
    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        raise wp.BadFunctionCall("Should be patched.")
