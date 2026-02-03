from typing import override

import wavepacket.typing as wpt

from ..grid import DofBase


class DummyDof(DofBase):
    """
    Empty DOF without transformations.
    """

    def __init__(self, dvr_array: wpt.RealData, fbr_array: wpt.RealData) -> None:
        super().__init__(dvr_array, fbr_array)

    @override
    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        return data

    @override
    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        return data

    @override
    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        return data

    @override
    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        return data
