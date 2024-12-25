import wavepacket as wp
from wavepacket.typing import ComplexData, RealData


class DummyDof(wp.DofBase):
    def __init__(self, dvr_array: RealData, fbr_array: RealData):
        super().__init__(dvr_array, fbr_array)

    def from_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        return data

    def to_dvr(self, data: ComplexData, index: int) -> ComplexData:
        return data

    def from_dvr(self, data: ComplexData, index: int) -> ComplexData:
        return data

    def to_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        return data
