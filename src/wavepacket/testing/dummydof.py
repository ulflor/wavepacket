import wavepacket as wp
import wavepacket.typing as wpt


class DummyDof(wp.grid.DofBase):
    def __init__(self, dvr_array: wpt.RealData, fbr_array: wpt.RealData):
        super().__init__(dvr_array, fbr_array)

    def from_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        return data

    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        return data

    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        return data

    def to_fbr(self, data: wpt.ComplexData, index: int, is_ket: bool = True) -> wpt.ComplexData:
        return data
