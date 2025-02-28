import wavepacket as wp
import wavepacket.typing as wpt


class DummyOperator(wp.operator.OperatorBase):
    def __init__(self, grid: wp.grid.Grid):
        super().__init__(grid)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        raise wp.BadFunctionCall("Should be patched.")

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        raise wp.BadFunctionCall("Should be patched.")

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        raise wp.BadFunctionCall("Should be patched.")


