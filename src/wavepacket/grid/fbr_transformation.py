import numpy as np

from .dof import DofType
from .grid import Grid
from ..exceptions import BadGridError, BadStateError
from ..exceptions import InvalidValueError
from ..state import State


class FbrTransformation:
    def ket_to_fbr(self, state: State) -> State:
        pass

    def ket_from_fbr(self, state: State) -> State:
        pass

    def bra_to_fbr(self, state: State) -> State:
        pass

    def bra_from_fbr(self, state: State) -> State:
        pass


class PlaneWaveTransformation(FbrTransformation):
    def __init__(self, grid: Grid, dof_index: int):
        dof = grid.dof[dof_index]
        if dof.type != DofType.PLANE_WAVE:
            raise InvalidValueError("Plane Wave Transformation requires plane wave degree of freedom.")

        xmin = dof.dvr[0]
        n = dof.dvr.size

        self._grid = grid
        self._dof = dof_index
        self._phase = np.exp(-1j * dof.fbr * xmin) / np.sqrt(n)
        self._inverse_phase = np.conj(self._phase)

    def ket_to_fbr(self, state: State) -> State:
        if state.grid != self._grid:
            raise BadGridError("State must have same grid as transformation.")

        if state.is_wave_function():
            tmp = np.fft.fftshift(np.fft.fft(state.data, axis=self._dof), axes=self._dof)
            return State(self._grid, tmp * self._phase)
        elif state.is_density_operator():
            phase = self._grid.operator_broadcast(self._phase, self._dof)

            tmp = np.fft.fftshift(np.fft.fft(state.data, axis=self._dof), axes=self._dof)
            return State(self._grid, tmp * phase)
        else:
            raise BadStateError("Cannot transform invalid state.")
