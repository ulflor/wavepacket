import numpy as np
import wavepacket as wp
import wavepacket.typing as wpt
from .operatorbase import OperatorBase


class Channel(OperatorBase):
    """
    Operator that projects onto a given channel.

    This operator requires a grid with a {py:class}`wavepacket.grid.ChannelDof`,
    and projects the state onto a specific channel. Formally, it is equivalent
    to a "potential" along the channel degree of freedom that consists of zeros
    except for the specific channel.

    Parameters
    ----------
    grid: wavepacket.grid.Grid
        The grid on which the operator acts.
    channel: int | str
        The channel on which to project. This can be given either as index
        or the name of the channel.

    Raises
    ------
    wp.BadGridError
        Raised if the grid has no or multiple channel degrees of freedoms.
    wp.InvalidValueError
        Raised if the channel parameter does not describe a valid channel
        (index out of bounds or name unknown).
    """

    def __init__(self, grid: wp.grid.Grid, channel: int | str):
        channel_dof = grid.get_single_channel_dof()
        if channel_dof is None:
            raise wp.BadGridError("Grid has no channel degree of freedom.")

        self._ket_index = grid.dofs.index(channel_dof)
        self._bra_index = len(grid.dofs) + self._ket_index

        self._channel = channel_dof.get_index(channel)
        if self._channel is None:
            raise wp.InvalidValueError(
                f"'{channel}' does not reference a valid channel (index out of bounds or name unknown)."
            )

        super().__init__(grid, False)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.swapaxes(psi, 0, self._ket_index)
        result = np.zeros_like(tmp)
        result[self._channel, ...] = tmp[self._channel, ...]
        return np.swapaxes(result, 0, self._ket_index)

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.swapaxes(rho, 0, self._ket_index)
        result = np.zeros_like(tmp)
        result[self._channel, ...] = tmp[self._channel, ...]
        return np.swapaxes(result, 0, self._ket_index)

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.swapaxes(rho, 0, self._bra_index)
        result = np.zeros_like(tmp)
        result[self._channel, ...] = tmp[self._channel, ...]
        return np.swapaxes(result, 0, self._bra_index)


class Coupling(OperatorBase):
    """
    Operator that describes a coupling between two channels.

    The coupling is real and symmetric in the two channels, that is,
    a coupling between channels 1 and 2 is identical to a coupling between channels 2 and 1.

    Parameters
    ----------
    grid: wavepacket.grid.Grid
        The grid on which the operator acts.
    from_channel: int | str
        The index or the name of the first coupled channel.
    to_channel: int | str
        The index or the name of the second coupled channel.

    Raises
    ------
    wp.BadGridError
        Raised if the grid has no or multiple channel degrees of freedoms.
    wp.InvalidValueError
        Raised if the from or to channel does not describe a valid channel,
        or if both channels are identical.
    """

    def __init__(self, grid: wp.grid.Grid, from_channel: int | str, to_channel: int | str):
        channel_dof = grid.get_single_channel_dof()
        if channel_dof is None:
            raise wp.BadGridError("Grid has no channel degree of freedom.")

        self._from_index = channel_dof.get_index(from_channel)
        self._to_index = channel_dof.get_index(to_channel)

        if self._from_index is None or self._to_index is None:
            raise wp.InvalidValueError(
                f"'{from_channel}' and/or '{to_channel}' do not reference a valid channel (index out of bounds or name unknown)."
            )
        if channel_dof.dvr_points[self._from_index] == channel_dof.dvr_points[self._to_index]:
            raise wp.InvalidValueError(
                "Coupling of a channel with itself is not supported. Use 'wavepacket.operator.Channel' for that."
            )

        data = np.zeros((channel_dof.size, channel_dof.size))
        data[self._from_index, self._to_index] = 1.0
        data[self._to_index, self._from_index] = 1.0

        self._ket_index = grid.dofs.index(channel_dof)
        self._bra_index = self._ket_index + len(grid.dofs)
        self._matrix = data

        super().__init__(grid, False)

    def apply_to_wave_function(self, psi: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.swapaxes(psi, 0, self._ket_index)
        result = np.zeros_like(tmp)
        result[self._to_index, ...] = tmp[self._from_index, ...]
        result[self._from_index, ...] = tmp[self._to_index, ...]
        return np.swapaxes(result, 0, self._ket_index)

    def apply_from_left(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.swapaxes(rho, 0, self._ket_index)
        result = np.zeros_like(tmp)
        result[self._to_index, ...] = tmp[self._from_index, ...]
        result[self._from_index, ...] = tmp[self._to_index, ...]
        return np.swapaxes(result, 0, self._ket_index)

    def apply_from_right(self, rho: wpt.ComplexData, t: float) -> wpt.ComplexData:
        tmp = np.swapaxes(rho, 0, self._bra_index)
        result = np.zeros_like(tmp)
        result[self._to_index, ...] = tmp[self._from_index, ...]
        result[self._from_index, ...] = tmp[self._to_index, ...]
        return np.swapaxes(result, 0, self._bra_index)
