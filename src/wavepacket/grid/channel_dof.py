import numbers
from collections.abc import Sequence
from copy import deepcopy
from typing import Final

import numpy as np

import wavepacket as wp
import wavepacket.typing as wpt

from .dofbase import DofBase


class ChannelDof(DofBase):
    """
    Degree of freedom that describes a set of (coupled) channels.

    This degree of freedom is used for channels / electronic states.
    Most importantly, the concept of DVR/FBR makes no sense here, so
    DVR, weighted DVR and FBR are the same, and the transformations do nothing.

    Channels have some special functionality; while a grid can in theory
    have multiple Channel degrees of freedom, it is unclear if this is actually useful,
    and several convenience functionality is only offered for a single ChannelDof.
    Channels can be referenced either by index or, if you supply names, by
    the name of the respective channel.

    Parameters
    ----------
    channels: list[str] | int
        You can supply a list of channel names, or the number of channels.
        In the latter case, the names are automatically assigned as "0", "1" etc.
        You can use names later to reference channels, it may be advisable to choose
        them not too long.

    Attributes
    ----------
    names: list[str], readonly
        The names of the individual channels.

    Raises
    ------
    wp.InvalidValueError
        If the number of channels is not positive (if the number of channels is supplied),
        if the list of names is empty, or if any channel name is empty.
    """

    def __init__(self, channels: int | Sequence[str]):
        if isinstance(channels, numbers.Integral):
            if channels <= 0:
                raise wp.InvalidValueError(
                    f"Number of channels must be positive, got {channels}"
                )
            num_channels = channels
            names = [str(n) for n in range(num_channels)]
        else:
            if not channels:
                raise wp.InvalidValueError("Need at least one channel.")
            if any(not name for name in channels):
                raise wp.InvalidValueError("Channel names must not be empty.")
            num_channels = len(channels)
            names = deepcopy(channels)

        self.names: Final[list[str]] = names
        grid = np.arange(num_channels, dtype=float)
        super().__init__(grid, grid)

    def get_index(self, channel: wpt.IndexOrName) -> int | None:
        """
        Returns the index/number of a channel that is given as a number or a name.

        This function solves / centralizes the recurring problem that a channel can be
        identified by its index or its name, but the numerics always require the index.

        Parameters
        ----------
        channel: int | str
            The number or name of the channel

        Returns
        -------
        int | None
            If the channel exists, returns the index of the referenced channel.
            If the channel does not exist, returns None.
        """
        if isinstance(channel, numbers.Integral):
            if -self.size <= channel < self.size:
                return channel
            else:
                return None
        else:
            assert isinstance(channel, str)
            if channel in self.names:
                return self.names.index(channel)
            else:
                return None

    def from_fbr(
        self, data: wpt.ComplexData, index: int, is_ket: bool = True
    ) -> wpt.ComplexData:
        return data

    def to_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        return data

    def from_dvr(self, data: wpt.ComplexData, index: int) -> wpt.ComplexData:
        return data

    def to_fbr(
        self, data: wpt.ComplexData, index: int, is_ket: bool = True
    ) -> wpt.ComplexData:
        return data
