from abc import ABC, abstractmethod
from typing import override

import matplotlib.pyplot as plt
import numpy as np

import wavepacket as wp
from wavepacket.operator import OperatorBase

from ._utilities import get_potential_values


class BasePlot1D(ABC):
    """
    Base class for the 1D plotters.

    The class contains common code for presenting and guessing the various parameters of the plots
    (x limit, y limit, conversion factors), and how to draw on a given axes. Derived classes only need
    to generate and maintain the figure and axes object(s).

    Attributes
    ----------
    xlim: tuple[float, float]
        The range (min, max) of the x-axis
    ylim: tuple[float, float]
        The range (min, max) of the y-axis of the plot
    conversion_factor: float
        The factor that converts from the density to energy units. Constant 1 if no potential is plotted.
    """

    def __init__(
        self,
        state: wp.grid.State,
        potential: OperatorBase | None = None,
        hamiltonian: OperatorBase | None = None,
    ) -> None:
        # Figure out if we have a simple plot or multiple electronic states
        # Both cases are set up similarly; instead of a unit transformation, we add a special
        # member function, though.
        channel_dof = state.grid.get_single_channel_dof()
        if channel_dof is None:
            assert len(state.grid.dofs) == 1
            self._transform = None
            self._plot_grid = state.grid
            self._num_channels = 1
            self._labels = [""]
        else:
            assert len(state.grid.dofs) == 2
            self._transform = wp.grid.ChannelProjectionTransformation(state.grid)
            self._plot_grid = self._transform.target_grid
            self._num_channels = channel_dof.size
            self._labels = channel_dof.names

        # By default, span the total grid range
        dvr_grid = self._plot_grid.dofs[0].dvr_points
        xrange = dvr_grid.max() - dvr_grid.min()
        self.xlim = (dvr_grid.min() - 1e-2 * xrange, dvr_grid.max() + 1e-2 * xrange)

        max_density = wp.dvr_density(state).max()
        if potential is None:
            # We only plot the density, and ignore whatever energy the states have.
            # Set the y ranges accordingly
            self.ylim = (-1e-2 * max_density, 1.01 * max_density)
            self.conversion_factor = 1.0
            self._potential = None
        else:
            self._potential = potential
            if hamiltonian is None:
                self._hamiltonian = potential
            else:
                self._hamiltonian = hamiltonian

            # We choose the y-range such that
            # a) the whole potential fits into the plot
            # b) the density also fits into the plot and is at least half as large as the plot
            potential_values = get_potential_values(potential, 0)
            min_potential = potential_values.min()
            max_potential = potential_values.max()
            energy = abs(wp.expectation_value(self._hamiltonian, state))

            self.ylim = (
                min_potential,
                max(max_potential, energy + 0.5 * (max_potential - min_potential)),
            )
            self.conversion_factor = (self.ylim[1] - energy) / max_density

    @abstractmethod
    def plot(self, t: float, state: wp.grid.State) -> plt.Axes:
        """
        Plots a state, possibly together with the potential.

        If a potential was supplied, it is also plotted, and the density shifted by the
        energy given as expectation value of the Hamiltonian. If the supplied potential
        is time-dependent, it is plotted at the given time.

        Parameters
        ----------
        t: float
            The time at which the state applies.
        state: wp.grid.State
            The state whose density is plotted.

        Returns
        -------
        plt.Axes
            The Matplotlib axes object on which we plotted the state for possible
            further manipulation.
        """
        raise NotImplementedError()

    def _to_plot_grid(self, state: wp.grid.State, channel: int) -> wp.grid.State:
        if self._transform is None:
            return state
        else:
            return self._transform.transform(state, channel=channel)

    def _plot(self, axes: plt.Axes, t: float, state: wp.grid.State) -> None:
        """
        Internal plotting function that actually draws the density on a given Axes.
        """
        axes.clear()
        axes.set_xlim(*self.xlim)
        axes.set_ylim(*self.ylim)

        dvr_grid = self._plot_grid.dofs[0].dvr_points
        line_styles = ["b-", "r-", "g-", "k-"]

        if self._potential is None:
            # Just plot the wave functions
            for channel in range(self._num_channels):
                channel_state = self._to_plot_grid(state, channel)
                axes.plot(
                    dvr_grid,
                    wp.dvr_density(channel_state),
                    line_styles[channel % len(line_styles)],
                    label=self._labels[channel],
                )
        else:
            potential_values = get_potential_values(self._potential, t)

            for channel in range(self._num_channels):
                # transform a pseudo state with the potential as content and extract the grid again.
                tmp = wp.grid.State(state.grid, potential_values)
                channel_potential = self._to_plot_grid(tmp, channel).data
                axes.plot(
                    dvr_grid,
                    channel_potential,
                    line_styles[channel % len(line_styles)],
                    label=self._labels[channel],
                )

                channel_state = self._to_plot_grid(state, channel)
                density = wp.dvr_density(channel_state)
                trace = wp.trace(channel_state)

                if trace < 1e-3:
                    # negligible channel, do not plot, we only get noise and numerical errors
                    continue

                if self._hamiltonian.grid.get_single_channel_dof() is None:
                    energy = wp.expectation_value(self._hamiltonian, state, t).real / trace
                else:
                    prj = wp.operator.Channel(self._hamiltonian.grid, channel)
                    energy = (
                        wp.expectation_value(self._hamiltonian * prj, state, t).real / trace
                    )

                axes.plot(
                    dvr_grid,
                    energy * np.ones(dvr_grid.shape),
                    line_styles[channel % len(line_styles)],
                )
                axes.plot(
                    dvr_grid,
                    energy + (self.conversion_factor * density),
                    line_styles[channel % len(line_styles)],
                )

        axes.legend(loc="upper right")


class SimplePlot1D(BasePlot1D):
    """
    Simple plot of a one-dimensional density.

    This class creates a single plot of the density of a wave function or density operator, optionally
    together with the potential and the state's energy. It can be used for a quick and dirty way of
    showing the dynamics of a simple quantum system.

    Customization of the plot is limited, see :py:class:`BasePlot1D` for the customizable attributes.
    The underlying grid must be one-dimensional or two-dimensional with a single channel degree of freedom..

    Parameters
    ----------
    state: wp.State
        An example state for plotting; usually the initial state.
        This is only used to derive some reasonable defaults for the plots.
    potential: wp.operator.OperatorBase, optional
        The potential that is also plotted together with the state's density.
        If no potential is given, only the density is plotted.
    hamiltonian: wp.operator.OperatorBase, optional
        The Hamiltonian of the system, usually the time-independent part.
        The plotted density is shifted in y (energy) direction by the expectation value of this Hamiltonian.
        If no Hamiltonian is  given, the potential operator stands in for the Hamiltonian.

    Attributes
    ----------
    figure: matplotlib.pylot.Figure
        The figure that we plot on.
    """

    def __init__(
        self,
        state: wp.grid.State,
        potential: OperatorBase | None = None,
        hamiltonian: OperatorBase | None = None,
    ) -> None:
        self.figure, self._axes = plt.subplots()

        super().__init__(state, potential, hamiltonian)

    @override
    def plot(self, t: float, state: wp.grid.State) -> plt.Axes:
        super()._plot(self._axes, t, state)

        self._axes.set_xlabel("x [a.u.]")
        self._axes.set_title(f"t = {t:.4g} a.u.")

        return self._axes


class StackedPlot1D(BasePlot1D):
    """
    Helper class to stack multiple plots on top of each other.

    This class does two things: It creates a Matplotlib figure
    with multiple axes stacked on top of each other, and it
    provides a plot function that conveniently plots the density
    of a state on in subsequent of these axes.

    Customization of the plots is possible but limited for ease of use.
    The underlying grid must be one-dimensional. See the base class
    BasePlot1D for the attributes to tweak the plotting behavior.

    This plot helper is probably most useful for Jupyter notebooks,
    where all created figures are implicitly plotted after execution
    of a code block, and where plot "animations" are difficult.

    Parameters
    ----------
    num_plots: int
        The number of plots to stack. Should equal the number of calls to
        th plot function. If the class runs out of axes to plot onto, it
        continues plotting on the last axes.
    state: wp.grid.State
        An example state for plotting; usually the initial state.
        This is only used to derive some reasonable defaults for the plots.
    potential: wp.operator.OperatorBase, optional
        The potential that is also plotted together with the state's density.
        If no potential is given, only the density is plotted.
    hamiltonian: wp.operator.OperatorBase, optional
        The Hamiltonian of the system, usually the time-independent part.
        The plotted density is shifted in y (energy) direction by the expectation value of this Hamiltonian.
        If no Hamiltonian is  given, the potential operator stands in for the Hamiltonian.

    Attributes
    ----------
    figure: matplotlib.pylot.Figure
        The figure that we plot on.
    """

    def __init__(
        self,
        num_plots: int,
        state: wp.grid.State,
        potential: OperatorBase | None = None,
        hamiltonian: OperatorBase | None = None,
    ) -> None:
        # First, create, layout and expose the figure
        self.figure, self._axes = plt.subplots(num_plots, 1, sharex=True)
        self._index = 0

        self.figure.subplots_adjust(hspace=0)
        self.figure.set_figheight(self.figure.get_figheight() * (1 + num_plots // 3))

        for ax in self._axes.flat:
            ax.set_yticks([])
            ax.set_xlabel("x [a.u.]")

        super().__init__(state, potential, hamiltonian)

    @override
    def plot(self, t: float, state: wp.grid.State) -> plt.Axes:
        axes: plt.Axes = self._axes.flat[self._index]
        self._index = min(self._index + 1, self._axes.size - 1)

        super()._plot(axes, t, state)

        axes.text(
            0.2 * self.xlim[0] + 0.8 * self.xlim[1],
            0.05 * self.ylim[0] + 0.95 * self.ylim[1],
            f"t = {t:.4g} a.u.",
            weight="heavy",
            horizontalalignment="right",
            verticalalignment="top",
        )

        return axes
