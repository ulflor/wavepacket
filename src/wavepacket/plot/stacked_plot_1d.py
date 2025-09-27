import matplotlib.pyplot as plt
import numpy as np

from ..builder import zero_wave_function
from ..grid import State, dvr_density
from ..operator import OperatorBase, expectation_value


class StackedPlot1D:
    """
    Helper class to stack multiple plots on top of each other.

    This class does two things: It creates a Matplotlib figure
    with multiple axes stacked on top of each other, and it
    provides a plot function that conveniently plots the density
    of a state on in subsequent of these axes.

    Customization of the plots is possible but limited for ease of use.
    The underlying grid must be one-dimensional.

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
        If the Hamiltonian is not given, the potential operator is used instead.

    Attributes
    ----------
    xlim: list[float]
        The range of the x-axis [min, max]
    ylim: list[float]
        The range of the y-axis of the plot [min, max]
    conversion_factor: float
        The factor that converts from the density to energy units. Constant 1 if no potential is plotted.
    figure: matplotlib.pylot.Figure
        The figure that we plot on.
    """

    def __init__(self, num_plots, state: State,
                 potential: OperatorBase = None, hamiltonian: OperatorBase = None):
        # First, create, layout and expose the figure
        self.figure, self._axes = plt.subplots(num_plots, 1,  sharex=True)
        self._index = 0

        self.figure.subplots_adjust(hspace=0)
        self.figure.set_figheight(self.figure.get_figheight() * (1 + num_plots // 3))

        for ax in self._axes.flat:
            ax.set_yticks([])
            ax.set_xlabel("x [a.u.]")

        # By default, span the total grid range
        assert len(state.grid.dofs) == 1
        dvr_grid = state.grid.dofs[0].dvr_points
        xrange = dvr_grid.max() - dvr_grid.min()
        self.xlim = [dvr_grid.min() - 1e-2 * xrange, dvr_grid.max() + 1e-2 * xrange]

        max_density = dvr_density(state).max()
        if potential is None:
            # We only plot the density, and ignore whatever energy the states have.
            # Set the y ranges accordingly
            self.ylim = [-1e-2 * max_density, 1.01 * max_density]
            self.conversion_factor = 1
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
            potential_values = potential.apply(zero_wave_function(state.grid) + 1).data
            min_potential = potential_values.min()
            max_potential = potential_values.max()
            energy = abs(expectation_value(self._hamiltonian, state))

            self.ylim = [min_potential, max(max_potential, energy + 0.5 * (max_potential - min_potential))]
            self.conversion_factor = (self.ylim[1] - energy) / max_density

    def plot(self, state: State, t: float) -> plt.Axes:
        """
        Plots a state, possibly together with the potential.

        If a potential was supplied, it is also plotted, and the density shifted by the
        energy given as expectation value of the Hamiltonian.

        Each call to plot populates a new plot (axes); if no more axes are left,
        the last one is overwritten.

        Parameters
        ----------
        state: wp.grid.State
            The state whose density is plotted.
        t: float
            The time at which the state applies.
            The time is plotted in the upper right corner.

        Return
        ------
        plt.Axes
            The Matplotlib axes object on which we plotted the state for possible
            further manipulation.
        """
        axes: plt.Axes = self._axes[self._index]

        axes.clear()
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
        axes.text(0.05 * self.xlim[0] + 0.95 * self.xlim[1], 0.05 * self.ylim[0] + 0.95 * self.ylim[1],
                  f"t = {t:.4} a.u.", weight="heavy",
                  horizontalalignment="right", verticalalignment="top")

        dvr_grid = state.grid.dofs[0].dvr_points

        if self._potential is None:
            # Just plot the wave function
            axes.plot(dvr_grid, dvr_density(state), 'b-')
        else:
            potential_values = self._potential.apply(zero_wave_function(state.grid) + 1).data
            energy = abs(expectation_value(self._hamiltonian, state, t) * np.ones(dvr_grid.shape))
            density = dvr_density(state)

            axes.plot(dvr_grid, potential_values, 'b-')
            axes.plot(dvr_grid, energy, 'r-')
            axes.plot(dvr_grid, energy + (self.conversion_factor * density), 'r-')

        self._index = min(self._index + 1, self._axes.size)

        return axes

