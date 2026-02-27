from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

import wavepacket as wp
from ._utilities import get_potential_values


class BaseContourPlot2D(ABC):
    """
    Base class for 2D contour plots.

    This class contains some common code and attributes for the different plots.
    Derived classes are mainly concerned with layout and specific extensions.

    Attributes
    ----------
    xlim: tuple[float,float]
        The range (min, max) of the x-axis.
    ylim: tuple[float, float]
        The range (min, max) of the y-axis.
    contours: array_like[float]
        Levels at which contour lines for the density should be drawn.
    potential_contours: array_like[float]
        Levels at which contour lines for the potential should be drawn.
    """
    def __init__(self, state: wp.grid.State, potential: wp.operator.OperatorBase | None = None):
        assert len(state.grid.dofs) == 2
        assert potential is None or potential.grid == state.grid

        x = state.grid.dofs[0].dvr_points
        y = state.grid.dofs[1].dvr_points
        density = wp.dvr_density(state)

        xrange = x[-1] - x[0]
        yrange = y[-1] - y[0]
        self.xlim = (x[0] - 1e-2 * xrange, x[-1] + 1e-2 * xrange)
        self.ylim = (y[0] - 1e-2 * yrange, y[-1] + 1e-2 * yrange)

        max_density = density.max()
        self.contours = np.linspace(0, max_density, 15)

        if potential is None:
            self._potential = None
            self.potential_contours = []
        else:
            self._potential = potential
            potential_values = get_potential_values(potential, 0)
            self.potential_contours = np.linspace(potential_values.min(),
                                                  potential_values.max(), 15)

    @abstractmethod
    def plot(self, state: wp.grid.State, t: float) -> plt.Axes:
        """
        Draws the contour plot and returns the axis of the contour plot.

        If a potential was supplied, its contours are also drawn.
        Specific plots may also populate other axes, these are not normally accessible here.
        If the supplied potential is time-dependent, it is plotted at the given time.

        Parameters
        ----------
        state: wp.grid.State
            The state whose density is plotted.
        t: float
            The time at which the state applies.

        Returns
        -------
        plt.Axes
            The Matplotlib axes object on which we plotted the state for possible
            further manipulation.
        """
        raise NotImplementedError("Abstract base method should not be called")

    def _contour(self, axes: plt.Axes, state: wp.grid.State, t: float) -> None:
        """
        Internal plotting function that actually draws the contours on a given Axes.
        """
        assert len(state.grid.dofs) == 2
        assert self._potential is None or state.grid == self._potential.grid

        axes.clear()
        axes.set_xlim(self.xlim)
        axes.set_ylim(self.ylim)

        x = state.grid.dofs[0].dvr_points
        y = state.grid.dofs[1].dvr_points
        z = wp.dvr_density(state)

        if self._potential is not None:
            potential_values = get_potential_values(self._potential, t)
            axes.contour(x, y, potential_values.T,
                         levels=self.potential_contours, colors='k',
                         linewidths=0.5, linestyles='--')

        axes.contour(x, y, z.T,
                     levels=self.contours, colors='b',
                     linewidths=1, linestyles='-')


class ContourPlot2D(BaseContourPlot2D):
    """
    Draws a two-dimensional contour plot and the reduced densities along each DOF.

    This plot draws not one, but three axis: The actual contour plot, but also
    two additional axes were the reduced density along the first/second degree
    of freedom are drawn.

    Customization is done almost exclusively through the attributes of the
    base class :py:class:`BaseContourPlot2D`.

    Parameters
    ----------
    state: wp.grid.State
        This state exclusively serves for deduction of parameters (contour levels etc.).
        It should be either somewhat representative, or you need to adjust the
        attributes later.
    potential: wp.operator.OperatorBase | None, default=None
        The potential to draw together with the contours of the state.
        May be time-dependent, in which case it is drawn at the corresponding point in time.

    Attributes
    ----------
    figure: plt.Figure
        The figure of the plot. Can be used for global changes, such as resizing, saving etc.
    max_marginals: tuple[float, float]
        The upper range of the marginals, i.e., reduced densities.
        The default is to initialize them from the supplied state.
    """
    def __init__(self, state: wp.grid.State, potential: wp.operator.OperatorBase | None = None):
        super().__init__(state, potential)

        # Create a square main plot and the two axes for the reduced densities
        # This seems to be a case that seems not well-supported by the layout engine,
        # so we use fixed layouts here
        self.figure = plt.figure(figsize=(8, 8), layout='none')
        self._axes = self.figure.add_axes((0.1, 0.3, 0.6, 0.6))
        self._ax_bottom = self.figure.add_axes((0.1, 0.1, 0.6, 0.2), sharex=self._axes)
        self._ax_right = self.figure.add_axes((0.7, 0.3, 0.2, 0.6), sharey=self._axes)

        self.max_marginals = (wp.dvr_density(state, 0).max(),
                              wp.dvr_density(state, 1).max())

    def plot(self, state: wp.grid.State, t: float) -> plt.Axes:
        # Plot the 2D contour plot
        self._contour(self._axes, state, t)

        # Plot the reduced density along x
        self._ax_bottom.clear()
        self._ax_bottom.set_ylim(0, self.max_marginals[0])

        x = state.grid.dofs[0].dvr_points
        reduced_density_x = wp.dvr_density(state, 0)
        self._ax_bottom.plot(x, reduced_density_x, 'b-')

        # Plot the reduced density along y
        self._ax_right.clear()
        self._ax_right.set_xlim(self.max_marginals[1], 0)

        y = state.grid.dofs[1].dvr_points
        reduced_density_y = wp.dvr_density(state, 1)
        self._ax_right.plot(reduced_density_y, y, 'b-')

        # Styling: Remove superfluous ticks and such
        self._axes.set_title(f"t = {t:.4g} a.u.")
        self._axes.xaxis.set_tick_params(labelbottom=False, tickdir='in', top=True)
        self._axes.yaxis.set_tick_params(labelleft=False, tickdir='in', right=True)

        self._ax_bottom.set_yticks([])
        self._ax_bottom.set_xlabel('x (a.u.)')

        self._ax_right.set_xticks([])
        self._ax_right.yaxis.set_tick_params(labelleft=False, labelright=True, left=False, right=True)
        self._ax_right.set_ylabel('y (a.u.)')
        self._ax_right.yaxis.set_label_position('right')

        return self._axes
