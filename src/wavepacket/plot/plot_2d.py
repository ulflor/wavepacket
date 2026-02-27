import matplotlib.pyplot as plt
import numpy as np

import wavepacket as wp
from ._utilities import get_potential_values


class ContourPlot2D:
    def __init__(self, state: wp.grid.State, potential: wp.operator.OperatorBase | None = None):
        assert len(state.grid.dofs) == 2
        assert potential is None or potential.grid == state.grid

        # Create a square main plot and the two axes for the reduced densities
        # This seems to be a case that seems not well-supported by the layout engine,
        # so we use fixed layouts here
        self.figure = plt.figure(figsize=(8, 8), layout='none')
        self._axes = self.figure.add_axes((0.1, 0.3, 0.6, 0.6))
        self._ax_bottom = self.figure.add_axes((0.1, 0.1, 0.6, 0.2), sharex=self._axes)
        self._ax_right = self.figure.add_axes((0.7, 0.3, 0.2, 0.6), sharey=self._axes)

        x = state.grid.dofs[0].dvr_points
        y = state.grid.dofs[1].dvr_points
        density = wp.dvr_density(state)

        xrange = x[-1] - x[0]
        yrange = y[-1] - y[0]
        self.xlim = [x[0] - 1e-2 * xrange, x[-1] + 1e-2 * xrange]
        self.ylim = [y[0] - 1e-2 * yrange, y[-1] + 1e-2 * yrange]

        max_density = density.max()
        self.contours = np.linspace(0, max_density, 15)

        self.max_marginals = (wp.dvr_density(state, 0).max(),
                              wp.dvr_density(state, 1).max())

        if potential is None:
            self._potential = None
        else:
            self._potential = potential
            potential_values = get_potential_values(potential, 0)
            self.potential_contours = np.linspace(potential_values.min(),
                                                  potential_values.max(), 15)

    def plot(self, state: wp.grid.State, t: float) -> None:
        # Plot the 2D contour plot
        self._axes.clear()
        self._axes.set_xlim(*self.xlim)
        self._axes.set_ylim(*self.ylim)

        x = state.grid.dofs[0].dvr_points
        y = state.grid.dofs[1].dvr_points
        z = wp.dvr_density(state)

        if self._potential is not None:
            potential_values = get_potential_values(self._potential, t)
            self._axes.contour(x, y, potential_values.T,
                               levels=self.potential_contours, colors='k',
                               linewidths=0.5, linestyles='--')

        self._axes.contour(x, y, z.T,
                           levels=self.contours, colors='b',
                           linewidths=1, linestyles='-')

        # Plot the reduced density along x
        self._ax_bottom.clear()
        self._ax_bottom.set_ylim(0, self.max_marginals[0])
        reduced_density_x = wp.dvr_density(state, 0)
        self._ax_bottom.plot(x, reduced_density_x, 'b-')

        # Plot the reduced density along y
        self._ax_right.clear()
        self._ax_right.set_xlim(self.max_marginals[1], 0)
        reduced_density_y = wp.dvr_density(state, 1)
        self._ax_right.plot(reduced_density_y, y, 'b-')

        # All the styling: Remove superfluous ticks and such
        self._axes.set_title(f"t = {t:.4g} a.u.")
        self._axes.xaxis.set_tick_params(labelbottom=False, tickdir='in', top=True)
        self._axes.yaxis.set_tick_params(labelleft=False, tickdir='in', right=True)

        self._ax_bottom.set_yticks([])
        self._ax_bottom.set_xlabel('x_0 (a.u.)')

        self._ax_right.set_xticks([])
        self._ax_right.yaxis.set_tick_params(labelleft=False, labelright=True, left=False, right=True)
        self._ax_right.set_ylabel('x_1 (a.u.)')
        self._ax_right.yaxis.set_label_position('right')
