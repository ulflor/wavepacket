import matplotlib.pyplot as plt
import numpy as np

import wavepacket as wp


class ContourPlot2D:
    def __init__(self, state: wp.grid.State, potential: wp.operator.OperatorBase | None = None):
        assert len(state.grid.dofs) == 2
        assert potential is None or potential.grid == state.grid

        self.figure, self._axes = plt.subplots()

        x = state.grid.dofs[0].dvr_points
        y = state.grid.dofs[1].dvr_points
        density = wp.dvr_density(state)

        xrange = x[-1] - x[0]
        yrange = y[-1] - y[0]
        self.xlim = [x[0] - 1e-2 * xrange, x[-1] + 1e-2 * xrange]
        self.ylim = [y[0] - 1e-2 * yrange, y[-1] + 1e-2 * yrange]

        max_density = density.max()
        self.contours = np.linspace(0, max_density, 15)

        if potential is None:
            self._potvals = None
        else:
            pot_state = potential.apply(wp.builder.unit_wave_function(state.grid), 0)
            self._potvals = np.real(pot_state.data)
            self.potential_contours = np.linspace(self._potvals.min(), self._potvals.max(), 15)

    def plot(self, state: wp.grid.State, t: float) -> plt.Axes:
        self._axes.clear()
        self._axes.set_xlim(*self.xlim)
        self._axes.set_ylim(*self.ylim)

        x = state.grid.dofs[0].dvr_points
        y = state.grid.dofs[1].dvr_points
        z = wp.dvr_density(state)

        if self._potvals is not None:
            self._axes.contour(x, y, self._potvals.T, levels=self.potential_contours,
                               colors='k', linewidths=0.5, linestyles='--')

        self._axes.contour(x, y, z.T, levels=self.contours,
                            colors='b', linewidths=1, linestyles='-')

        self._axes.set_xlabel("x [a.u.]")
        self._axes.set_ylabel("y [a.u.]")
        self._axes.set_title(f"t = {t:.4g} a.u.")

        return self._axes
