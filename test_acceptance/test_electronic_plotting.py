import math

import matplotlib.pyplot as plt
import numpy as np
import wavepacket as wp


def trace_out_y(psi: wp.grid.State, target: wp.grid.Grid):
    result_data = np.tensordot(psi.data, np.conj(psi.data), (1, 1))
    return wp.grid.State(target, result_data)


def test_plotting(delay: int = 0):
    dof = wp.grid.PlaneWaveDof(-10, 10, 64)
    grid = wp.grid.Grid([dof, dof, wp.grid.ChannelDof(2)])
    grid_1d = wp.grid.Grid([dof, wp.grid.ChannelDof(2)])

    x = wp.operator.Potential1D(grid, 0, lambda x: x, cutoff=20)
    y = wp.operator.Potential1D(grid, 1, lambda x: x, cutoff=20)
    T_0 = wp.operator.CartesianKineticEnergy(grid, 0, mass=1, cutoff=20)
    T_1 = wp.operator.CartesianKineticEnergy(grid, 1, mass=1, cutoff=20)
    V = x * (wp.operator.Channel(grid, 0) - wp.operator.Channel(grid, 1))
    V_12 = y * wp.operator.Coupling(grid, 0, 1)

    x = wp.operator.Potential1D(grid_1d, 0, lambda x: x)
    pot_reduced = x * (wp.operator.Channel(grid_1d, 0) - wp.operator.Channel(grid_1d, 1))

    hamiltonian = T_0 + T_1 + V + V_12
    equation = wp.expression.SchroedingerEquation(hamiltonian)

    psi0 = wp.builder.product_wave_function(
        grid,
        [
            wp.special.Gaussian(-4, 4, rms=0.7 * math.sqrt(2)),
            wp.special.Gaussian(4, 0, rms=math.sqrt(2)),
            1,
        ],
    )

    solver = wp.solver.ChebychevSolver(equation, 0.2, (-30, 100))
    plotter_2d = wp.plot.ContourPlot2D(psi0, V)
    plotter_1d = wp.plot.SimplePlot1D(trace_out_y(psi0, grid_1d), pot_reduced)
    for t, psi in solver.propagate(psi0, 0.0, 10):
        reduced = trace_out_y(psi, grid_1d)

        wp.log(t, psi, precision=4, truncate=1e-6)
        plotter_1d.plot(t, reduced)
        plotter_2d.plot(t, psi)

        plt.pause(delay)


if __name__ == "__main__":
    test_plotting(1)
