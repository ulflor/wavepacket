import numpy as np
import wavepacket as wp

def test_state():
    grid = wp.grid.Grid([wp.grid.plane_wave_dof(1, 2, 3),
                         wp.grid.plane_wave_dof(4, 5, 2)])

    wave_function = wp.grid.State(grid, np.ones(grid.shape))
    density_operator = wp.grid.State(grid, np.ones(grid.operator_shape))

    assert wave_function.is_wave_function()
    assert not wave_function.is_density_operator()

    assert density_operator.is_density_operator()
    assert not density_operator.is_wave_function()