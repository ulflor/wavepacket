import numpy as np

import wavepacket as wp
from wavepacket.testing import assert_close


def _expected_solution(psi_init: wp.grid.State, t: float) -> wp.grid.State:
    grid = psi_init.grid
    pot = grid.broadcast(grid.dofs[0].dvr_points, 0)
    return wp.grid.State(grid, psi_init.data * np.exp(-1j * pot * t))


def test_propagation(grid_2d):
    # Note: 2D grid to ensure that the underlying scipy solver preserves the multidimensional shape.
    op = wp.operator.Potential1D(grid_2d, 0, lambda dvr_points: dvr_points)
    eq = wp.expression.SchroedingerEquation(op)
    dt = 0.5

    solver = wp.solver.OdeSolver(eq, dt)
    psi0 = wp.testing.random_state(grid_2d, 7 * 9)
    psi = psi0

    for step in range(5):
        t_new = (step + 1) * dt
        psi = solver.step(psi, step * dt)

        assert_close(psi, _expected_solution(psi0, t_new), 1e-3)


def test_forward_kwargs_to_scipy_solver(grid_1d):
    op = wp.operator.Potential1D(grid_1d, 0, lambda dvr_points: dvr_points)
    eq = wp.expression.SchroedingerEquation(op)
    psi = wp.testing.random_state(grid_1d, 10)

    default_solver = wp.solver.OdeSolver(eq, 5.0)
    kw_solver = wp.solver.OdeSolver(eq, 5.0, method="DOP853", rtol=1e-7, atol=1e-9)

    expected = _expected_solution(psi, 5.0)
    result = default_solver.step(psi, 0.0)
    better_result = kw_solver.step(psi, 0.0)

    diff = np.abs(expected.data - result.data).sum()
    better_diff = np.abs(expected.data - better_result.data).sum()

    # very indirect check, the parameters are _way_ more accurate
    assert better_diff * 100 < diff
