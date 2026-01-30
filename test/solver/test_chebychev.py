import numpy as np
import scipy

import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


def test_reject_improper_constructor_parameters(grid_1d):
    op = wp.operator.Constant(grid_1d, 1)
    eq = wp.expression.SchroedingerEquation(op)

    with pytest.raises(wp.InvalidValueError):
        wp.solver.ChebychevSolver(eq, 1, (0, 0))
    with pytest.raises(wp.InvalidValueError):
        wp.solver.RelaxationSolver(op, 1, (0, 0))

    with pytest.raises(wp.InvalidValueError):
        wp.solver.ChebychevSolver(eq, 1, (1, 0))
    with pytest.raises(wp.InvalidValueError):
        wp.solver.RelaxationSolver(op, 1, (1, 0))

    td_op = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)
    with pytest.raises(wp.InvalidValueError):
        td_eq = wp.expression.SchroedingerEquation(td_op)
        wp.solver.ChebychevSolver(td_eq, 1, (0, 1))
    with pytest.raises(wp.InvalidValueError):
        wp.solver.RelaxationSolver(td_op, 1, (0, 1))


def test_order_of_expansion(grid_1d):
    op = wp.operator.Constant(grid_1d, 1)
    eq = wp.expression.SchroedingerEquation(op)
    internal_precision = 1e-12

    solver = wp.solver.ChebychevSolver(eq, 0.5, (-1.0, 1.0))
    assert scipy.special.jv(solver.order - 1, solver.alpha) >= internal_precision
    assert scipy.special.jv(solver.order, solver.alpha) < internal_precision

    relaxation = wp.solver.RelaxationSolver(op, 0.5, (-1.0, 1.0))
    assert scipy.special.iv(relaxation.order - 1, solver.alpha) >= internal_precision
    assert scipy.special.iv(relaxation.order, solver.alpha) < internal_precision


def _expected_solution(psi_init: wp.grid.State, t: float) -> wp.grid.State:
    grid = psi_init.grid
    pot = grid.broadcast(grid.dofs[0].dvr_points, 0)
    return wp.grid.State(grid, psi_init.data * np.exp(-1j * pot * t))


def test_propagation(grid_1d):
    op = wp.operator.Potential1D(grid_1d, 0, lambda dvr_points: dvr_points)
    eq = wp.expression.SchroedingerEquation(op)
    dt = 5

    pot_min = np.min(grid_1d.dofs[0].dvr_points)
    pot_max = np.max(grid_1d.dofs[0].dvr_points)
    spectrum = pot_min - 0.1 * abs(pot_min), pot_max + 0.1 * abs(pot_max)

    solver = wp.solver.ChebychevSolver(eq, dt, spectrum)
    psi0 = wp.testing.random_state(grid_1d, 1)
    psi = psi0

    for step in range(5):
        t_new = (step + 1) * dt
        psi = solver.step(psi, step * dt)

        assert_close(psi, _expected_solution(psi0, t_new), 1e-8)

    # Now try out a density operator.
    # Be aware that the spectrum of the Liouvillian is not the spectrum of the Hamiltonian!
    liouvillian = wp.expression.CommutatorLiouvillian(op)

    pot_diff = pot_max - pot_min
    rho_solver = wp.solver.ChebychevSolver(liouvillian, dt, (-pot_diff * 1.1, pot_diff * 1.1))
    rho0 = wp.builder.pure_density(psi0)

    psi = psi0
    rho = rho0

    for step in range(5):
        psi = solver.step(psi, 0.0)
        rho = rho_solver.step(rho, 0.0)

        assert_close(rho, wp.builder.pure_density(psi), 1e-8)


def _relaxed_wave_function(psi_init: wp.grid.State, t: float) -> wp.grid.State:
    grid = psi_init.grid
    pot = grid.broadcast(grid.dofs[0].dvr_points, 0)
    return wp.grid.State(grid, psi_init.data * np.exp(-pot * t))


def _relaxed_density(grid: wp.grid.Grid, t: float) -> wp.grid.State:
    pot = grid.broadcast(grid.dofs[0].dvr_points, 0)
    diagonal = np.exp(-pot * t)
    return wp.grid.State(grid, np.diag(diagonal))


def test_relaxation(grid_1d):
    op = wp.operator.Potential1D(grid_1d, 0, lambda dvr_points: dvr_points)
    dt = 5

    pot_min = np.min(grid_1d.dofs[0].dvr_points)
    pot_max = np.max(grid_1d.dofs[0].dvr_points)
    spectrum = pot_min - 0.1 * abs(pot_min), pot_max + 0.1 * abs(pot_max)

    solver = wp.solver.RelaxationSolver(op, dt, spectrum)
    psi0 = wp.testing.random_state(grid_1d, 1)
    psi = psi0

    for step in range(5):
        t_new = (step + 1) * dt
        psi = solver.step(psi, step * dt)

        assert_close(psi, _relaxed_wave_function(psi0, t_new), 1e-8)

    rho0 = wp.builder.unit_density(grid_1d)
    rho = rho0
    for step in range(5):
        t_new = (step + 1) * dt
        rho = solver.step(rho, step * dt)

        assert_close(rho, _relaxed_density(grid_1d, t_new), 1e-8)
