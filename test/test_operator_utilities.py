import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
from wavepacket.testing import assert_close


def test_require_time_for_time_dependent_operator(grid_1d):
    ti_op = wp.operator.Constant(grid_1d, 1)
    td_op = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)
    state = wp.testing.random_state(grid_1d, 1)

    wp.expectation_value(ti_op, state)
    wp.expectation_value(ti_op, state, 0.0)

    wp.expectation_value(td_op, state, 0.0)
    with pytest.raises(wp.InvalidValueError):
        wp.expectation_value(td_op, state)

    _ = [x for x in wp.diagonalize(ti_op)]
    _ = [x for x in wp.diagonalize(ti_op, 0.0)]

    _ = [x for x in wp.diagonalize(td_op, 0.0)]
    with pytest.raises(wp.InvalidValueError):
        _ = [x for x in wp.diagonalize(td_op)]


def test_expectation_value():
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(-10, 10, 256), wp.grid.PlaneWaveDof(1, 2, 3)])
    op = wp.operator.Potential1D(grid, 0, lambda x: x)

    psi_left = wp.builder.product_wave_function(grid, [wp.Gaussian(-3, fwhm=2.0), lambda x: x])
    psi_right = wp.builder.product_wave_function(grid, [wp.Gaussian(4, fwhm=2.0), lambda x: x])
    rho = 0.5 * wp.builder.pure_density(psi_left) + 0.5 * wp.builder.pure_density(psi_right)

    assert_allclose(wp.expectation_value(op, psi_left), -3, atol=1e-2)
    assert_allclose(wp.expectation_value(op, psi_right), 4, atol=1e-2)
    assert_allclose(wp.expectation_value(op, rho), 0.5, atol=1e-2)

    with pytest.raises(wp.BadGridError):
        other_grid = wp.grid.Grid(wp.grid.PlaneWaveDof(1, 2, 3))
        other_op = wp.operator.Potential1D(other_grid, 0, lambda x: x)
        wp.expectation_value(other_op, psi_left)


def test_expectation_values_forwards_time_correctly(grid_1d, monkeypatch):
    t = 17.0

    def check(data, time):
        assert time == t
        return data

    op = wp.testing.DummyOperator(grid_1d)
    monkeypatch.setattr(op, "apply_to_wave_function", check)
    monkeypatch.setattr(op, "apply_from_left", check)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.builder.pure_density(psi)

    wp.expectation_value(op, psi, t)
    wp.expectation_value(op, rho, t)


def _diagonalization_setup() -> wp.operator.OperatorBase:
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(-10, 10, 5),
                         wp.grid.SphericalHarmonicsDof(6, 1)])
    return (wp.operator.PlaneWaveFbrOperator(grid, 0, lambda p: p)
            + wp.operator.RotationalKineticEnergy(grid, 1, 10))


def test_return_eigenstates_and_values():
    op = _diagonalization_setup()

    result = [x for x in wp.diagonalize(op)]
    assert len(result) == op.grid.size

    # energies have no duplicates and are sorted
    energies = [e for (e, s) in result]
    for i in range(len(energies) - 1):
        assert energies[i + 1] - energies[i] > 1e-3

    # states are normal (duplicates are checked below)
    states = [s for (e, s) in result]
    for s in states:
        assert abs(wp.trace(s) - 1) < 1e-12

    # results are eigenvectors
    for energy, state in result:
        applied = op.apply(state, 0.0)
        expected = energy * state

        assert_close(applied, expected, 1e-12)


def test_diagonalize_with_degenerate_states(grid_1d):
    op = wp.operator.CartesianKineticEnergy(grid_1d, 0, 1.0)

    energies = [e for e, _ in wp.diagonalize(op)]

    # every state except the first and possibly last is doubly degenerate
    assert abs(energies[0]) < 1e-12
    assert abs(energies[1] - energies[2]) < 1e-12


def test_diagonalize_forwards_time_correctly(grid_1d):
    op = wp.operator.TimeDependentOperator(grid_1d, lambda t: t)

    energies_t2 = [e for e, _ in wp.diagonalize(op, 2.0)]

    assert abs(energies_t2[0] - 2.0) < 1e-12
