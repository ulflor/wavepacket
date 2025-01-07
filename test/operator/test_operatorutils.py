import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.testing


def test_expectation_value():
    grid = wp.Grid([wp.PlaneWaveDof(-10, 10, 256), wp.PlaneWaveDof(1, 2, 3)])
    op = wp.Potential1D(grid, 0, lambda x: x)

    psi_left = wp.build_product_wave_function(grid, [wp.Gaussian(-3, fwhm=2.0), lambda x: x])
    psi_right = wp.build_product_wave_function(grid, [wp.Gaussian(4, fwhm=2.0), lambda x: x])
    rho = 0.5 * wp.pure_density(psi_left) + 0.5 * wp.pure_density(psi_right)

    assert_allclose(wp.expectation_value(op, psi_left), -3, atol=1e-2)
    assert_allclose(wp.expectation_value(op, psi_right), 4, atol=1e-2)
    assert_allclose(wp.expectation_value(op, rho), 0.5, atol=1e-2)

    with pytest.raises(wp.BadGridError):
        other_grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
        other_op = wp.Potential1D(other_grid, 0, lambda x: x)
        wp.expectation_value(other_op, psi_left)


def test_forward_time_correctly(grid_1d, monkeypatch):
    t = 17.0

    def check(data, time):
        assert time == t
        return data

    op = wp.testing.DummyOperator(grid_1d)
    monkeypatch.setattr(op, "apply_to_wave_function", check)
    monkeypatch.setattr(op, "apply_from_left", check)

    psi = wp.testing.random_state(grid_1d, 42)
    rho = wp.pure_density(psi)

    wp.expectation_value(op, psi, t)
    wp.expectation_value(op, rho, t)
