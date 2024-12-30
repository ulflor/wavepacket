import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.testing


def test_expectation_value():
    grid = wp.Grid([wp.PlaneWaveDof(10, 20, 128), wp.PlaneWaveDof(1, 2, 3)])
    op = wp.Potential1D(grid, 0, lambda x: x)

    psi1 = wp.build_product_wave_function(grid, [wp.Gaussian(14, fwhm=2.0), lambda x: x])
    psi2 = wp.build_product_wave_function(grid, [wp.Gaussian(16, fwhm=2.0), lambda x: x])
    rho = 0.5 * wp.pure_density(psi1) + 0.5 * wp.pure_density(psi2)

    assert_allclose(wp.expectation_value(op, psi1), 14, atol=1e-2)
    assert_allclose(wp.expectation_value(op, psi2), 16, atol=1e-2)
    assert_allclose(wp.expectation_value(op, rho), 15, atol=1e-2)

    with pytest.raises(wp.BadGridError):
        other_grid = wp.Grid(wp.PlaneWaveDof(1, 2, 3))
        other_op = wp.Potential1D(other_grid, 0, lambda x: x)
        wp.expectation_value(other_op, psi1)


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
