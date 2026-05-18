import numpy as np
import pytest
import wavepacket as wp

from wavepacket.testing import assert_close


def _build_grid() -> wp.grid.Grid:
    return wp.grid.Grid(
        [
            wp.grid.PlaneWaveDof(-10, 10, 5),
            wp.grid.PlaneWaveDof(0, 5, 7),
            wp.grid.SphericalHarmonicsDof(4, 1),
        ]
    )


def test_throw_on_invalid_index():
    grid = _build_grid()

    wp.grid.PartialTraceTransformation(grid, 2)
    with pytest.raises(IndexError):
        wp.grid.PartialTraceTransformation(grid, 3)

    wp.grid.PartialTraceTransformation(grid, -3)
    with pytest.raises(IndexError):
        wp.grid.PartialTraceTransformation(grid, -4)


def test_throw_on_invalid_grid(grid_1d):
    with pytest.raises(wp.BadGridError):
        wp.grid.PartialTraceTransformation(grid_1d, 0)


def test_target_grid_has_only_corresponding_dof():
    grid = _build_grid()

    transformation = wp.grid.PartialTraceTransformation(grid, 1)
    assert transformation.target_grid.dofs == [grid.dofs[1]]

    transformation = wp.grid.PartialTraceTransformation(grid, -1)
    assert transformation.target_grid.dofs == [grid.dofs[2]]


def test_reject_invalid_input():
    grid = _build_grid()
    transformation = wp.grid.PartialTraceTransformation(grid, 0)

    invalid_state = wp.grid.State(grid, np.ones(5))
    with pytest.raises(wp.BadStateError):
        transformation.transform(invalid_state)

    bad_grid_state = wp.builder.unit_wave_function(transformation.target_grid)
    with pytest.raises(wp.BadGridError):
        transformation.transform(bad_grid_state)


def test_partial_trace_of_states():
    grid = _build_grid()
    transformation = wp.grid.PartialTraceTransformation(grid, 0)

    left = wp.special.Gaussian(rms=3)
    right = wp.special.Gaussian(x=1, rms=1)
    other = wp.special.Gaussian(x=3, rms=1)
    harmonic1 = wp.special.SphericalHarmonic(1, 1)
    harmonic2 = wp.special.SphericalHarmonic(2, 1)

    # Partial tracing yields a density operator that is a coherent superposition
    # of left and right plus two incoherent additions of "other"
    psi1 = wp.builder.product_wave_function(grid, [left, left, harmonic1])
    psi2 = wp.builder.product_wave_function(grid, [right, left, harmonic1])
    psi3 = wp.builder.product_wave_function(grid, [other, left, harmonic2])
    psi = psi1 + psi2 + psi3

    target_grid = transformation.target_grid
    rho1 = wp.builder.product_wave_function(target_grid, left)
    rho2 = wp.builder.product_wave_function(target_grid, right)
    rho3 = wp.builder.product_wave_function(target_grid, other)

    result = transformation.transform(psi)
    expected_result = wp.builder.pure_density(rho1 + rho2) + wp.builder.pure_density(rho3)
    assert_close(result, expected_result, 1e-10)

    # For the density operator, let us add for fun another incoherent addition of "psi1"
    rho = wp.builder.pure_density(psi) + wp.builder.pure_density(psi1)

    result = transformation.transform(rho)
    expected_result = (
        wp.builder.pure_density(rho1 + rho2)
        + wp.builder.pure_density(rho3)
        + wp.builder.pure_density(rho1)
    )
    assert_close(result, expected_result, 1e-10)
