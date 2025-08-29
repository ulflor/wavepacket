import numpy as np

import wavepacket as wp

import pytest
from numpy.testing import assert_allclose
from wavepacket.testing import assert_close


def test_reject_invalid_states(grid_2d):
    invalid_state = wp.grid.State(grid_2d, np.ones(5))
    good_state = wp.testing.random_state(grid_2d, 47)

    with pytest.raises(wp.BadStateError):
        wp.grid.dvr_density(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.grid.fbr_density(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.grid.trace(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.grid.orthonormalize([good_state, invalid_state])


def test_wave_function_dvr_density():
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(5, 6, 3),
                         wp.grid.PlaneWaveDof(10, 12, 5)])
    psi = wp.testing.random_state(grid, 42)

    result = wp.grid.dvr_density(psi)

    dx = 1 / 3.0 * 2 / 5.0
    expected = np.abs(psi.data ** 2) / dx
    assert_allclose(result, expected, atol=1e-14, rtol=0)


def test_density_operator_dvr_density(grid_2d):
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)

    density_from_psi = wp.grid.dvr_density(psi)
    density_from_rho = wp.grid.dvr_density(rho)

    assert_allclose(density_from_rho, density_from_psi, atol=1e-14, rtol=0)


def test_wave_function_fbr_density():
    grid = wp.grid.Grid([wp.grid.SphericalHarmonicsDof(10, 2),
                         wp.grid.SphericalHarmonicsDof(12, 5)])
    psi = 2j * wp.builder.product_wave_function(grid, [wp.SphericalHarmonic(5, 2),
                                                       wp.SphericalHarmonic(7, 5)])

    density = wp.grid.fbr_density(psi)
    expected = np.zeros(grid.shape)
    expected[5 - 2, 7 - 5] = 4
    assert_allclose(expected, density, rtol=0, atol=1e-12)


def test_density_operator_fbr_density(grid_2d):
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)

    density_from_psi = wp.grid.fbr_density(psi)
    density_from_rho = wp.grid.fbr_density(rho)

    assert_allclose(density_from_psi, density_from_rho, rtol=0, atol=1e-12)


def test_trace():
    grid = wp.grid.Grid(wp.grid.PlaneWaveDof(1, 3, 4))
    psi = wp.grid.State(grid, np.array([0.5, 0.5j, 1, 2]))

    result = wp.grid.trace(psi)
    assert_allclose(result, 5.5, atol=1e-12, rtol=0)

    rho = wp.builder.pure_density(psi)
    result_rho = wp.grid.trace(rho)
    assert_allclose(result_rho, result, atol=1e-12, rtol=0)


def test_orthonormalization_invalid_input(grid_1d, grid_2d):
    state_1d = wp.testing.random_state(grid_1d, 42)
    state_2d = wp.testing.random_state(grid_2d, 43)
    zero_2d = wp.builder.zero_wave_function(grid_2d)

    with pytest.raises(wp.BadGridError):
        wp.grid.orthonormalize([state_1d, state_2d])

    with pytest.raises(wp.BadStateError):
        wp.grid.orthonormalize([state_2d, zero_2d])


def test_trivial_orthonormalization(grid_1d):
    state = wp.testing.random_state(grid_1d, 42)

    empty_result = wp.grid.orthonormalize([])
    assert len(empty_result) == 0

    single_result = wp.grid.orthonormalize([state])
    assert len(single_result) == 1
    assert 1 - 1e-12 <= wp.grid.trace(single_result[0]) <= 1 + 1e-12
    scale = single_result[0].data[0] / state.data[0]
    assert_close(single_result[0], scale * state, 1e-12)


def test_result_is_orthonormal_and_spans_same_subspace(grid_2d):
    input = [wp.testing.random_state(grid_2d, seed) for seed in range(42, 47)]

    result = wp.grid.orthonormalize(input)

    assert len(result) == len(input)
    for state in result:
        assert 1 - 1e-12 <= wp.grid.trace(state) <= 1 + 1e-12

    for i in range(len(result)):
        for j in range(i+1, len(result)):
            scalar_product = (np.conj(result[i].data) * result[j].data).sum()
            assert abs(scalar_product) < 1e-12

    # Checking for the same subspace is a bit involved, we need ot explicitly
    # calculate all scalar products.
    for original in input:
        reconstructed = wp.builder.zero_wave_function(grid_2d).data
        for v in result:
            scalar_product = (np.conj(v.data) * original.data).sum()
            reconstructed = reconstructed + scalar_product * v.data

        assert_allclose(reconstructed, original.data, rtol=0, atol=1e-12)
