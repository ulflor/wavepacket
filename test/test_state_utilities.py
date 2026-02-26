import math

import numpy as np

import wavepacket as wp

import pytest
from numpy.testing import assert_allclose
from wavepacket.testing import assert_close


def test_reject_invalid_states(grid_2d):
    invalid_state = wp.grid.State(grid_2d, np.ones(5))
    good_state = wp.testing.random_state(grid_2d, 47)

    with pytest.raises(wp.BadStateError):
        wp.dvr_density(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.fbr_density(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.trace(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.normalize(invalid_state)

    with pytest.raises(wp.BadStateError):
        wp.orthonormalize([good_state, invalid_state])

    with pytest.raises(wp.BadStateError):
        wp.population(invalid_state, good_state)

    with pytest.raises(IndexError):
        wp.dvr_density(good_state, -3)

    with pytest.raises(IndexError):
        wp.fbr_density(good_state, 2)


def test_wave_function_dvr_density():
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(5, 6, 3),
                         wp.grid.PlaneWaveDof(10, 12, 5)])
    psi = wp.testing.random_state(grid, 42)

    result = wp.dvr_density(psi)

    dx = 1 / 3.0 * 2 / 5.0
    expected = np.abs(psi.data ** 2) / dx
    assert_allclose(result, expected, atol=1e-14, rtol=0)


def test_density_operator_dvr_density(grid_2d):
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)

    density_from_psi = wp.dvr_density(psi)
    density_from_rho = wp.dvr_density(rho)

    assert_allclose(density_from_rho, density_from_psi, atol=1e-14, rtol=0)


def test_wave_function_fbr_density():
    grid = wp.grid.Grid([wp.grid.SphericalHarmonicsDof(10, 2),
                         wp.grid.SphericalHarmonicsDof(12, 5)])
    psi = 2j * wp.builder.product_wave_function(grid, [wp.special.SphericalHarmonic(5, 2),
                                                       wp.special.SphericalHarmonic(7, 5)])

    density = wp.fbr_density(psi)
    expected = np.zeros(grid.shape)
    expected[5 - 2, 7 - 5] = 4
    assert_allclose(expected, density, rtol=0, atol=1e-12)


def test_density_operator_fbr_density(grid_2d):
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)

    density_from_psi = wp.fbr_density(psi)
    density_from_rho = wp.fbr_density(rho)

    assert_allclose(density_from_psi, density_from_rho, rtol=0, atol=1e-12)


def test_reduced_dvr_and_fbr_density(grid_2d):
    grid_x = wp.grid.Grid(grid_2d.dofs[0])
    grid_y = wp.grid.Grid(grid_2d.dofs[1])

    gauss_x = wp.special.Gaussian(5, rms=3)
    gauss_y = wp.special.Gaussian(10, rms=4)

    psi_x = wp.builder.product_wave_function(grid_x, gauss_x)
    psi_y = wp.builder.product_wave_function(grid_y, gauss_y)
    psi_2d = wp.builder.product_wave_function(grid_2d, [gauss_x, gauss_y])

    assert_allclose(wp.dvr_density(psi_x), wp.dvr_density(psi_2d, 0), atol=1e-12, rtol=0)
    assert_allclose(wp.dvr_density(psi_y), wp.dvr_density(psi_2d, 1), atol=1e-12, rtol=0)

    assert_allclose(wp.fbr_density(psi_x), wp.fbr_density(psi_2d, 0), atol=1e-12, rtol=0)
    assert_allclose(wp.fbr_density(psi_y), wp.fbr_density(psi_2d, 1), atol=1e-12, rtol=0)


def test_trace():
    grid = wp.grid.Grid(wp.grid.PlaneWaveDof(1, 3, 4))
    psi = wp.grid.State(grid, np.array([0.5, 0.5j, 1, 2]))

    result = wp.trace(psi)
    assert_allclose(result, 5.5, atol=1e-12, rtol=0)

    rho = wp.builder.pure_density(psi)
    result_rho = wp.trace(rho)
    assert_allclose(result_rho, result, atol=1e-12, rtol=0)


def test_normalize(grid_1d):
    psi = 75 * wp.testing.random_state(grid_1d, 42)
    assert wp.trace(psi) > 10

    norm_psi = wp.normalize(psi)
    assert (abs(norm_psi.data) ** 2).sum() == pytest.approx(1.0, abs=1e-12)

    rho = wp.builder.pure_density(psi)
    norm_rho = wp.normalize(rho)
    assert_close(norm_rho, wp.builder.pure_density(norm_psi), 1e-12)


def test_orthonormalization_invalid_input(grid_1d, grid_2d):
    state_1d = wp.testing.random_state(grid_1d, 42)
    state_2d = wp.testing.random_state(grid_2d, 43)
    zero_2d = wp.builder.zero_wave_function(grid_2d)

    with pytest.raises(wp.BadGridError):
        wp.orthonormalize([state_1d, state_2d])

    with pytest.raises(wp.BadStateError):
        wp.orthonormalize([state_2d, zero_2d])


def test_trivial_orthonormalization(grid_1d):
    state = wp.testing.random_state(grid_1d, 42)

    empty_result = wp.orthonormalize([])
    assert len(empty_result) == 0

    single_result = wp.orthonormalize([state])
    assert len(single_result) == 1
    assert 1 - 1e-12 <= wp.trace(single_result[0]) <= 1 + 1e-12
    scale = single_result[0].data[0] / state.data[0]
    assert_close(single_result[0], scale * state, 1e-12)


def test_result_is_orthonormal_and_spans_same_subspace(grid_2d):
    input_states = [wp.testing.random_state(grid_2d, seed) for seed in range(42, 47)]

    result = wp.orthonormalize(input_states)

    assert len(result) == len(input_states)
    for state in result:
        assert 1 - 1e-12 <= wp.trace(state) <= 1 + 1e-12

    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            scalar_product = (np.conj(result[i].data) * result[j].data).sum()
            assert abs(scalar_product) < 1e-12

    # Checking for the same subspace is a bit involved, we need ot explicitly
    # calculate all scalar products.
    for original in input_states:
        reconstructed = wp.builder.zero_wave_function(grid_2d).data
        for v in result:
            scalar_product = (np.conj(v.data) * original.data).sum()
            reconstructed = reconstructed + scalar_product * v.data

        assert_allclose(reconstructed, original.data, rtol=0, atol=1e-12)


def test_invalid_population_targets(grid_1d, grid_2d):
    good_state = wp.testing.random_state(grid_1d, 1)
    wrong_grid = wp.testing.random_state(grid_2d, 2)
    density_operator = wp.builder.pure_density(good_state)

    with pytest.raises(wp.BadStateError):
        wp.population(good_state, density_operator)

    with pytest.raises(wp.BadGridError):
        wp.population(good_state, wrong_grid)


def test_population(grid_2d):
    states = wp.orthonormalize([wp.testing.random_state(grid_2d, seed) for seed in [1, 2]])
    a, target = (states[0], states[1])

    psi = 2 * a + 0.7 * target
    rho = wp.builder.pure_density(psi)
    expected = 0.7 ** 2

    psi_projection = wp.population(psi, target)
    assert_allclose(psi_projection, expected, rtol=0, atol=1e-12)

    rho_projection = wp.population(rho, target)
    assert_allclose(rho_projection, expected, rtol=0, atol=1e-12)


def test_normalize_population_target(grid_2d):
    target = wp.testing.random_state(grid_2d, 1)
    target /= math.sqrt(wp.trace(target))

    psi = 0.7 * target
    rho = wp.builder.pure_density(psi)
    expected = 0.7 ** 2

    psi_projection = wp.population(psi, 2 * target)
    assert_allclose(psi_projection, expected, rtol=0, atol=1e-12)

    rho_projection = wp.population(rho, 2 * target)
    assert_allclose(rho_projection, expected, rtol=0, atol=1e-12)
