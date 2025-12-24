import numpy as np
import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.typing as wpt


def dummy_func(data: wpt.RealData) -> wpt.RealData:
    return data


@pytest.fixture
def op(grid_2d) -> wp.operator.Potential1D:
    return wp.operator.Potential1D(grid_2d, 0, dummy_func)


def test_reject_invalid_degree_of_freedom(grid_2d):
    with pytest.raises(IndexError):
        wp.operator.Potential1D(grid_2d, 2, dummy_func)


def test_apply_to_data(grid_2d, op):
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)
    dvr_points = grid_2d.dofs[0].dvr_points

    result = op.apply_to_wave_function(psi.data, 0.0)
    expected = psi.data * np.reshape(dvr_points, (len(dvr_points), 1))
    assert_allclose(result, expected, 1e-12)

    result = op.apply_from_left(rho.data, 0.0)
    expected = rho.data * np.reshape(dvr_points, (len(dvr_points), 1, 1, 1))
    assert_allclose(result, expected, 1e-12)

    result = op.apply_from_right(rho.data, 0.0)
    expected = rho.data * np.reshape(dvr_points, (1, 1, len(dvr_points), 1))
    assert_allclose(result, expected, 1e-12)


def test_negative_indices(grid_2d):
    op_positive = wp.operator.Potential1D(grid_2d, 0, dummy_func)
    op_negative = wp.operator.Potential1D(grid_2d, -2, dummy_func)
    psi = wp.testing.random_state(grid_2d, 42)
    rho = wp.builder.pure_density(psi)

    result_positive = op_positive.apply_to_wave_function(psi.data, 0.0)
    result_negative = op_negative.apply_to_wave_function(psi.data, 0.0)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)

    result_positive = op_positive.apply_from_left(rho.data, 0.0)
    result_negative = op_negative.apply_from_left(rho.data, 0.0)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)

    result_positive = op_positive.apply_from_right(rho.data, 0.0)
    result_negative = op_negative.apply_from_right(rho.data, 0.0)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)


def test_cutoff(grid_1d):
    cutoff = 5
    raw_potential = wp.operator.Potential1D(grid_1d, 0, dummy_func)
    cut_potential = wp.operator.Potential1D(grid_1d, 0, dummy_func, cutoff)
    psi = wp.builder.zero_wave_function(grid_1d) + 1

    raw_potential_values = raw_potential.apply(psi, 0.0).data
    cut_potential_values = cut_potential.apply(psi, 0.0).data

    assert np.any(raw_potential_values > cutoff)
    assert np.all(cut_potential_values <= cutoff)
