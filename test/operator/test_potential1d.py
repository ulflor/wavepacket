import numpy as np
import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.testing
import wavepacket.typing as wpt


def dummy_func(data: wpt.RealData) -> wpt.RealData:
    return data


@pytest.fixture
def grid() -> wp.Grid:
    return wp.Grid([wp.PlaneWaveDof(10, 20, 6),
                    wp.PlaneWaveDof(0, 10, 5)])


@pytest.fixture
def op(grid) -> wp.Potential1D:
    return wp.Potential1D(grid, 0, dummy_func)


def test_reject_invalid_degree_of_freedom(grid):
    with pytest.raises(IndexError):
        wp.Potential1D(grid, 2, dummy_func)


def test_apply_to_data(grid, op):
    psi = wp.testing.random_state(grid, 42)
    rho = wp.pure_density(psi)
    dvr_points = grid.dofs[0].dvr_points

    result = op.apply_to_wave_function(psi.data)
    expected = psi.data * np.reshape(dvr_points, (len(dvr_points), 1))
    assert_allclose(result, expected, 1e-12)

    result = op.apply_from_left(rho.data)
    expected = rho.data * np.reshape(dvr_points, (len(dvr_points), 1, 1, 1))
    assert_allclose(result, expected, 1e-12)

    result = op.apply_from_right(rho.data)
    expected = rho.data * np.reshape(dvr_points, (1, 1, len(dvr_points), 1))
    assert_allclose(result, expected, 1e-12)


def test_negative_indices(grid):
    op_positive = wp.Potential1D(grid, 0, dummy_func)
    op_negative = wp.Potential1D(grid, -2, dummy_func)
    psi = wp.testing.random_state(grid, 42)
    rho = wp.pure_density(psi)

    result_positive = op_positive.apply_to_wave_function(psi.data)
    result_negative = op_negative.apply_to_wave_function(psi.data)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)

    result_positive = op_positive.apply_from_left(rho.data)
    result_negative = op_negative.apply_from_left(rho.data)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)

    result_positive = op_positive.apply_from_right(rho.data)
    result_negative = op_negative.apply_from_right(rho.data)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)
