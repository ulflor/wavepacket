import pytest
import numpy as np
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.typing as wpt


def dummy_func(data: wpt.RealData) -> wpt.RealData:
    return 2.5 * data


@pytest.fixture
def grid() -> wp.grid.Grid:
    return wp.grid.Grid([wp.testing.DummyDof(np.ones(3), np.ones(3)),
                         wp.grid.PlaneWaveDof(10, 20, 10),
                         wp.testing.DummyDof(np.ones(2), np.ones(2))])


@pytest.fixture
def op(grid) -> wp.operator.FbrOperator1D:
    return wp.operator.FbrOperator1D(grid, 1, dummy_func)


def test_reject_bad_indices(grid):
    with pytest.raises(IndexError):
        wp.operator.FbrOperator1D(grid, 3, dummy_func)

    with pytest.raises(IndexError):
        wp.operator.FbrOperator1D(grid, -4, dummy_func)


def test_apply_to_wave_function(op, grid):
    dof = grid.dofs[1]
    assert isinstance(dof, wp.grid.PlaneWaveDof)
    k = dof.fbr_points[4]

    psi = wp.builder.product_wave_function(grid, [dummy_func, wp.PlaneWave(k), dummy_func])

    expected = dummy_func(k) * psi.data
    got = op.apply_to_wave_function(psi.data, 0.0)
    assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_apply_to_density_operator(op, grid):
    psi = wp.testing.random_state(grid, 43)
    psi_result = op.apply(psi, 0.0)
    rho = wp.builder.pure_density(psi)

    expected_left = wp.builder.direct_product(psi_result, psi).data
    got_left = op.apply_from_left(rho.data, 0.0)
    assert_allclose(got_left, expected_left, atol=1e-12, rtol=0)

    expected_right = wp.builder.direct_product(psi, psi_result).data
    got_right = op.apply_from_right(rho.data, 0.0)
    assert_allclose(got_right, expected_right, atol=1e-12, rtol=0)


def test_negative_indices(grid):
    op_positive = wp.operator.FbrOperator1D(grid, 1, dummy_func)
    op_negative = wp.operator.FbrOperator1D(grid, -2, dummy_func)
    psi = wp.testing.random_state(grid, 47)
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
