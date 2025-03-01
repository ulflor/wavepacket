import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import wavepacket as wp
import wavepacket.typing as wpt


def dummy_func(data: wpt.RealData) -> wpt.RealData:
    return data


@pytest.fixture
def grid() -> wp.grid.Grid:
    return wp.grid.Grid([wp.grid.PlaneWaveDof(10, 20, 10),
                         wp.testing.DummyDof(np.ones(2), np.ones(2))])


@pytest.fixture
def op(grid) -> wp.operator.PlaneWaveFbrOperator:
    return wp.operator.PlaneWaveFbrOperator(grid, 0, dummy_func)


def test_reject_bad_constructor_args(grid):
    with pytest.raises(IndexError):
        wp.operator.PlaneWaveFbrOperator(grid, 2, dummy_func)

    with pytest.raises(wp.InvalidValueError):
        wp.operator.PlaneWaveFbrOperator(grid, 1, dummy_func)


def test_apply_to_data(grid, op):
    k = 3 * 2 * math.pi / 10.0
    psi = wp.builder.product_wave_function(grid, [wp.PlaneWave(k), dummy_func])
    rho = wp.builder.pure_density(psi)

    result = op.apply_to_wave_function(psi.data, 0.0)
    assert_allclose(result, psi.data * k, atol=1e-12, rtol=0)

    result = op.apply_from_left(rho.data, 0.0)
    expected = wp.builder.direct_product(k * psi, psi).data
    assert_allclose(result, expected, atol=1e-12, rtol=0)

    result = op.apply_from_right(rho.data, 0.0)
    expected = wp.builder.direct_product(psi, k * psi).data
    assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_negative_indices(grid):
    op_positive = wp.operator.PlaneWaveFbrOperator(grid, 0, dummy_func)
    op_negative = wp.operator.PlaneWaveFbrOperator(grid, -2, dummy_func)
    psi = wp.testing.random_state(grid, 42)
    rho = wp.builder.pure_density(psi)

    result_positive = op_positive.apply_to_wave_function(psi.data, 0.0)
    result_negative = op_negative.apply_to_wave_function(psi.data, 0.0)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)

    result_positive = op_positive.apply_from_left(rho.data, 0.0)
    result_negative = op_negative.apply_from_left(rho.data, 0.0)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)

    result_positive = op_positive.apply_from_right(rho.data, 00)
    result_negative = op_negative.apply_from_right(rho.data, 0.0)
    assert_allclose(result_positive, result_negative, atol=1e-12, rtol=0)
