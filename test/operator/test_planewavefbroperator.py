import math

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
    return wp.Grid([wp.PlaneWaveDof(10, 20, 10),
                    wp.testing.DummyDof(np.ones(2), np.ones(2))])


@pytest.fixture
def op(grid) -> wp.PlaneWaveFbrOperator:
    return wp.PlaneWaveFbrOperator(grid, 0, dummy_func)


def test_reject_bad_constructor_args(grid):
    with pytest.raises(IndexError):
        wp.PlaneWaveFbrOperator(grid, 2, dummy_func)

    with pytest.raises(wp.InvalidValueError):
        wp.PlaneWaveFbrOperator(grid, 1, dummy_func)


def test_apply_to_data(grid, op):
    k = 3 * 2 * math.pi / 10.0
    psi = wp.build_product_wave_function(grid, [wp.PlaneWave(k), dummy_func])
    rho = wp.pure_density(psi)

    result = op.apply_to_wave_function(psi.data)
    assert_allclose(result, psi.data * k, atol=1e-12)

    result = op.apply_from_left(rho.data)
    expected = wp.direct_product(k * psi, psi).data
    assert_allclose(result, expected, atol=1e-12)

    result = op.apply_from_right(rho.data)
    expected = wp.direct_product(psi, k * psi).data
    assert_allclose(result, expected, atol=1e-12)


def test_negative_indices(grid):
    op_positive = wp.PlaneWaveFbrOperator(grid, 0, dummy_func)
    op_negative = wp.PlaneWaveFbrOperator(grid, -2, dummy_func)
    psi = wp.testing.random_state(grid, 42)
    rho = wp.pure_density(psi)

    result_positive = op_positive.apply_to_wave_function(psi.data)
    result_negative = op_negative.apply_to_wave_function(psi.data)
    assert_allclose(result_positive, result_negative)

    result_positive = op_positive.apply_from_left(rho.data)
    result_negative = op_negative.apply_from_left(rho.data)
    assert_allclose(result_positive, result_negative, atol=1e-12)

    result_positive = op_positive.apply_from_right(rho.data)
    result_negative = op_negative.apply_from_right(rho.data)
    assert_allclose(result_positive, result_negative, atol=1e-12)
