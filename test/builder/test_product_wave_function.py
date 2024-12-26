import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import wavepacket as wp
import wavepacket.typing as wpt


@pytest.fixture
def grid() -> wp.Grid:
    return wp.Grid([wp.PlaneWaveDof(1, 5, 10), wp.PlaneWaveDof(10, 13, 3)])


def generator(dvr_points: wpt.RealData) -> wpt.ComplexData:
    return 1j * dvr_points


def zero_generator(dvr_points: wpt.RealData) -> wpt.ComplexData:
    return np.zeros(dvr_points.shape)


def test_reject_wrong_number_of_inputs(grid):
    with pytest.raises(wp.InvalidValueError):
        wp.build_product_wave_function(grid, generator)


def test_build_product_state(grid):
    result = wp.build_product_wave_function(grid, [generator, generator], normalize=False)

    expected_data = np.einsum("i, j -> ij",
                              generator(grid.dofs[0].dvr_points), generator(grid.dofs[1].dvr_points))
    scaling = result.data.flat[0] / expected_data.flat[0]
    assert result.grid == grid
    assert_allclose(result.data, expected_data * scaling, atol=1e-12, rtol=0)

    # and again with normalization
    normalized_result = wp.build_product_wave_function(grid, [generator, generator])

    scaling = normalized_result.data.flat[0] / expected_data.flat[0]
    assert_allclose(normalized_result.data, expected_data * scaling, atol=1e-14, rtol=0)
    assert_allclose(wp.trace(normalized_result), 1.0, atol=1e-12, rtol=0)


def test_handle_zero_norm_gracefully(grid):
    result = wp.build_product_wave_function(grid, [zero_generator, zero_generator])

    assert_array_equal(result.data, np.zeros(grid.shape))
