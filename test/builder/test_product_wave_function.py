import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import wavepacket as wp
import wavepacket.typing as wpt


def generator(dvr_points: wpt.RealData) -> wpt.ComplexData:
    return 1j * dvr_points


def zero_generator(dvr_points: wpt.RealData) -> wpt.ComplexData:
    return np.zeros(dvr_points.shape)


def test_reject_wrong_number_of_inputs(grid_2d):
    with pytest.raises(wp.InvalidValueError):
        wp.builder.product_wave_function(grid_2d, generator)


def test_build_product_state(grid_2d):
    result = wp.builder.product_wave_function(grid_2d, [generator, generator], normalize=False)

    expected_data = np.einsum("i, j -> ij",
                              generator(grid_2d.dofs[0].dvr_points), generator(grid_2d.dofs[1].dvr_points))
    scaling = result.data.flat[0] / expected_data.flat[0]
    assert result.grid == grid_2d
    assert_allclose(result.data, expected_data * scaling, atol=1e-12, rtol=0)

    # and again with normalization
    normalized_result = wp.builder.product_wave_function(grid_2d, [generator, generator])

    scaling = normalized_result.data.flat[0] / expected_data.flat[0]
    assert_allclose(normalized_result.data, expected_data * scaling, atol=1e-14, rtol=0)
    assert_allclose(wp.grid.trace(normalized_result), 1.0, atol=1e-12, rtol=0)


def test_handle_zero_norm_gracefully(grid_2d):
    result = wp.builder.product_wave_function(grid_2d, [zero_generator, zero_generator])

    assert_array_equal(result.data, np.zeros(grid_2d.shape))
