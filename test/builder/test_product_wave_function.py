import numpy as np
import pytest
import wavepacket as wp

from numpy.testing import assert_allclose
from wavepacket.typing import ComplexData, RealData


@pytest.fixture
def grid() -> wp.Grid:
    return wp.Grid([wp.PlaneWaveDof(1, 5, 10), wp.PlaneWaveDof(10, 13, 3)])


def generator(dvr_points: RealData) -> ComplexData:
    return 1j * dvr_points


def zero_generator(dvr_points: RealData) -> ComplexData:
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
    assert_allclose(result.data, expected_data * scaling, atol=1e-12)

    # and again with normalization
    normalized_result = wp.build_product_wave_function(grid, [generator, generator])

    scaling = normalized_result.data.flat[0] / expected_data.flat[0]
    assert_allclose(normalized_result.data, expected_data * scaling, atol=1e-14)
    assert_allclose(wp.trace(normalized_result), 1.0, 1e-12)


def test_handle_zero_norm_gracefully(grid):
    result = wp.build_product_wave_function(grid, [zero_generator, zero_generator])

    assert_allclose(result.data, np.zeros(grid.shape))
