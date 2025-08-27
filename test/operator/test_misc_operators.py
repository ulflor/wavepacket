import math
from typing import Tuple

import numpy as np
import wavepacket as wp

import pytest
from numpy.testing import assert_allclose
from wavepacket.testing import assert_close


def orthogonal_states(grid: wp.grid.Grid) -> Tuple[wp.grid.State, wp.grid.State]:
    assert len(grid.dofs) == 2
    lower_half = np.ones(grid.shape)
    lower_half[grid.dofs[0].size // 2:, :] = 0.

    s1 = wp.grid.State(grid, lower_half)
    s2 = wp.grid.State(grid, 1 - lower_half)

    return s1 / math.sqrt(wp.grid.trace(s1)), s2 / math.sqrt(wp.grid.trace(s2))


def test_projection_requires_normalizable_wave_function(grid_1d):
    bad_state = wp.builder.unit_density(grid_1d)
    zero_state = wp.builder.zero_wave_function(grid_1d)

    with pytest.raises(wp.BadStateError):
        wp.operator.Projection(bad_state)

    with pytest.raises(wp.BadStateError):
        wp.operator.Projection(zero_state)


def test_project_wave_function(grid_2d):
    p, q = orthogonal_states(grid_2d)
    projection = wp.operator.Projection(p)

    psi = 0.7 * p + 0.8 * q
    result = projection.apply_to_wave_function(psi.data, 0.0)
    assert_allclose(result, 0.7 * p.data, rtol=0, atol=1e-12)


def test_normalize_wave_function(grid_1d):
    p = wp.testing.random_state(grid_1d, 42)
    projection = wp.operator.Projection(p)
    other_projection = wp.operator.Projection(2 * p)

    psi = wp.testing.random_state(grid_1d, 43)
    assert_close(projection.apply(psi, 0), other_projection.apply(psi, 0), 1e-12)


def test_project_density_operator(grid_1d):
    p = wp.testing.random_state(grid_1d, 42)
    projection = wp.operator.Projection(p)

    psi = wp.testing.random_state(grid_1d, 43)
    projected = projection.apply(psi, 0.0)
    rho = wp.builder.pure_density(psi)

    result_left = projection.apply_from_left(rho.data, 0.0)
    expected_left = wp.builder.direct_product(projected, psi).data
    assert_allclose(result_left, expected_left, rtol=0, atol=1e-12)

    result_right = projection.apply_from_right(rho.data, 0.0)
    expected_right = wp.builder.direct_product(psi, projected).data
    assert_allclose(result_right, expected_right, rtol=0, atol=1e-12)
