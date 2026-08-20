import numpy as np
import pytest

import wavepacket as wp
from wavepacket.testing import assert_close


@pytest.fixture
def subspace_4d(grid_2d) -> list[wp.grid.State]:
    states = [wp.testing.random_state(grid_2d, seed) for seed in range(4)]
    return wp.orthonormalize(states)


def test_subspace_defined_by_basis(grid_1d, subspace_4d):
    with pytest.raises(Exception):
        wp.grid.SubspaceTransformation([])

    with pytest.raises(wp.BadGridError):
        s = subspace_4d + [wp.testing.random_state(grid_1d, 0)]
        wp.grid.SubspaceTransformation(s)

    with pytest.raises(wp.BadStateError):
        rho = wp.builder.pure_density(subspace_4d[0])
        s = [rho] + subspace_4d[1:]
        wp.grid.SubspaceTransformation(s)


def test_require_nonzero_norm(subspace_4d):
    with pytest.raises(wp.InvalidValueError):
        subspace_4d[0] = subspace_4d[0] * 0
        wp.grid.SubspaceTransformation(subspace_4d)


def test_grids(subspace_4d):
    transform = wp.grid.SubspaceTransformation(subspace_4d)

    assert transform.source_grid is subspace_4d[0].grid
    assert transform.target_grid.ndim == 1
    assert transform.target_grid.size == len(subspace_4d)


def test_reject_states_on_wrong_grid(subspace_4d):
    transform = wp.grid.SubspaceTransformation(subspace_4d)

    bad_state = wp.testing.random_state(transform.target_grid, 0)
    with pytest.raises(wp.BadGridError):
        transform.transform(bad_state)

    bad_state = wp.testing.random_state(transform.source_grid, 0)
    with pytest.raises(wp.BadGridError):
        transform.transform_back(bad_state)


def test_transform_states(subspace_4d):
    one_more_state = wp.testing.random_state(subspace_4d[0].grid, 42)
    orthogonal_state = wp.orthonormalize(subspace_4d + [one_more_state])[-1]

    transform = wp.grid.SubspaceTransformation(subspace_4d)

    psi = subspace_4d[0] + 1j * subspace_4d[2] + orthogonal_state
    result = transform.transform(psi)

    expected_psi = wp.grid.State(transform.target_grid, np.array([1, 0, 1j, 0]))
    assert_close(result, expected_psi, 1e-12)

    rho = wp.builder.pure_density(psi)
    result = transform.transform(rho)

    expected_rho = wp.builder.pure_density(expected_psi)
    assert_close(result, expected_rho, 1e-12)


def test_transform_states_back(subspace_4d):
    transform = wp.grid.SubspaceTransformation(subspace_4d)
    psi = wp.grid.State(transform.target_grid, np.array([1j, 0, 1, 0]))
    rho = wp.builder.pure_density(psi)

    result_psi = transform.transform_back(psi)
    expected_psi = 1j * subspace_4d[0] + subspace_4d[2]
    assert_close(result_psi, expected_psi, 1e-12)

    result_rho = transform.transform_back(rho)
    expected_rho = wp.builder.pure_density(expected_psi)
    assert_close(result_rho, expected_rho, 1e-12)
