import numpy as np
import pytest
from numpy.testing import assert_array_equal

import wavepacket as wp


def grid_with_channel_dof() -> wp.grid.Grid:
    dof1 = wp.grid.PlaneWaveDof(1, 2, 5)
    channel_dof = wp.grid.ChannelDof(2)
    dof2 = wp.grid.SphericalHarmonicsDof(4, 1)

    return wp.grid.Grid([dof1, channel_dof, dof2])


def test_require_grid_with_channel_dof_and_more(grid_1d):
    with pytest.raises(wp.BadGridError):
        wp.grid.ChannelProjectionTransformation(grid_1d)

    with pytest.raises(wp.BadGridError):
        grid = wp.grid.Grid(wp.grid.ChannelDof(3))
        wp.grid.ChannelProjectionTransformation(grid)


def test_target_grid_has_no_channel_dof():
    grid = grid_with_channel_dof()
    trafo = wp.grid.ChannelProjectionTransformation(grid)

    assert isinstance(grid.dofs[1], wp.grid.ChannelDof)
    assert trafo.target_grid.dofs == [grid.dofs[0], grid.dofs[2]]


def test_transformation_requires_valid_input(grid_1d):
    grid = grid_with_channel_dof()
    trafo = wp.grid.ChannelProjectionTransformation(grid)
    psi = wp.testing.random_state(grid, 1)

    with pytest.raises(wp.BadGridError):
        bad_grid_state = wp.testing.random_state(grid_1d, 1)
        trafo.transform(bad_grid_state, channel=0)

    with pytest.raises(wp.BadStateError):
        invalid_state = wp.grid.State(grid, np.zeros((5, 5)))
        trafo.transform(invalid_state, channel=0)

    with pytest.raises(wp.BadFunctionCall):
        # channel is required
        trafo.transform(psi)

    with pytest.raises(wp.InvalidValueError):
        trafo.transform(psi, channel="not_exist")


def test_transform_states():
    grid = grid_with_channel_dof()
    trafo = wp.grid.ChannelProjectionTransformation(grid)

    psi = wp.testing.random_state(grid, 1)
    rho = wp.builder.pure_density(psi)

    trafo_psi = trafo.transform(psi, channel=1)
    trafo_rho = trafo.transform(rho, channel=1)

    assert trafo_psi.grid is trafo.target_grid
    assert trafo_psi.is_wave_function()
    assert_array_equal(trafo_psi.data.flat, psi.data[:, 1, :].flat)

    assert trafo_rho.grid is trafo.target_grid
    assert trafo_rho.is_density_operator()
    assert_array_equal(trafo_rho.data.flat, rho.data[:, 1, :, :, 1, :].flat)

    # to avoid test that incorrectly align with the code
    projector = wp.operator.Channel(grid, 1)

    expected_pop = wp.expectation_value(projector, psi)
    projected_pop = wp.trace(trafo_psi)
    projected_rho_pop = wp.trace(trafo_rho)

    assert abs(expected_pop - projected_pop) < 1e-12
    assert abs(expected_pop - projected_rho_pop) < 1e-12
